import asyncio, json, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import aiohttp

# Fallbacks for a contract whose filters are missing or unparsable. They match the
# exchange's own defaults closely enough that an order still rounds to something
# tradeable rather than being silently rejected.
class _RateLimited(Exception):
    """Raised instead of silently retrying when the venue answers 418/429."""

    def __init__(self, status, retry_after_s=5.0):
        super().__init__('rate_limited:%s' % status)
        self.status = status
        self.retry_after_s = retry_after_s


DEFAULT_TICK_SIZE = .01
DEFAULT_STEP_SIZE = .001
DEFAULT_MIN_NOTIONAL = 5.0


def _filter_value(contract, kind, name, default):
    for item in contract.get('filters') or []:
        if item.get('filterType') != kind:
            continue
        try:
            value = float(item.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default
    return default


def build_contract_specs(info):
    """Per-symbol ContractSpec built from the exchange's own filters.

    Every symbol used to share one hardcoded spec (tick 0.01, step 0.001, minNotional
    5). That cannot be right at both ends of the universe: a contract quoted at 0.0006
    and one quoted at 78000 do not round quantity or price the same way, and the
    rounding is what decides whether an order is tradeable at all.
    """
    from ..core.instruments import ContractSpec

    specs = {}
    for contract in (info or {}).get('symbols') or []:
        symbol = contract.get('symbol')
        if not symbol or contract.get('contractType') != 'PERPETUAL':
            continue
        if contract.get('quoteAsset') != 'USDT' or contract.get('status') != 'TRADING':
            continue
        # MARKET_LOT_SIZE is the step that applies to the market orders this service
        # submits; LOT_SIZE is the general one and can be coarser.
        step = _filter_value(contract, 'MARKET_LOT_SIZE', 'stepSize', 0) or             _filter_value(contract, 'LOT_SIZE', 'stepSize', DEFAULT_STEP_SIZE)
        specs[symbol] = ContractSpec(
            symbol,
            tick_size=_filter_value(contract, 'PRICE_FILTER', 'tickSize', DEFAULT_TICK_SIZE),
            step_size=step,
            min_notional=_filter_value(contract, 'MIN_NOTIONAL', 'notional', DEFAULT_MIN_NOTIONAL))
    return specs

def _kline_rows(data):
    """Raw kline arrays into candle dicts, keeping the order-flow fields.

    The live loop writes these straight into the candles table, so dropping fields 7-9 here
    is what made the order-flow columns permanently empty for every bar the service saw.
    """
    import time as _time

    now = int(_time.time() * 1000)
    rows = []
    for item in data or []:
        try:
            close_time = int(item[6])
            rows.append({'open_time': int(item[0]), 'open': float(item[1]), 'high': float(item[2]),
                         'low': float(item[3]), 'close': float(item[4]), 'volume': float(item[5]),
                         'close_time': close_time, 'is_closed': int(close_time) <= now,
                         'quote_volume': float(item[7]) if len(item) > 7 else 0.0,
                         'trades': int(float(item[8])) if len(item) > 8 else 0,
                         'taker_buy_volume': float(item[9]) if len(item) > 9 else 0.0})
        except (IndexError, TypeError, ValueError):
            continue
    return rows


class BinancePublic:
    def __init__(self, base_url, data_dir='data'):
        self.base_url=base_url.rstrip('/'); self.data_dir=Path(data_dir); self.data_dir.mkdir(parents=True,exist_ok=True); self.session=None
        self.used_weight_1m=0; self.last_rate_limit=None; self._funding_intervals={}; self._leverage_brackets={}
    async def _urllib_get(self,path,params=None):
        query=('?'+urlencode(params)) if params else ''
        def fetch():
            request=Request(self.base_url+path+query,headers={'User-Agent':'paper-research-agent/1.0'})
            with urlopen(request,timeout=30) as response: return json.loads(response.read().decode('utf-8'))
        return await asyncio.to_thread(fetch)
    async def _get(self,session,path,params=None):
        try:
            async with session.get(self.base_url+path,params=params,timeout=20) as response:
                # The used-weight header is the only advance warning the venue gives before
                # it starts refusing requests. It used to be discarded, so the first sign of
                # a rate-limit problem was a 429 that nothing retried.
                used = response.headers.get('X-MBX-USED-WEIGHT-1M')
                if used:
                    self.used_weight_1m = int(float(used))
                if response.status in (418, 429):
                    retry = float(response.headers.get('Retry-After', 0) or 0)
                    self.last_rate_limit = {'status': response.status, 'at': int(time.time()*1000),
                                            'retry_after_s': retry}
                    raise _RateLimited(response.status, retry or 5.0)
                response.raise_for_status(); return await response.json()
        except _RateLimited:
            # A rate-limit answer is not a network failure; the urllib fallback would hit
            # the same wall from the same IP and burn the budget twice.
            raise
        except aiohttp.ClientResponseError:
            # The venue answered. It answered with a refusal, which is information: the
            # request reached it, so re-asking over urllib reaches the same venue from the
            # same IP and gets the same refusal, only slower. An unauthenticated
            # /fapi/v1/leverageBracket returns 401 every time, and the fallback retried it
            # anyway -- turning an instant, correct 'no' into a 30s timeout at every start.
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return await self._urllib_get(path,params)
    async def market_snapshot(self):
        if self.session is None or self.session.closed:
            self.session=aiohttp.ClientSession(trust_env=True)
        session=self.session
        info,tickers=await asyncio.gather(self._get(session,'/fapi/v1/exchangeInfo'),self._get(session,'/fapi/v1/ticker/24hr'))
        try: premium=await self._get(session,'/fapi/v1/premiumIndex')
        except Exception: premium=[]
        contracts={x['symbol']:x for x in info['symbols'] if x.get('contractType')=='PERPETUAL' and x.get('quoteAsset')=='USDT' and x.get('status')=='TRADING'}
        funding={x.get('symbol'):x for x in premium}; rows=[]
        for ticker in tickers:
            contract=contracts.get(ticker.get('symbol'))
            if not contract: continue
            extra=funding.get(ticker['symbol'],{})
            rows.append({
                'symbol': ticker['symbol'],
                'price': float(ticker.get('lastPrice', 0)),
                'change': float(ticker.get('priceChangePercent', 0)),
                'volume': float(ticker.get('quoteVolume', 0)),
                'high': float(ticker.get('highPrice', 0)),
                'low': float(ticker.get('lowPrice', 0)),
                'count': int(ticker.get('count', 0)),
                'open_time': int(contract.get('onboardDate', 0)),
                'mark_price': float(extra.get('markPrice', ticker.get('lastPrice', 0))),
                'index_price': float(extra.get('indexPrice', ticker.get('lastPrice', 0))),
                'funding_rate': float(extra.get('lastFundingRate', 0) or 0),
                'next_funding_time': int(extra.get('nextFundingTime', 0) or 0),
                'interest_rate': float(extra.get('interestRate', 0) or 0),
                'estimated_settle_price': float(extra.get('estimatedSettlePrice', 0) or 0),
            })
        # exchangeInfo is already fetched above, so the contract filters ride along at no
        # extra request. Callers pop this key before caching the rest: it is a broker
        # input, not part of the dashboard payload.
        return {'all':rows,'specs':build_contract_specs(info),'hot':sorted(rows,key=lambda x:x['volume'],reverse=True)[:20],'gainers':sorted(rows,key=lambda x:x['change'],reverse=True)[:20],'losers':sorted(rows,key=lambda x:x['change'])[:20],'new':sorted(rows,key=lambda x:x['open_time'],reverse=True)[:20]}

    async def leverage_brackets(self):
        """Per-symbol maintenance brackets from the venue, as MaintenanceTier tuples.

        Falls back to the published defaults for any symbol the venue does not answer
        for, so a partial response degrades the model rather than removing it.
        """
        from ..core.instruments import MaintenanceTier

        if self.session is None or self.session.closed:
            self.session=aiohttp.ClientSession(trust_env=True)
        try:
            rows=await self._get(self.session,'/fapi/v1/leverageBracket')
        except Exception:
            return {}
        brackets={}
        for row in rows or []:
            symbol=row.get('symbol')
            if not symbol:
                continue
            tiers=[]
            for item in row.get('brackets') or []:
                try:
                    tiers.append(MaintenanceTier(float(item.get('notionalCap', float('inf'))),
                                                 float(item['maintMarginRatio']),
                                                 float(item.get('cum', 0) or 0)))
                except (KeyError, TypeError, ValueError):
                    continue
            if tiers:
                brackets[str(symbol)]=tuple(tiers)
        self._leverage_brackets=brackets
        return brackets

    async def funding_intervals(self):
        """Per-symbol settlement window in milliseconds, from the venue itself.

        Most perpetuals settle every eight hours, but the venue runs four-hour and
        one-hour funding on a minority. Hardcoding eight hours over-charged the short
        ones and under-charged nothing, so the paper account paid funding less often
        than the contract it was imitating.
        """
        if self.session is None or self.session.closed:
            self.session=aiohttp.ClientSession(trust_env=True)
        try:
            rows=await self._get(self.session,'/fapi/v1/fundingInfo')
        except Exception:
            rows=[]
        intervals={}
        for row in rows or []:
            try:
                hours=float(row.get('fundingIntervalHours') or 8)
            except (TypeError, ValueError):
                hours=8.0
            if hours>0:
                intervals[str(row.get('symbol'))]=int(hours*3600*1000)
        self._funding_intervals=intervals
        return intervals

    async def contract_specs(self):
        """Per-symbol ContractSpec, for the startup fetch before the first snapshot."""
        if self.session is None or self.session.closed:
            self.session=aiohttp.ClientSession(trust_env=True)
        return build_contract_specs(await self._get(self.session,'/fapi/v1/exchangeInfo'))
    async def price_snapshot(self):
        if self.session is None or self.session.closed: self.session=aiohttp.ClientSession(trust_env=True)
        rows=await self._get(self.session,'/fapi/v1/ticker/price')
        return {str(x.get('symbol')):float(x.get('price',0) or 0) for x in rows if x.get('symbol')}

    async def klines(self,symbol,interval='1m',limit=250):
        if self.session is None or self.session.closed: self.session=aiohttp.ClientSession(trust_env=True)
        data=await self._get(self.session,'/fapi/v1/klines',{'symbol':symbol,'interval':interval,'limit':limit})
        return _kline_rows(data)
    async def mark_snapshot(self):
        if self.session is None or self.session.closed:
            self.session=aiohttp.ClientSession(trust_env=True)
        return await self._get(self.session,'/fapi/v1/premiumIndex')

    async def open_interest(self,symbol):
        async with aiohttp.ClientSession(trust_env=True) as session: return await self._get(session,'/fapi/v1/openInterest',{'symbol':symbol})
