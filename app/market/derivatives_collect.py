"""Collection of the derivative statistics that expire.

The venue publishes open interest, long/short ratios and taker volume ratios on a rolling
thirty-day window and then discards them. There is no archive to backfill from and no
historical endpoint: whatever is not collected today cannot be recovered, at any price.
That makes this the one data problem in the system with a hard deadline, and the
derivatives table it feeds was measured to hold a column of zeros.

Two entry points:

* :func:`collect_once` polls every endpoint for every symbol once and returns the rows.
* :class:`DerivativesCollector` runs that on a timer from the decision loop.

Endpoint shapes differ enough that each gets its own parser rather than one generic
projection, because a silently mis-parsed ratio is worse than a missing one.
"""
import asyncio
import time

# The venue's own interval for these endpoints. It is not configurable: the stored
# derivatives_detail table is keyed by this granularity, and mixing granularities in one
# table is what makes a window average silently span the wrong amount of time.
DEFAULT_PERIOD = '5m'
# How many points to ask for. 30 is the venue's cap on most of these endpoints and covers
# two and a half hours at 5m, which is enough to close a gap after a short outage.
DEFAULT_LIMIT = 30


def parse_open_interest(payload):
    """Rows from /futures/data/openInterestHist."""
    rows = []
    for item in payload or []:
        try:
            rows.append({'event_time': int(item['timestamp']),
                         'open_interest': float(item.get('sumOpenInterest', 0) or 0),
                         'open_interest_value': float(item.get('sumOpenInterestValue', 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def parse_long_short_ratio(payload, key='longShortRatio'):
    """Rows from the topLongShortPositionRatio / topLongShortAccountRatio family."""
    rows = []
    for item in payload or []:
        try:
            rows.append({'event_time': int(item['timestamp']),
                         'long_short_ratio': float(item.get(key, 0) or 0),
                         'long_account': float(item.get('longAccount', 0) or 0),
                         'short_account': float(item.get('shortAccount', 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def parse_taker_ratio(payload):
    """Rows from /futures/data/takerlongshortRatio."""
    rows = []
    for item in payload or []:
        try:
            rows.append({'event_time': int(item['timestamp']),
                         'buy_sell_ratio': float(item.get('buySellRatio', 0) or 0),
                         'buy_volume': float(item.get('buyVol', 0) or 0),
                         'sell_volume': float(item.get('sellVol', 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return rows


# name -> (path, parser, key used to merge multiple series onto one timestamp)
ENDPOINTS = (
    ('open_interest', '/futures/data/openInterestHist', parse_open_interest, None),
    ('top_position_ratio', '/futures/data/topLongShortPositionRatio', parse_long_short_ratio, None),
    ('top_account_ratio', '/futures/data/topLongShortAccountRatio', parse_long_short_ratio, None),
    ('global_account_ratio', '/futures/data/globalLongShortAccountRatio', parse_long_short_ratio, None),
    ('taker_ratio', '/futures/data/takerlongshortRatio', parse_taker_ratio, None),
)


def merge_series(series_by_name):
    """Join the per-endpoint series into one row per timestamp, for storage.

    The derivatives table has a fixed column set, so the five endpoints are projected onto
    it. Each endpoint is optional: a missing one contributes nothing rather than zeroing
    the others, so a partial collection is visible as a missing column and not as a
    genuine reading of zero.
    """
    merged = {}
    for name, rows in series_by_name.items():
        for row in rows:
            merged.setdefault(int(row['event_time']), {})[name] = row
    out = []
    for event_time in sorted(merged):
        fields = merged[event_time]
        out.append({
            'event_time': event_time,
            'open_interest': _pick(fields, 'open_interest', 'open_interest'),
            'open_interest_value': _pick(fields, 'open_interest', 'open_interest_value'),
            'long_short_ratio': _pick(fields, 'top_position_ratio', 'long_short_ratio'),
            'top_account_ratio': _pick(fields, 'top_account_ratio', 'long_short_ratio'),
            'global_account_ratio': _pick(fields, 'global_account_ratio', 'long_short_ratio'),
            'taker_buy_sell_ratio': _pick(fields, 'taker_ratio', 'buy_sell_ratio'),
            'taker_buy_volume': _pick(fields, 'taker_ratio', 'buy_volume'),
            'taker_sell_volume': _pick(fields, 'taker_ratio', 'sell_volume'),
        })
    return out


def _pick(fields, name, key):
    row = fields.get(name)
    if not row:
        return None
    value = row.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


class DerivativesCollector:
    """Polls the expiring endpoints on a timer and writes what it gets.

    Failures are counted per endpoint and never raise: a collector that stops the decision
    loop on a rate limit would trade one kind of data loss for another.
    """

    def __init__(self, api, store, period=DEFAULT_PERIOD, limit=DEFAULT_LIMIT,
                 interval_seconds=300.0, symbols=None):
        self.api = api
        self.store = store
        self.period = period
        self.limit = int(limit)
        self.interval_seconds = float(interval_seconds)
        self.symbols = list(symbols or ())
        self.last_run_ms = 0
        self.stats = {'runs': 0, 'rows': 0, 'empty': 0, 'failed': 0, 'last_error': None,
                      'last_rows': 0, 'last_run_at': None}

    def due(self, now_ms=None):
        now = float(now_ms if now_ms is not None else time.time() * 1000)
        return (now - self.last_run_ms) >= self.interval_seconds * 1000

    async def run(self, symbols=None):
        universe = list(symbols or self.symbols)
        if not universe:
            return 0
        total = 0
        for symbol in universe:
            try:
                total += await self.collect_symbol(symbol)
            except Exception as exc:
                self.stats['failed'] += 1
                self.stats['last_error'] = '%s: %r' % (symbol, exc)
        self.last_run_ms = time.time() * 1000
        self.stats['runs'] += 1
        self.stats['rows'] += total
        self.stats['last_rows'] = total
        self.stats['last_run_at'] = int(self.last_run_ms)
        if not total:
            self.stats['empty'] += 1
        return total

    async def collect_symbol(self, symbol, session=None):
        session = session or getattr(self.api, 'session', None)
        if session is None or getattr(session, 'closed', True):
            await self.api.mark_snapshot()  # lazily builds the shared aiohttp session
            session = self.api.session
        params = {'symbol': symbol, 'period': self.period, 'limit': self.limit}
        results = await asyncio.gather(
            *(self.api._get(session, path, dict(params)) for _name, path, _parser, _key in ENDPOINTS),
            return_exceptions=True)
        series = {}
        for (name, _path, parser, _key), payload in zip(ENDPOINTS, results):
            if isinstance(payload, Exception) or payload is None:
                continue
            rows = parser(payload)
            if rows:
                series[name] = rows
        rows = merge_series(series)
        if not rows:
            return 0
        stored = [{'symbol': symbol,
                   'event_time': row['event_time'],
                   'open_interest': row['open_interest'] or 0.0,
                   'funding_rate': 0.0,
                   'mark_price': 0.0,
                   **{key: value for key, value in row.items() if key != 'event_time'}}
                  for row in rows]
        self.store.record_derivatives_detail(stored)
        return len(stored)

    def health(self):
        return {**self.stats, 'period': self.period, 'symbols': len(self.symbols),
                'interval_seconds': self.interval_seconds, 'due': self.due()}
