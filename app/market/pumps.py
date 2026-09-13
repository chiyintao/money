"""Background market-data pumps: position marking, price and mark snapshots."""
import asyncio
import time

from ..core import app_keys
from ..core.price_state import update_price

# How far the venue's own timestamp may sit from our clock before a mark price is refused.
# This bounds clock disagreement, not transport time: the two are unrelated, and conflating
# them is what silently disabled the whole funding feature family. Observed skew on the
# venue is a few seconds in either direction.
MARK_STAMP_SKEW_MS = 60_000


async def realtime_pump(runtime, app):
    """Push the lean realtime payload to websocket subscribers."""
    while True:
        await asyncio.sleep(0.1)
        await app[app_keys.BROADCAST]()


async def mark_pump(runtime):
    """Refresh mark prices; drives unrealized PnL and liquidation checks."""
    pending = {}
    last_write_ms = 0
    while True:
        try:
            # Measured before the request, not after it. The venue's 'time' field is when
            # *it* built the response, so comparing it against a clock read after the
            # response arrives measures the round trip, not the age of the data. On this
            # network /fapi/v1/premiumIndex (900 symbols) takes ~5.1s, which sat just
            # outside the old 5s window -- so every row was discarded, pending stayed
            # empty, and the derivatives table had not been written for 47 hours. Funding
            # has four features built from that table and all four were permanently
            # degraded, which vetoed every decision and left the service at 100% FLAT.
            #
            # The bound is a clock-skew tolerance now, so it is deliberately loose and
            # independent of how slow the link is. A venue stamp far in the future is
            # still rejected: that is skew large enough to be a real problem.
            requested_at = int(time.time() * 1000)
            marks = await runtime.api.mark_snapshot()
            now_ms = int(time.time() * 1000)
            for row in marks:
                symbol = row['symbol']
                if symbol not in runtime.cache.latest and symbol not in runtime.account.positions:
                    continue
                stamp = int(row.get('time', 0))
                if abs(stamp - requested_at) > MARK_STAMP_SKEW_MS:
                    continue
                item = runtime.cache.latest.setdefault(symbol, {'symbol': symbol})
                if stamp <= item.get('mark_time', 0):
                    continue
                event = {'type': 'mark_price', 'mark_price': float(row['markPrice']), 'event_time': stamp,
                         'funding_rate': float(row.get('lastFundingRate', 0)), 'source': 'rest-premium-index'}
                if update_price(item, event):
                    runtime.feeds.last_mark_event[symbol] = stamp
                    # Persist the mark price and funding rate. Nothing ever wrote a
                    # derivatives row from the live loop, so the funding series the feature
                    # builder reads only ever contained whatever the backfill CLI had
                    # loaded, and it stopped advancing the day the backfill was run.
                    pending.setdefault(symbol, {'symbol': symbol, 'event_time': stamp,
                                                'mark_price': event['mark_price'],
                                                'funding_rate': event['funding_rate'],
                                                'open_interest': 0.0})
                    if runtime.session.status == 'running':
                        # mark_price is the venue's mark, not a last trade: it feeds
                        # funding and liquidation, and it is not a bar boundary, so it
                        # must not append an equity sample.
                        runtime.session.mark({symbol: event['mark_price']},
                                             mark_prices={symbol: event['mark_price']},
                                             record=False)
            # One write per minute is far more than funding needs (it settles every eight
            # hours) and cheap enough not to matter, while keeping the series current enough
            # for mark_basis to mean something.
            if pending and now_ms - last_write_ms >= 60_000:
                try:
                    runtime.store.record_derivatives(list(pending.values()))
                    pending.clear()
                    last_write_ms = now_ms
                    runtime.feature_source().invalidate()
                except Exception as exc:
                    runtime.errors.note('derivatives_write', exc)
        except Exception as exc:
            runtime.errors.note('mark_pump', exc)
        else:
            runtime.errors.ok('mark_pump')
        await asyncio.sleep(2)


async def price_pump(runtime):
    """Fallback price feed so decisions keep flowing when the websocket stalls."""
    while True:
        try:
            requested_at = int(time.time() * 1000)
            prices = await runtime.api.price_snapshot()
            for row in runtime.cache.market.get('all', []):
                symbol = row.get('symbol')
                price = prices.get(symbol)
                if not price:
                    continue
                row['price'] = price
                update_price(runtime.cache.latest.setdefault(symbol, {'symbol': symbol}),
                             {'type': 'trade', 'price': price, 'event_time': requested_at, 'source': 'rest-1s'})
            runtime.errors.ok('price_pump')
        except Exception as exc:
            runtime.errors.note('price_pump', exc)
        await asyncio.sleep(1.0)
