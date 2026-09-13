"""WebSocket event handling.

The handler was a closure inside run() that reached into a dozen locals. It now takes
the runtime container and delegates the decision step, so the data flow is explicit.
"""
import datetime

from ..core.domain import Event
from ..core.price_state import update_price


async def handle_ws_event(runtime, event, evaluate_tick):
    runtime.feeds.ws_events += 1
    symbol = event['symbol']
    event_now = int(event.get('event_time', 0) or 0)
    audit_gap = runtime.settings.market_event_audit_ms
    if event_now - runtime.feeds.last_event_audit.get(symbol, 0) >= audit_gap:
        runtime.event_writer.submit_nowait(Event('market_event', event).json())
        runtime.feeds.last_event_audit[symbol] = event_now

    # Trade prints and liquidations accumulate here rather than in the database: this runs
    # on every print of every symbol and must not block on a write.
    flow = getattr(runtime, 'flow', None)
    if flow is not None:
        flow.observe(event)
    item = runtime.cache.latest.setdefault(symbol, {'symbol': symbol})
    if not update_price(item, event):
        return
    item['updated'] = datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S.%f')[:-3]
    # Observed from the socket, with the exchange's own event time. The bar loop alone
    # left every symbol looking stale for most of a bar and rejected most entries.
    runtime.guard.observe(symbol, event_now)

    if event.get('type') == 'kline':
        candle = {**event, 'is_closed': int(bool(event.get('is_closed')))}
        runtime.cache.live_candles[symbol] = candle
        runtime.candle_writer.submit_nowait((symbol, runtime.settings.interval, [candle], event_now))
        if candle['is_closed']:
            _merge_closed_candle(runtime, symbol, candle)

    session = runtime.session
    if event.get('type') == 'book':
        # Proof this symbol's book stream is live. Some actively traded contracts never
        # publish bookTicker at all, and without this the session keeps a position slot
        # on a symbol that cannot fill anything.
        session.note_book(symbol, event_now or None)
    if event.get('type') == 'book' and session.status == 'running':
        from ..trading.execution import settle_order
        from .decision_loop import _bar_volatility_bps

        # A fill that happens on a book update is a fill that crossed the spread at a
        # moment the market was moving. The latency term needs to know how much it moves,
        # so the same measurement the sizing path uses is passed through to the fill model.
        volatility_bps = _bar_volatility_bps(runtime, symbol)
        for order in list(runtime.broker.open_orders(symbol)):
            settle_order(runtime, order.order_id, volatility_bps=volatility_bps)

    if event.get('type') in ('mark_price', 'trade', 'book') and session.status == 'running':
        price = event.get('mark_price') or event.get('price') or ((event.get('bid', 0) + event.get('ask', 0)) / 2)
        if price:
            if event['type'] == 'mark_price':
                runtime.feeds.last_mark_event[symbol] = event['event_time']
                # The venue mark price, and not a bar boundary.
                session.mark({symbol: price}, mark_prices={symbol: price}, record=False)
            runtime.cache.latest_rows[symbol] = {'price': float(price),
                                                 'event_time': event.get('event_time', 0),
                                                 'feed': event.get('type')}
            await evaluate_tick(runtime, symbol, price)


def _merge_closed_candle(runtime, symbol, candle):
    cached = runtime.cache.closed_rows.get(symbol, [])
    by_time = {row['open_time']: row for row in cached}
    by_time[candle['open_time']] = candle
    lookback = runtime.settings.lookback
    runtime.cache.closed_rows[symbol] = [by_time[key] for key in sorted(by_time)][-lookback:]
