"""Keep trade, quote and mark prices separate with per-source timestamps."""
import math
import time

# Exchanges stamp events with their own clock, which can run ahead of the host.
# A future timestamp is not stale data, so a small forward skew is tolerated
# instead of silently rejecting every order for being "stale".
FUTURE_SKEW_MS = 3000


def event_age_ms(timestamp, now_ms=None):
    """Age of an exchange timestamp, allowing for clock skew between hosts."""
    if not timestamp:
        return None
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return now_ms - int(timestamp)


def execution_ready(item, now_ms, max_age_ms=5000, max_future_ms=FUTURE_SKEW_MS):
    """Whether the book and mark data are fresh enough to fill an order.

    Returns (ok, reason). A negative age means the exchange clock is ahead of this
    host; that is a skew, not staleness, so it is accepted within max_future_ms.
    """
    for field in ('book_time', 'mark_time'):
        age = event_age_ms(item.get(field), now_ms)
        if age is None:
            return False, 'missing_' + field
        if age > max_age_ms:
            return False, 'stale_' + field
        if age < -max_future_ms:
            return False, 'future_' + field
    bid, ask, mark = (float(item.get(k, 0)) for k in ('bid', 'ask', 'mark_price'))
    if not all(math.isfinite(v) and v > 0 for v in (bid, ask, mark)) or bid > ask:
        return False, 'invalid_execution_prices'
    return True, 'fresh_execution_prices'


def update_price(item, event):
    kind = event['type']
    timestamp = int(event.get('event_time', 0))
    key = 'book_time' if kind == 'book' else 'mark_time' if kind == 'mark_price' else 'price_time'
    if timestamp <= 0 or timestamp < item.get(key, 0):
        return False
    if kind == 'book':
        bid, ask = float(event['bid']), float(event['ask'])
        if not all(math.isfinite(v) and v > 0 for v in (bid, ask)) or bid > ask:
            return False
        values = {k: event[k] for k in ('bid', 'ask', 'bid_qty', 'ask_qty', 'bids', 'asks') if k in event}
        values['spread'] = ask - bid
    else:
        value = float(event.get('mark_price') if kind == 'mark_price' else event.get('close', event.get('price', 0)))
        if not math.isfinite(value) or value <= 0:
            return False
        values = {'mark_price' if kind == 'mark_price' else 'price': value}
        if kind == 'mark_price' and 'funding_rate' in event:
            values['funding_rate'] = event['funding_rate']
    item.update(values)
    item[key] = timestamp
    item['event_time'] = max(timestamp, item.get('event_time', 0))
    item['feed'] = event.get('source', 'websocket')
    return True
