import hashlib
import json

from ..core.engine import MarketEvent


class FeedNormalizer:
    """Assign monotonic sequence numbers and drop duplicate exchange events."""

    def __init__(self, source='binance.websocket', max_seen=100000):
        self.source = source
        self.max_seen = int(max_seen)
        self.sequence = 0
        self.seen = set()
        self.duplicates = 0
        self.rejected = 0
        self.accepted = 0

    def _key(self, event):
        kind = event.get('type', '')
        symbol = event.get('symbol', '')
        if kind == 'trade' and event.get('trade_id'):
            identity = event['trade_id']
        elif kind == 'kline':
            identity = tuple(event.get(key) for key in ('open_time', 'event_time', 'is_closed', 'open', 'high', 'low', 'close', 'volume'))
        elif kind == 'book':
            identity = (event.get('event_time'), event.get('bid'), event.get('ask'), event.get('bid_qty'), event.get('ask_qty'))
        else:
            identity = tuple(sorted((key, repr(value)) for key, value in event.items() if key not in ('source', 'sequence', 'event_key')))
        return hashlib.sha256(json.dumps([kind, symbol, identity], sort_keys=True, default=str).encode()).hexdigest()

    def normalize(self, event):
        if not event or not event.get('type') or not event.get('symbol'):
            self.rejected += 1
            return None
        if int(event.get('event_time', 0) or 0) <= 0:
            self.rejected += 1
            return None
        key = self._key(event)
        if key in self.seen:
            self.duplicates += 1
            return None
        if len(self.seen) >= self.max_seen:
            self.seen.clear()
        self.seen.add(key)
        normalized = {**event, 'source': self.source, 'event_key': key, 'sequence': self.sequence}
        self.sequence += 1
        self.accepted += 1
        return normalized

    def to_market_event(self, event):
        normalized = self.normalize(event)
        if normalized is None:
            return None
        return MarketEvent(int(normalized.get('event_time', 0)), int(normalized['sequence']), normalized['type'], normalized)

    def stats(self):
        return {'source': self.source, 'accepted': self.accepted, 'duplicates': self.duplicates, 'rejected': self.rejected, 'sequence': self.sequence, 'seen': len(self.seen)}
