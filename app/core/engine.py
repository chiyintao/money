from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class MarketEvent:
    """A normalized, replayable event ordered by exchange time."""

    event_time: int
    sequence: int
    kind: str
    payload: dict[str, Any]

    def __post_init__(self):
        if int(self.event_time) < 0:
            raise ValueError('invalid_event_time')
        if int(self.sequence) < 0:
            raise ValueError('invalid_sequence')
        if not self.kind:
            raise ValueError('missing_event_kind')

    @property
    def sort_key(self):
        return int(self.event_time), int(self.sequence)


class DeterministicEventEngine:
    """Stable event ordering shared by replay, backtest, and live adapters."""

    def __init__(self, handler: Callable[[MarketEvent], Any] | None = None):
        self.handler = handler
        self.processed = 0
        self.last_event_time = None
        self.rejected = 0

    def order(self, events: Iterable[MarketEvent]):
        ordered = sorted(events, key=lambda event: event.sort_key)
        for previous, current in zip(ordered, ordered[1:]):
            if current.sort_key < previous.sort_key:
                self.rejected += 1
                raise ValueError('event_order_violation')
        return ordered

    def process(self, events: Iterable[MarketEvent]):
        results = []
        for event in self.order(events):
            if self.last_event_time is not None and event.event_time < self.last_event_time:
                self.rejected += 1
                raise ValueError('event_time_regression')
            result = self.handler(event) if self.handler else event
            results.append(result)
            self.last_event_time = event.event_time
            self.processed += 1
        return results

    def snapshot(self):
        return {'processed': self.processed, 'last_event_time': self.last_event_time, 'rejected': self.rejected}


def candle_events(series):
    """Normalize symbol candle series into a deterministic multi-symbol stream."""
    events = []
    sequence = 0
    for symbol, rows in series.items():
        for row in rows:
            events.append(MarketEvent(int(row['open_time']), sequence, 'candle', {'symbol': symbol, 'bar': row}))
            sequence += 1
    return events
