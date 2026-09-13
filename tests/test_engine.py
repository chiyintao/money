import pytest

from app.core.engine import DeterministicEventEngine, MarketEvent, candle_events


def test_engine_orders_same_timestamp_by_sequence():
    events = [MarketEvent(10, 2, 'b', {}), MarketEvent(10, 1, 'a', {}), MarketEvent(9, 0, 'early', {})]
    assert [event.kind for event in DeterministicEventEngine().order(events)] == ['early', 'a', 'b']


def test_engine_replay_is_deterministic_and_tracks_state():
    seen = []
    engine = DeterministicEventEngine(lambda event: seen.append(event.kind))
    engine.process([MarketEvent(2, 0, 'second', {}), MarketEvent(1, 0, 'first', {})])
    assert seen == ['first', 'second']
    assert engine.snapshot() == {'processed': 2, 'last_event_time': 2, 'rejected': 0}


def test_engine_rejects_invalid_event_and_candle_stream_is_normalized():
    with pytest.raises(ValueError, match='invalid_event_time'):
        MarketEvent(-1, 0, 'bad', {})
    events = candle_events({'B': [{'open_time': 2}], 'A': [{'open_time': 1}]})
    assert [(event.event_time, event.payload['symbol']) for event in DeterministicEventEngine().order(events)] == [(1, 'A'), (2, 'B')]
