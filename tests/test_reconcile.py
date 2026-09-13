"""Candle-series continuity: a hole that no downstream indicator can see.

The gap detector was written for exactly this and then never called by anything, so a
stored series could lose bars silently: a 20-bar average spanning 22 bars of elapsed time
is not a 20-bar average, and every indicator computed over the series sees only a list of
rows. `Store.candles(..., check_gaps=True)` is now the caller, and it records what it found
so the hole is visible in the audit trail rather than inferred from wrong numbers later.
"""
import json

import pytest

from app.storage.reconcile import find_gaps, merge_events
from app.storage.storage import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


def bars(*open_times, close=100.0):
    return [{'symbol': 'BTCUSDT', 'interval': '1m', 'open_time': t,
             'close_time': t + 59_999, 'open': close, 'high': close,
             'low': close, 'close': close, 'volume': 1.0, 'is_closed': 1}
            for t in open_times]


def test_gap_and_dedup():
    rows = [{'open_time': 0}, {'open_time': 60000}, {'open_time': 180000}]
    assert find_gaps(rows, 60000) == [60000]
    assert len(merge_events(rows, [{'open_time': 60000, 'close': 2}])) == 3


def test_contiguous_series_has_no_gap():
    rows = [{'open_time': i * 60000} for i in range(4)]
    assert find_gaps(rows, 60000) == []


def test_check_gaps_records_an_event_for_a_hole(store):
    """A missing bar must leave evidence, not just a shorter list."""
    store.upsert_candles('BTCUSDT', '1m', bars(0, 60_000, 180_000))
    series = store.candles('BTCUSDT', '1m', check_gaps=True)
    assert len(series) == 3
    gaps = [e for e in store.recent_events(50) if e['type'] == 'candle_gap']
    assert len(gaps) == 1
    payload = json.loads(gaps[0]['payload'])
    assert payload['gap_count'] == 1
    assert payload['first_gap_open_time'] == 60_000


def test_check_gaps_is_silent_on_a_clean_series(store):
    store.upsert_candles('BTCUSDT', '1m', bars(0, 60_000, 120_000, 180_000))
    store.candles('BTCUSDT', '1m', check_gaps=True)
    assert [e for e in store.recent_events(50) if e['type'] == 'candle_gap'] == []


def test_gap_check_is_off_by_default(store):
    """It is O(n) on the read path and the live loop calls this every tick."""
    store.upsert_candles('BTCUSDT', '1m', bars(0, 180_000))
    store.candles('BTCUSDT', '1m')
    assert [e for e in store.recent_events(50) if e['type'] == 'candle_gap'] == []


def test_check_candle_gaps_returns_the_gap_open_times(store):
    store.upsert_candles('BTCUSDT', '1m', bars(0, 60_000, 300_000))
    series = store.candles('BTCUSDT', '1m')
    assert store.check_candle_gaps('BTCUSDT', '1m', series) == [60_000]
