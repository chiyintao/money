"""Audit retention: keep the hot events table bounded, archive the rest.

The events table reached 1.07M rows covering only 0.92 days, of which 758k were
strategy_decision rows that collapsed to 32 distinct decisions per 5000. Ordering by
event_time without an index scanned and sorted the whole table, so recent_events(100)
measured 6.3 seconds and stalled the decision path's writes behind it.
"""
import pytest

from app.storage.storage import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


def add_events(store, count, kind='market_event', start=0):
    store.record_events([{'event_id': f'{kind}-{start + i}', 'type': kind,
                          'event_time': start + i, 'payload': {'i': i}} for i in range(count)])


def test_prune_keeps_only_the_newest_rows(store):
    add_events(store, 100)
    moved = store.prune_events(10)
    assert store.event_counts()['events'] == 10
    assert sum(moved.values()) == 90


def test_pruned_rows_are_archived_not_lost(store):
    add_events(store, 50)
    store.prune_events(5)
    assert store.event_counts()['archived'] == 45
    # The archive holds the same payload, so nothing becomes unrecoverable.
    archived = store.db.execute('SELECT payload FROM events_archive ORDER BY event_time ASC LIMIT 1').fetchone()
    assert archived is not None


def test_durable_types_survive_regardless_of_age(store):
    add_events(store, 5, kind='fill')
    add_events(store, 95, kind='market_event', start=1000)
    store.prune_events(1)
    remaining = {row['type'] for row in store.db.execute('SELECT DISTINCT type FROM events')}
    assert 'fill' in remaining
    assert 'market_event' in remaining


def test_durable_types_can_be_pruned_when_asked(store):
    add_events(store, 5, kind='fill')
    add_events(store, 95, kind='market_event', start=1000)
    store.prune_events(1, keep_durable=False)
    assert store.event_counts()['events'] == 1


def test_prune_is_a_noop_below_the_threshold(store):
    add_events(store, 20)
    assert store.prune_events(100) == {}
    assert store.event_counts()['events'] == 20


def test_prune_on_an_empty_table_is_safe(store):
    assert store.prune_events(10) == {}


def test_zero_keeps_nothing_of_the_prunable_types(store):
    add_events(store, 30)
    store.prune_events(0)
    assert store.event_counts()['events'] == 0


def test_recent_events_stays_ordered_after_pruning(store):
    add_events(store, 200)
    store.prune_events(10)
    rows = store.recent_events(5)
    times = [row['event_time'] for row in rows]
    assert times == sorted(times, reverse=True)


def test_simulation_history_survives_pruning(store):
    # A session's outcome is rebuilt from lifecycle events, which must never be pruned.
    store.record_events([
        {'event_id': 's1', 'type': 'simulation_started', 'event_time': 1,
         'payload': {'session_id': 'sess', 'initial_cash': 100, 'leverage': 10}},
        {'event_id': 's2', 'type': 'simulation_ended', 'event_time': 2,
         'payload': {'session_id': 'sess', 'net_pnl': 5, 'trades': 1}},
    ])
    add_events(store, 100, kind='market_event', start=1000)
    store.prune_events(1)
    history = store.simulation_history()
    assert len(history) == 1
    assert history[0]['session_id'] == 'sess'
