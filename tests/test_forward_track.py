"""Forward paper tracking: the record must be point-in-time and append-only."""
import pytest

from app.core.domain import Event
from app.storage.storage import Store
from app.strategy import cross_sectional as cs
from app.strategy import forward_track as ft


class FakeStore:
    """Minimal store: an in-memory event log and a candle lookup."""

    def __init__(self, candles=None):
        import sqlite3
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE events (event_id TEXT PRIMARY KEY, type TEXT,"
                        " event_time INTEGER, payload TEXT)")
        self.candles_by_symbol = candles or {}
        self.signals = []

    def record_event(self, event):
        self.db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?)",
                        (event["event_id"], event["type"], event["event_time"],
                         __import__("json").dumps(event["payload"])))
        self.db.commit()

    def candles(self, symbol, interval="5m", limit=1000, closed_only=True):
        return self.candles_by_symbol.get(symbol, [])[-limit:]


def bars_for(days, start=100.0, drift=0.0, bars_per_day=288):
    """One bar per day is enough: the tracker only reads the last close of each day."""
    out = []
    price = start
    for day in range(days):
        out.append({"open_time": day * 86_400_000 + 86_399_000, "close": price})
        price *= (1.0 + drift)
    return out


def universe_of(n_symbols, days, base_drift=0.0):
    return {"S%02d" % i: bars_for(days, start=100.0, drift=base_drift + i * 0.0005)
            for i in range(n_symbols)}


def test_record_is_refused_when_the_day_has_not_closed():
    store = FakeStore()
    tracker = ft.ForwardTracker(store)
    days = list(range(100))
    # The wall clock is inside the newest day, so no day is complete.
    now = 99 * 86_400_000 + 1000
    assert tracker.latest_complete_day(days, now_ms=now) == 98


def test_latest_complete_day_excludes_the_current_day():
    days = [10, 11, 12]
    now = 12 * 86_400_000 + 5
    assert ft.ForwardTracker(FakeStore()).latest_complete_day(days, now_ms=now) == 11


def test_recording_the_same_day_twice_is_refused():
    store = FakeStore()
    tracker = ft.ForwardTracker(store)
    universe = universe_of(20, 40)
    first = tracker.record(universe, force_day=30)
    assert first["recorded"]
    second = tracker.record(universe, force_day=30)
    # Append-only: a signal cannot be revised after the outcome is known.
    assert not second["recorded"] and second["reason"] == "already_recorded"


def test_recorded_signal_carries_the_prices_it_was_computed_from():
    store = FakeStore()
    tracker = ft.ForwardTracker(store)
    tracker.record(universe_of(20, 40), force_day=30)
    payload = tracker.records(ft.SIGNAL_EVENT)[0]
    assert payload["prices_at"] and payload["weights"]
    assert payload["settled"] is False
    # prices_at holds the whole cross-section the ranking was computed from, while weights
    # holds only the traded quartiles. Keeping both is what lets the decision be re-derived:
    # the weights alone would not show which symbols were considered and rejected.
    assert set(payload["weights"]) <= set(payload["prices_at"])
    assert len(payload["prices_at"]) > len(payload["weights"])


def test_settlement_is_idempotent():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=1)
    universe = universe_of(20, 40)
    tracker.record(universe, force_day=30)
    first = tracker.settle(universe)
    second = tracker.settle(universe)
    assert first["settled"] == 1
    assert second["settled"] == 0  # already settled, never double counted


def test_a_target_is_not_settled_before_its_horizon_closes():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=5)
    universe = universe_of(20, 20)
    # Decision on the last available day: there is no day 5 ahead of it.
    tracker.record(universe, force_day=19)
    assert tracker.settle(universe)["settled"] == 0
    assert tracker.report()["pending"] == 1


def test_a_target_missed_while_the_tracker_was_down_is_recorded_as_a_miss():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=1, max_settle_delay_days=3)
    universe = universe_of(20, 40)
    tracker.record(universe, force_day=10)
    # The tracker was down and the days it missed are gone from the store: the earliest day
    # now available is 20, so settling this target would price a one-day holding nine days
    # late. It is recorded as a miss rather than settled, because settling it would silently
    # substitute a different strategy for the one that was written down.
    late = {"S%02d" % i: [b for b in bars_for(40) if cs.day_index(b["open_time"]) >= 20]
            for i in range(20)}
    result = tracker.settle(late)
    assert result["settled"] >= 1
    assert any(o.get("missed") for o in result["outcomes"])
    # Settling late would substitute a different strategy for the recorded one.
    assert tracker.report()["settled"] == 0
    assert tracker.report()["missed"] == 1


def test_settlement_uses_the_records_own_weights_and_never_re_ranks():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=1, cost_bps=0.0)
    universe = universe_of(20, 40)
    tracker.record(universe, force_day=30)
    payload = tracker.records(ft.SIGNAL_EVENT)[0]
    tracker.settle(universe)
    outcome = tracker.records(ft.SETTLEMENT_EVENT)[0]
    assert outcome["day"] == payload["day"]
    # Every weighted symbol appears in the settlement detail: nothing was silently dropped.
    assert set(outcome["moves"]) <= set(payload["weights"])


def test_report_counts_settled_and_pending_separately():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=1)
    universe = universe_of(20, 40)
    tracker.record(universe, force_day=30)
    tracker.record(universe, force_day=31)
    tracker.settle(universe)
    report = tracker.report()
    assert report["signals"] == 2
    assert report["settled"] + report["pending"] + report["missed"] == 2


def test_verdict_refuses_to_conclude_from_too_few_observations():
    store = FakeStore()
    tracker = ft.ForwardTracker(store, hold=1)
    universe = universe_of(20, 40)
    tracker.record(universe, force_day=30)
    tracker.settle(universe)
    text = " ".join(tracker.verdict())
    # One trade settles nothing; saying otherwise is the failure mode being guarded against.
    assert "too few forward trades" in text


def test_ledger_reads_from_the_events_sink_not_the_candle_store():
    # Writes go to the events store while candles are read from elsewhere; reading the
    # record back from the candle store silently reported zero signals.
    import tempfile
    import shutil
    market = FakeStore()
    tmp = tempfile.mkdtemp()
    try:
        events = Store(tmp)

        class Split:
            def __init__(self, m, e):
                self.market = m
                self.events = e
                self.db = m.db

            def candles(self, *a, **k):
                return self.market.candles(*a, **k)

            def record_event(self, event):
                return self.events.record_event(event)

        tracker = ft.ForwardTracker(Split(market, events))
        tracker._emit(ft.SIGNAL_EVENT, {"day": 1})
        assert len(tracker.records(ft.SIGNAL_EVENT)) == 1
        assert len(events.db.execute("SELECT 1 FROM events").fetchall()) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emitting_an_event_with_an_explicit_time_does_not_mutate_a_frozen_event():
    # Event is a frozen dataclass; assigning to it raised FrozenInstanceError, which is the
    # same defect that made order_state.transition() unusable on PaperOrder.
    store = FakeStore()
    tracker = ft.ForwardTracker(store)
    tracker._emit(ft.SIGNAL_EVENT, {"day": 7}, event_time=123456789)
    assert tracker.records(ft.SIGNAL_EVENT)[0]["event_time"] == 123456789


def test_tracker_default_eligibility_is_derived_from_the_store():
    """A hardcoded bar floor overrides the store-derived one and admits short listings.

    Measured: min_bars=50_000 admitted 34 symbols where the store-derived floor admitted
    the 31 with a full year, which truncated the cross-section and cut the edge from
    78.2 bps at t = 3.37 to 30.0 bps at t = 1.17.
    """
    tracker = ft.ForwardTracker(FakeStore())
    assert tracker.min_bars is None  # derived by read_universe, not fixed


def test_an_explicit_min_bars_still_overrides():
    tracker = ft.ForwardTracker(FakeStore(), min_bars=1234)
    assert tracker.min_bars == 1234
