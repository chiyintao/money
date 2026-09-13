"""Session scope: a feed outage is not symbol churn, and a mark-out is not an exit.

Two findings from the live service, each of which made a number mean something it did
not:

1. without_market_data retired every selected symbol on every pass because no book had
   *ever* arrived -- the websocket was down for five hours. The session re-picked its
   symbols every ~70 seconds (54 reselections in 24h). The bar loop only resolves
   forecasts for symbols it still holds, so each reselection discarded the pending
   forecasts along with the symbol that made them, and the per-symbol meta-labeling gate
   never accumulated a single sample. A data outage presented as a modelling result.

2. end() passed the session end reason straight to account.close, so a position still
   open when a session stopped was filed as a "manual" exit -- indistinguishable from a
   discretionary one. Nine of the sixteen stored "manual" rounds were these mark-outs.
"""
import time

import pytest

from app.backtest.simulation_session import SimulationSession
from app.strategy import decision_loop
from app.trading.simulation import PaperAccount, Position


class _Store:
    """Minimal store: the session only needs event recording and runtime persistence."""

    def __init__(self):
        self.events = []
        self.runtime = {}

    def record_event(self, payload):
        self.events.append(payload)

    def set_runtime(self, key, value):
        self.runtime[key] = value


def _session(**kwargs):
    account = PaperAccount(10_000, fee_rate=0.0, slippage_bps=0.0)
    store = _Store()
    return SimulationSession(store, account=account, initial_cash=10_000, **kwargs), store, account


# ------------------------------------------------- a total outage is not symbol churn
def test_a_total_book_outage_does_not_retire_symbols():
    """No book anywhere means the feed is down, not that the symbols are bad."""
    session, _, _ = _session(selected_symbols=["A", "B"])
    session.last_selection_at = int(time.time() * 1000) - 300_000   # past the grace
    session.book_seen_at = {}                                       # nothing arrived
    assert session.without_market_data(feed_live=False) == []


def test_a_stale_but_present_feed_still_retires_a_silent_symbol():
    """The per-symbol rule must survive the fix.

    One symbol silent while others are publishing is the case the rule exists for: it
    occupies a position slot that something tradable could hold.
    """
    now = int(time.time() * 1000)
    session, _, _ = _session(selected_symbols=["A", "B"])
    session.last_selection_at = now - 300_000
    session.book_seen_at = {"B": now - 1_000}           # only B is publishing
    # feed_live is the caller's verdict that the socket is up, so a silent symbol here
    # is the symbol's own problem.
    assert session.without_market_data(feed_live=True) == ["A"]


def test_a_freshly_selected_symbol_gets_its_grace_period():
    """A symbol is not retired merely for being slow to start."""
    now = int(time.time() * 1000)
    session, _, _ = _session(selected_symbols=["A"])
    session.last_selection_at = now - 1_000
    session.book_seen_at = {}
    assert session.without_market_data(feed_live=True) == []


# ------------------------------------------------- an exit rule is not an end reason
def test_ending_a_session_labels_the_markout_as_session_end():
    """A mark-out is not a discretionary exit and must not be filed as one.

    In the stored history "manual" held every profitable round (16 rounds, +404.47, 75%
    win rate) against -1101.76 for real stop/target exits, which reads as "the exit rule
    is the problem". Nine of those sixteen were sessions stopped mid-position.
    """
    session, _, account = _session(selected_symbols=["A"])
    session.status = "running"
    account.positions["A"] = Position("A", "LONG", 1.0, 100.0, 95.0, 120.0)
    account.marks["A"] = 103.0
    session.end("manual")
    assert [t.get("reason") for t in account.trades] == ["session_end"]
    assert account.trades[0]["reason_detail"]["session_reason"] == "manual"


def test_end_still_records_the_sessions_own_reason():
    """The session-level reason is a separate fact and must survive."""
    session, store, _ = _session(selected_symbols=["A"])
    session.status = "running"
    session.end("start_failed")
    ended = [e for e in store.events if e.get("type") == "simulation_ended"]
    assert ended, "the ended event must still be written"
    assert ended[-1]["payload"]["reason"] == "start_failed"


# ------------------------------------------------- initial capital is bounded
def test_initial_cash_is_bounded_at_both_ends():
    """A slipped decimal point used to create a new account size silently."""
    for bad in (0.0, -1.0, 50.0, 1_000_000.0):
        session, _, _ = _session()
        with pytest.raises(ValueError):
            session.start(bad, 3, "gainers", 3)


def test_ordinary_capitals_are_still_accepted():
    for good in (100.0, 1_000.0, 10_000.0, 100_000.0):
        session, _, _ = _session()
        session.start(good, 3, "gainers", 3)
        assert session.initial_cash == good


def test_leverage_and_symbol_count_are_still_validated():
    """The capital bound must not have replaced the other checks."""
    for kwargs in ({"leverage": 0}, {"leverage": 21}, {"symbol_count": 0},
                   {"symbol_count": 11}):
        session, _, _ = _session()
        args = {"leverage": 3, "symbol_count": 3, **kwargs}
        with pytest.raises(ValueError):
            session.start(10_000, args["leverage"], "gainers", args["symbol_count"])


# ------------------------------------------------- feed liveness is the caller's call
class _Health:
    def __init__(self, connected):
        self.connected = connected


class _Feed:
    def __init__(self, connected, running=True):
        self.health = _Health(connected)
        self.running = running


def _runtime_with(public=None, ws=None):
    feeds = type("F", (), {"public": public, "ws": ws})()
    return type("R", (), {"feeds": feeds})()


def test_a_connected_socket_counts_as_a_live_feed():
    runtime = _runtime_with(public=_Feed(connected=True))
    assert decision_loop._book_feed_live(runtime) is True


def test_a_configured_but_disconnected_socket_is_a_dead_feed():
    """The five-hour outage case: the socket is running and cannot connect."""
    runtime = _runtime_with(public=_Feed(connected=False, running=True))
    assert decision_loop._book_feed_live(runtime) is False


def test_no_socket_at_all_is_not_evidence_the_market_is_silent():
    """Otherwise a missing feed becomes a permanently stuck selection."""
    runtime = _runtime_with()
    assert decision_loop._book_feed_live(runtime) is True


def test_a_stopped_socket_is_not_counted_as_configured():
    runtime = _runtime_with(public=_Feed(connected=False, running=False))
    assert decision_loop._book_feed_live(runtime) is True

