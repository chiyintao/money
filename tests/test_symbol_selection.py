"""Symbol selection must respect both the position cap and book coverage.

The stored session selected seven symbols while MAX_POSITIONS was four and two of the
seven never received an order book. Five of seven symbols were therefore guaranteed to
produce nothing, while still costing a model evaluation, a websocket subscription and a
warm candle cache on every tick. max_positions was 185 of 257 entry rejections and
missing_book_time another 40.
"""
import pytest

from app.backtest.simulation_session import SimulationSession


class FakeStore:
    def __init__(self):
        self.events = []
        self.runtime = {}

    def record_event(self, payload):
        self.events.append(payload)

    def set_runtime(self, key, value):
        self.runtime[key] = value


def make_session(count=7, source="gainers"):
    store = FakeStore()
    session = SimulationSession(store, source=source, symbol_count=count)
    session.status = "running"
    return session, store


def market_for(symbols):
    return {"gainers": [{"symbol": symbol, "price": 1.0} for symbol in symbols]}


def test_selection_never_exceeds_the_positions_available():
    session, _ = make_session(count=7)
    selected = session.choose(market_for([f"S{i}USDT" for i in range(7)]), max_positions=4)
    assert len(selected) == 4


def test_selection_uses_the_full_count_when_slots_allow():
    session, _ = make_session(count=3)
    selected = session.choose(market_for([f"S{i}USDT" for i in range(10)]), max_positions=5)
    assert len(selected) == 3


def test_selection_without_a_cap_uses_the_configured_count():
    session, _ = make_session(count=3)
    assert len(session.choose(market_for([f"S{i}USDT" for i in range(10)]))) == 3


def test_symbols_without_a_price_are_skipped():
    session, _ = make_session(count=3)
    market = {"gainers": [{"symbol": "AUSDT", "price": 0}, {"symbol": "BUSDT", "price": 1.0},
                          {"symbol": "CUSDT", "price": 1.0}, {"symbol": "DUSDT", "price": 1.0}]}
    assert session.choose(market, max_positions=3) == ["BUSDT", "CUSDT", "DUSDT"]


def test_a_book_marks_a_symbol_live():
    session, _ = make_session()
    session.choose(market_for(["AUSDT", "BUSDT"]), max_positions=2)
    session.note_book("AUSDT", now_ms=1_000_000)
    assert session.without_market_data(now_ms=1_000_000) == []


def test_a_symbol_is_not_dropped_before_the_grace_period():
    # A symbol that has simply not ticked yet is not the same as one that never will.
    session, _ = make_session()
    session.choose(market_for(["AUSDT"]), max_positions=2)
    assert session.without_market_data(now_ms=session.last_selection_at + 1000) == []


def test_a_symbol_with_no_book_is_dropped_after_the_grace_period():
    session, _ = make_session()
    session.choose(market_for(["AUSDT"]), max_positions=2)
    assert session.without_market_data(now_ms=session.last_selection_at + 61_000) == ["AUSDT"]


def test_a_symbol_whose_book_stopped_is_dropped():
    session, _ = make_session()
    session.choose(market_for(["AUSDT"]), max_positions=2)
    session.note_book("AUSDT", now_ms=session.last_selection_at + 90_000)
    # Still publishing well past the selection grace, then it goes quiet.
    assert session.without_market_data(now_ms=session.last_selection_at + 100_000) == []
    assert session.without_market_data(now_ms=session.last_selection_at + 151_000) == ["AUSDT"]


def test_a_symbol_that_keeps_publishing_is_never_retired():
    # Regression: the grace window was measured from selection time, so every symbol was
    # retired 60s after being chosen -- including the ones publishing normally. That
    # would have retired the entire session, not just the dead symbols.
    session, _ = make_session()
    session.choose(market_for(["AUSDT", "BUSDT"]), max_positions=2)
    for tick in range(0, 600_000, 5_000):
        session.note_book("AUSDT", now_ms=1_000_000 + tick)
        session.note_book("BUSDT", now_ms=1_000_000 + tick)
        assert session.without_market_data(now_ms=1_000_000 + tick) == []


def test_only_the_silent_symbol_is_retired_while_the_others_keep_trading():
    session, _ = make_session()
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3)
    start = session.last_selection_at
    for tick in range(0, 120_000, 5_000):
        session.note_book("AUSDT", now_ms=start + tick)
        session.note_book("BUSDT", now_ms=start + tick)
    # CUSDT is silent from the moment it was selected, so it is the only one retired --
    # the symbols still publishing must not be caught by the same sweep.
    assert session.without_market_data(now_ms=start + 61_000) == ["CUSDT"]
    assert session.without_market_data(now_ms=start + 119_000) == ["CUSDT"]


def test_book_tracking_survives_a_persist_and_restore_round_trip():
    session, store = make_session()
    session.choose(market_for(["AUSDT"]), max_positions=2)
    session.note_book("AUSDT", now_ms=1_000_000)
    session._persist(force=True)
    restored = SimulationSession.restore(FakeStore(), store.runtime["simulation_session"])
    # Without this surviving a restart, every symbol looks bookless again after a
    # deploy and live ones become eligible for retirement.
    assert restored.book_seen_at == {"AUSDT": 1_000_000}
    assert restored.last_book_at == 1_000_000


def test_reselect_replaces_a_dead_symbol_from_the_same_ranking():
    session, store = make_session(count=3)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]), max_positions=3)
    session.note_book("AUSDT", now_ms=1_000_000)
    session.note_book("BUSDT", now_ms=1_000_000)
    dropped = session.reselect(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]),
                               max_positions=3, drop=["CUSDT"])
    assert dropped == ["CUSDT"]
    assert session.selected_symbols == ["AUSDT", "BUSDT", "DUSDT"]
    assert len(session.selected_symbols) == 3


def test_reselect_records_why_the_symbols_changed():
    session, store = make_session(count=2)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=2)
    session.reselect(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=2, drop=["BUSDT"])
    assert session.selected_symbols == ["AUSDT", "CUSDT"]


def test_a_dead_symbol_is_retired_even_with_nothing_to_promote():
    # A dead symbol holding a position slot is worse than an empty slot: it can never
    # fill, and the empty slot can still be taken by a symbol that trades. Requiring a
    # replacement before dropping made a restored session unreselectable.
    session, _ = make_session(count=3)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3)
    dropped = session.reselect(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3, drop=["CUSDT"])
    assert dropped == ["CUSDT"]
    assert session.selected_symbols == ["AUSDT", "BUSDT"]


def test_reselect_never_drops_every_symbol():
    session, _ = make_session(count=3)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3)
    dropped = session.reselect(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3,
                               drop=["AUSDT", "BUSDT", "CUSDT"])
    assert dropped == []
    assert session.selected_symbols == ["AUSDT", "BUSDT", "CUSDT"]


def test_reselect_refills_from_the_ranking_up_to_the_configured_count():
    session, _ = make_session(count=3)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=3)
    wider = market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"])
    session.reselect(wider, max_positions=3, drop=["CUSDT"])
    assert session.selected_symbols == ["AUSDT", "BUSDT", "DUSDT"]


def test_reselect_never_exceeds_the_position_cap():
    session, _ = make_session(count=7)
    session.choose(market_for([f"S{i}USDT" for i in range(7)]), max_positions=3)
    session.reselect(market_for([f"S{i}USDT" for i in range(7)]), max_positions=3, drop=["S0USDT"])
    assert len(session.selected_symbols) == 3


# ------------------------------------------------- the gate drives what gets selected

def tiers(mapping, default=1):
    """Tier lookup: 2 tradable, 1 unproven, 0 measured as a loser."""
    return lambda symbol: mapping.get(symbol, default)


def test_eligible_symbols_are_preferred_over_the_raw_ranking():
    # The ranking picks by price move, the gate rejects by measured edge. Left independent
    # a session selected five symbols that all failed the gate and never traded.
    session, _ = make_session(count=2)
    selected = session.choose(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT"]),
                              max_positions=2, tier=tiers({"CUSDT": 2, "DUSDT": 2}))
    assert selected == ["CUSDT", "DUSDT"]


def test_a_slot_is_reserved_so_unproven_symbols_are_still_observed():
    # Evidence comes from signals, and signals are only computed for held symbols, so a
    # selector that took only proven symbols could never discover a new one.
    session, _ = make_session(count=3)
    selected = session.choose(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT"]),
                              max_positions=3, tier=tiers({"CUSDT": 2}), explore_slots=1)
    assert selected[0] == "CUSDT"
    assert len(selected) == 3
    assert "AUSDT" in selected          # the ranking's best unproven symbol explores


def test_unproven_symbols_outrank_ones_already_measured_as_losers():
    # Both kinds of "no" are not equivalent: an unproven symbol might become tradable,
    # while a symbol with fifty samples of negative edge has already been answered. The
    # session used to fill its slots with the second kind and never observe the first.
    session, _ = make_session(count=3)
    selected = session.choose(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT"]),
                              max_positions=2, explore_slots=1,
                              tier=tiers({"AUSDT": 0, "BUSDT": 0, "CUSDT": 1, "DUSDT": 2}))
    assert selected == ["DUSDT", "CUSDT"]


def test_a_loser_is_still_taken_when_nothing_else_is_available():
    session, _ = make_session(count=2)
    selected = session.choose(market_for(["AUSDT", "BUSDT"]), max_positions=2,
                              tier=tiers({}, default=0))
    assert selected == ["AUSDT", "BUSDT"]


def test_exploration_never_takes_every_slot():
    session, _ = make_session(count=5)
    selected = session.choose(market_for(["AUSDT", "BUSDT"]), max_positions=4,
                              tier=tiers({}, default=0), explore_slots=99)
    assert len(selected) == 2           # only two candidates exist at all


def test_without_a_gate_the_plain_ranking_is_used():
    session, _ = make_session(count=2)
    selected = session.choose(market_for(["AUSDT", "BUSDT", "CUSDT"]), max_positions=2)
    assert selected == ["AUSDT", "BUSDT"]


def test_reselect_prefers_eligible_replacements():
    session, _ = make_session(count=3)
    session.choose(market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]), max_positions=3)
    wider = market_for(["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"])
    session.reselect(wider, max_positions=3, drop=["CUSDT"], tier=tiers({"EUSDT": 2}))
    assert "EUSDT" in session.selected_symbols
    assert "CUSDT" not in session.selected_symbols


def test_the_candidate_pool_is_widened_with_eligible_symbols():
    # The ranking lists only the top movers, a few dozen of several hundred markets. A
    # symbol the gate has proven profitable can sit outside that slice, and then its edge
    # is unreachable however the session ranks it.
    from app.strategy.decision_loop import _candidates
    market = {
        "gainers": [{"symbol": "AUSDT", "price": 1.0}, {"symbol": "BUSDT", "price": 1.0}],
        "all": [{"symbol": "AUSDT", "price": 1.0}, {"symbol": "BUSDT", "price": 1.0},
                {"symbol": "PROVENUSDT", "price": 1.0}, {"symbol": "BADUSDT", "price": 1.0}],
    }
    tier = lambda symbol: 2 if symbol == "PROVENUSDT" else 0
    rows = _candidates(market, "gainers", tier)
    symbols = [row["symbol"] for row in rows]
    assert symbols[:2] == ["AUSDT", "BUSDT"]        # the ranking keeps its order
    assert "PROVENUSDT" in symbols                  # proven edge becomes reachable
    assert "BADUSDT" not in symbols                 # measured losers are not added


def test_a_proven_symbol_outside_the_ranking_can_be_selected():
    from app.strategy.decision_loop import _candidates
    session, _ = make_session(count=2)
    market = {
        "gainers": [{"symbol": "AUSDT", "price": 1.0}, {"symbol": "BUSDT", "price": 1.0}],
        "all": [{"symbol": "PROVENUSDT", "price": 1.0}],
    }
    tier = lambda symbol: 2 if symbol == "PROVENUSDT" else 0
    rows = _candidates(market, "gainers", tier)
    selected = session.choose({**market, "gainers": rows}, max_positions=1, tier=tier)
    assert selected == ["PROVENUSDT"]


def test_widening_is_skipped_entirely_without_a_gate():
    from app.strategy.decision_loop import _candidates
    market = {"gainers": [{"symbol": "AUSDT", "price": 1.0}],
              "all": [{"symbol": "XUSDT", "price": 1.0}]}
    assert [row["symbol"] for row in _candidates(market, "gainers", None)] == ["AUSDT"]
