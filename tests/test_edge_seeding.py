"""The gate must be able to learn from blocked symbols.

The deadlock this locks down: symbol_edge.allows() denies until a symbol has 30 scored
observations, and observations come from the primary model's *proposed* direction. But
the audit trail recorded only the post-gate side, which is FLAT for every symbol the gate
is currently blocking -- so seed_from_history, which rebuilds the gate's evidence from
that trail, recovered samples for symbols that never needed them and none for the blocked
ones. The gate therefore stayed blind on exactly the symbols it was refusing: 1238 stored
FLAT decisions rebuilt a single symbol with 19 samples, and every newly selected symbol
had to survive 33 bars (165 minutes) of refusals before it could clear the floor.

Two fields had to survive into the audit row for the replay to work: the proposed
direction, and the bar the decision was anchored to. The second was missing too --
bar_time read None on every stored decision, so the replay substituted the wall-clock
time the row was written, which lands mid-bar and scored forecasts against the wrong
reference.
"""

from app.strategy.symbol_edge import SymbolEdge
from app.trading.execution import decision_payload


def _signal(**overrides):
    signal = {
        "symbol": "X",
        "side": "FLAT",
        "proposed_side": "SHORT",
        "confidence": 0.8,
        "reason_codes": ["ensemble_agrees", "insufficient_edge_samples"],
        "features": {"price": 1.23},
        "bar_time": 1789198799999,
        "expected_return": 0.002,
        "edge_bps": 20.9,
        "agreement": 0.8,
        "votes": {},
        "source": "real_models",
        "model_mode": "candidate",
        "model_version": "v",
    }
    signal.update(overrides)
    return signal


def test_the_audit_row_keeps_the_proposed_direction():
    """side is the verdict; proposed_side is what the gate needs to learn from."""
    decision, _ = decision_payload(_signal(), None, "bar")
    assert decision["side"] == "FLAT"
    assert decision["proposed_side"] == "SHORT"


def test_the_audit_row_is_anchored_to_a_bar():
    """Without this the replay substitutes wall-clock time and scores the wrong bar."""
    _, market = decision_payload(_signal(), None, "bar")
    assert market["bar_time"] == 1789198799999


def test_a_payload_without_a_proposal_falls_back_to_the_side():
    """Older callers build signals without the field and must not crash."""
    signal = _signal(side="LONG")
    signal.pop("proposed_side")
    decision, _ = decision_payload(signal, None, "bar")
    assert decision["proposed_side"] == "LONG"


def test_a_blocked_symbol_can_still_accumulate_evidence():
    """The property the whole meta-labeling layer depends on."""
    edge = SymbolEdge(horizon_bars=3, window=200, min_samples=30, require_samples=True,
                      round_trip_cost=0.0012)
    for i in range(30):
        edge.observe("X", "SHORT", 1_700_000_000_000 + i * 300_000, 1.0)
    assert len(edge.pending["X"]) == 30
    times = [1_700_000_000_000 + i * 300_000 for i in range(40)]
    closes = [1.0] * 40
    assert edge.resolve("X", times, closes) == 30
    stats = edge.stats("X")
    assert stats["n"] == 30


def test_a_flat_proposal_is_not_observed():
    """A model with no view has no forecast to score."""
    edge = SymbolEdge(horizon_bars=3, min_samples=1, require_samples=True)
    edge.observe("X", "FLAT", 1_700_000_000_000, 1.0)
    assert edge.pending == {}


def test_direction_is_recovered_from_the_member_votes():
    """Rows written before proposed_side existed still carry the votes."""
    from app.strategy.symbol_edge import _proposed_from_votes
    assert _proposed_from_votes({"votes": {"a": 0.002, "b": 0.001}}) == "LONG"
    assert _proposed_from_votes({"votes": {"a": -0.002, "b": 0.001}}) == "SHORT"


def test_votes_that_cancel_out_yield_no_direction():
    """A flat ensemble has no forecast worth scoring."""
    from app.strategy.symbol_edge import _proposed_from_votes
    assert _proposed_from_votes({"votes": {"a": 0.002, "b": -0.002}}) is None
    assert _proposed_from_votes({"votes": {}}) is None
    assert _proposed_from_votes({}) is None


def test_malformed_votes_do_not_raise():
    from app.strategy.symbol_edge import _proposed_from_votes
    assert _proposed_from_votes({"votes": {"a": "not-a-number"}}) is None


def test_seeding_uses_the_proposal_over_the_gated_side():
    """The gate wrote FLAT; the proposal is what the gate learns from."""
    from app.strategy.symbol_edge import _proposed_from_votes
    decision = {"side": "FLAT", "proposed_side": "LONG"}
    assert (decision.get("proposed_side") or _proposed_from_votes(decision)
            or decision.get("side")) == "LONG"
