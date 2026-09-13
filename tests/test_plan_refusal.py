"""A refused plan has to say so.

`plan()` returned None from four places without saying which, and every caller read that
as "do nothing". A decision the edge gate had approved -- reason codes showing
ensemble_agrees and no edge refusal -- then produced no order and no record of why.
The gap is real and not small: the edge gate accepts a predicted move of 2.5 bp while the
multiple gate demands 4.0 bp, so between those numbers a decision is approved and then
silently dropped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.strategy.live_models import ModelDecision  # noqa: E402


def decision(**overrides):
    options = dict(fee_rate=0.0004, maker_fee_rate=0.0002, slippage_bps=2.0,
                   entry_order_type="limit", min_edge_bps=-1.5, min_agreement=0.3,
                   min_edge_multiple=1.0)
    options.update(overrides)
    return ModelDecision({}, **options)


def signal(expected_bps, side="LONG", price=100.0):
    return {"symbol": "TESTUSDT", "side": side, "expected_return": expected_bps / 10000.0,
            "features": {"price": price, "atr": 1.0}}


def test_the_round_trip_follows_the_execution_mode():
    # A resting entry pays the maker rate on both legs; a crossing one pays the taker rate
    # and the spread. The gate is measured against whichever is configured.
    assert abs(decision(entry_order_type="limit").round_trip_cost_pct - 0.0004) < 1e-12
    assert abs(decision(entry_order_type="market").round_trip_cost_pct - 0.0012) < 1e-12


def test_a_plan_that_passes_reports_ok():
    plan, reason = decision().plan_or_reason(signal(6.0))
    assert plan is not None
    assert reason == "ok"
    assert plan["stop"] < plan["entry"] < plan["take_profit"]


def test_a_move_inside_the_dead_zone_names_the_multiple_gate():
    # 3.0 bp clears the edge gate (2.5 bp) and fails the multiple gate (4.0 bp). Before
    # this, that decision was approved and then dropped with nothing recorded.
    plan, reason = decision().plan_or_reason(signal(3.0))
    assert plan is None
    assert reason.startswith("plan_edge_below_multiple")
    assert "3.00bp" in reason and "4.00bp" in reason


def test_a_side_that_is_not_directional_is_named():
    plan, reason = decision().plan_or_reason(signal(6.0, side="FLAT"))
    assert plan is None
    assert reason == "plan_no_direction"


def test_a_signal_without_a_price_is_named():
    blank = {"side": "LONG", "expected_return": 0.0006, "features": {}}
    plan, reason = decision().plan_or_reason(blank)
    assert plan is None
    assert reason == "plan_no_price"


def test_plan_still_returns_only_the_plan():
    # The old signature is what every other caller uses; it keeps working.
    assert decision().plan(signal(6.0)) is not None
    assert decision().plan(signal(3.0)) is None


def test_every_refusal_has_a_distinct_reason():
    reasons = {
        decision().plan_or_reason(signal(6.0, side="FLAT"))[1],
        decision().plan_or_reason({"side": "LONG", "features": {}})[1],
        decision().plan_or_reason(signal(3.0))[1].split(":")[0],
    }
    assert len(reasons) == 3, reasons
