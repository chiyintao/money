"""The round trip must be priced by the role each leg actually trades in.

The gate priced a resting entry as `2 x maker`, which is only true if the EXIT also
rests. It does not: the paper account closes at market and charges the taker rate plus
slippage on that leg. On the shipped defaults that is a 4bp assumption against an 8bp
bill -- the gate cleared trades the account then lost money on, and the loss is the
difference. Both sides of the system have to price the same fill.
"""
import inspect
import math

from app.strategy.live_models import ModelDecision


def decision(**overrides):
    # exit_order_type='market' is what the service actually does: the stop, the target,
    # the time stop and the manual close all send a market order.
    options = dict(fee_rate=0.0004, maker_fee_rate=0.0002, slippage_bps=2.0,
                   entry_order_type='limit', exit_order_type='market')
    options.update(overrides)
    return ModelDecision({}, **options)


def test_a_resting_entry_still_pays_the_taker_rate_on_the_exit():
    assert math.isclose(decision().round_trip_cost_pct, 0.0008, rel_tol=1e-12)


def test_a_market_entry_pays_the_taker_rate_and_spreads_on_both_legs():
    assert math.isclose(decision(entry_order_type='market').round_trip_cost_pct,
                        0.0012, rel_tol=1e-12)


def test_the_cost_is_broken_down_by_leg_so_a_refusal_can_name_the_leg():
    breakdown = decision().cost_breakdown()
    assert math.isclose(breakdown['entry_fee_pct'], 0.0002, rel_tol=1e-12)
    assert math.isclose(breakdown['exit_fee_pct'], 0.0004, rel_tol=1e-12)
    assert math.isclose(breakdown['exit_slippage_pct'], 0.0002, rel_tol=1e-12)
    assert math.isclose(sum(breakdown.values()), decision().round_trip_cost_pct, rel_tol=1e-12)


def test_a_market_entry_pays_entry_slippage_too():
    breakdown = decision(entry_order_type='market').cost_breakdown()
    assert math.isclose(breakdown['entry_slippage_pct'], 0.0002, rel_tol=1e-12)


def test_a_legacy_caller_that_only_names_a_maker_rate_keeps_its_old_answer():
    """The pre-existing signature had no exit order type, and callers that fit one
    round trip into that shape must not silently change size."""
    source = inspect.signature(ModelDecision.__init__).parameters
    assert 'exit_order_type' in source
    assert math.isclose(ModelDecision({}, fee_rate=0.0004, maker_fee_rate=0.0002,
                                       slippage_bps=2.0, entry_order_type='limit',
                                       exit_order_type='limit').round_trip_cost_pct,
                        0.0004, rel_tol=1e-12)
