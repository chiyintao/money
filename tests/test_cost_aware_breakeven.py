"""Break-even has to include the costs the trade actually pays.

The rule moved the stop to the entry price, so every "breakeven" exit booked a loss: the
stored sessions contain six stop-outs whose stop sat exactly on the entry, all negative,
together about -8.61 across accounts of 100, 200 and 10,000. Those were not losses the
market inflicted; they were the fee and spread the rule ignored.

The offset is per-leg because the legs are not symmetric. A market entry crosses the
spread; a resting one does not, but this system still exits at market, so that leg always
pays.
"""
import pytest

from app.trading.exit_policy import cost_aware_breakeven, manage_position, PRESETS
from app.trading.simulation import Position


def position(side='LONG', entry=100.0, stop=98.0, target=104.0):
    p = Position('TESTUSDT', side, 1.0, entry, stop, target)
    p.initial_stop = stop
    p.initial_target = target
    return p


def test_a_long_breakeven_sits_above_the_entry_by_the_round_trip():
    price = cost_aware_breakeven(100.0, 'LONG', fee_rate=0.0004, slippage_bps=2.0)
    assert price == pytest.approx(100.0 * (1 + 0.0004 + 0.0004 + 0.0002))


def test_a_short_breakeven_sits_below_the_entry():
    price = cost_aware_breakeven(100.0, 'SHORT', fee_rate=0.0004, slippage_bps=2.0)
    assert price < 100.0
    assert price == pytest.approx(100.0 * (1 - 0.001), rel=1e-9)


def test_a_resting_entry_does_not_pay_the_entry_spread():
    maker = cost_aware_breakeven(100.0, 'LONG', fee_rate=0.0004, slippage_bps=2.0,
                                 maker_entry=True)
    taker = cost_aware_breakeven(100.0, 'LONG', fee_rate=0.0004, slippage_bps=2.0)
    assert maker < taker
    # The exit is still a market order, so the spread is still paid on that leg.
    assert maker == pytest.approx(100.0 * (1 + 0.0004 + 0.0004))


def test_funding_is_part_of_the_cost_of_holding():
    paid = cost_aware_breakeven(100.0, 'LONG', fee_rate=0.0004, slippage_bps=2.0,
                                funding_rate=0.0001)
    unpaid = cost_aware_breakeven(100.0, 'LONG', fee_rate=0.0004, slippage_bps=2.0)
    assert paid > unpaid


def test_the_rule_uses_the_price_it_is_given():
    policy = PRESETS['scalp']
    p = position()
    # 0.6R of a 2.0 risk unit: 101.2 is the first qualifying price.
    out = manage_position(policy, p, 101.3, atr=0.0, breakeven_price=100.11)
    assert out['stop'] == pytest.approx(100.11)


def test_without_a_cost_model_the_entry_is_still_the_fallback():
    policy = PRESETS['scalp']
    p = position()
    out = manage_position(policy, p, 101.3, atr=0.0)
    assert out['stop'] == pytest.approx(100.0)


def test_a_cost_aware_breakeven_may_never_loosen_an_existing_stop():
    """The rule is still one-directional: it can only ever tighten."""
    policy = PRESETS['scalp']
    p = position()
    p.stop = 100.5  # already tightened past the cost-aware breakeven
    out = manage_position(policy, p, 101.3, atr=0.0, breakeven_price=100.11)
    assert out is None or out['stop'] >= 100.5
