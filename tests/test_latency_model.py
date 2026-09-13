"""Latency: the cost that a backtest fills at the signal price pretends does not exist.

A decision is made on a bar close and the fill does not happen at that price. The old fill
model charged the spread and, optionally, market impact -- both of which describe the state
of the book *at the instant of the fill*. Neither describes the move that happened while the
order was in flight, and that move is systematically adverse: an order fills because it
became marketable, which means the price came to it.

Two properties matter more than the exact number. The term has to be off by default, so no
existing result changes, and it has to scale with volatility rather than being a flat toll --
a quiet market is genuinely cheap to reach and a violent one is not.
"""
import pytest

from app.trading.fill_models import FillModel


def model(latency_ms=0.0, adverse=1.0, spread=2.0):
    return FillModel(spread_bps=spread, entry_latency_ms=latency_ms,
                     latency_adverse_share=adverse)


# ------------------------------------------------------------- it is off
def test_no_latency_configured_fills_exactly_as_before():
    """The default has to leave every existing result untouched."""
    priced = model(latency_ms=0.0)
    assert priced.slip_bps(volatility_bps=500.0) == pytest.approx(2.0)
    assert priced.price(50_000.0, 'BUY') == pytest.approx(50_010.0)


def test_latency_without_a_volatility_measurement_charges_nothing():
    """No measurement means no estimate, not a zero-volatility estimate."""
    priced = model(latency_ms=2_000)
    assert priced.slip_bps(volatility_bps=0.0) == pytest.approx(2.0)
    assert priced.latency_bps(2_000, None) == pytest.approx(0.0)


def test_zero_adverse_share_disables_the_term():
    """The explicit opt-out, for anyone who wants the old behaviour with latency set."""
    priced = model(latency_ms=2_000, adverse=0.0)
    assert priced.slip_bps(volatility_bps=200.0) == pytest.approx(2.0)


# --------------------------------------------------------- it is a cost
def test_latency_makes_a_buy_pay_more_and_a_sell_receive_less():
    """Adverse by construction: it is a cost, not a symmetric perturbation."""
    priced = model(latency_ms=2_000)
    quote = 50_000.0
    assert priced.price(quote, 'BUY', volatility_bps=200.0) > quote
    assert priced.price(quote, 'SELL', volatility_bps=200.0) < quote
    assert priced.price(quote, 'LONG', volatility_bps=200.0) == \
        pytest.approx(priced.price(quote, 'BUY', volatility_bps=200.0))


def test_a_resting_fill_pays_no_latency():
    """A maker fill happens at its own price; there is no crossing to be late for."""
    priced = model(latency_ms=2_000)
    assert priced.slip_bps(liquidity='maker', volatility_bps=200.0) == 0.0


def test_cost_rises_with_volatility():
    """The point of the term: a violent market is more expensive to reach."""
    priced = model(latency_ms=2_000)
    quiet = priced.slip_bps(volatility_bps=20.0)
    normal = priced.slip_bps(volatility_bps=100.0)
    violent = priced.slip_bps(volatility_bps=400.0)
    assert quiet < normal < violent


def test_cost_scales_with_the_square_root_of_latency():
    """Diffusion, not linear drift: quadrupling the flight time doubles the cost."""
    priced = model(latency_ms=2_000)
    short = priced.latency_bps(500, 200.0)
    long = priced.latency_bps(2_000, 200.0)
    assert long == pytest.approx(short * 2, rel=1e-9)


def test_latency_is_charged_on_top_of_spread_and_impact():
    """Three separate causes, three separate terms."""
    priced = FillModel(spread_bps=2.0, impact_coefficient_bps=4.0,
                       entry_latency_ms=2_000, reference_notional=10_000.0)
    spread_only = priced.slip_bps(notional=0.0)
    with_impact = priced.slip_bps(notional=10_000.0)
    with_latency = priced.slip_bps(notional=10_000.0, volatility_bps=100.0)
    assert spread_only == pytest.approx(2.0)
    assert with_impact == pytest.approx(6.0)
    assert with_latency > with_impact


def test_a_longer_bar_makes_the_same_latency_cheaper():
    """Latency is a share of a bar, so a 1h bar absorbs 350ms better than a 1m bar."""
    priced = model(latency_ms=350)
    on_one_minute = priced.latency_bps(volatility_bps=100.0, bar_ms=60_000)
    on_one_hour = priced.latency_bps(volatility_bps=100.0, bar_ms=3_600_000)
    assert on_one_hour < on_one_minute


# ------------------------------------------------ the depth path is special
def test_the_depth_path_does_not_charge_the_spread_twice():
    """Walking a real ladder already pays the spread and causes the impact.

    The levels it consumed are the book; adding the model's spread on top would count the
    same cost twice, and the first version of this did exactly that -- it moved a
    documented depth-fill price from 101.33 to 101.35.
    """
    priced = FillModel(spread_bps=2.0, impact_coefficient_bps=4.0,
                       entry_latency_ms=2_000, reference_notional=10_000.0)
    walked = priced.slip_bps(notional=10_000.0, volatility_bps=200.0,
                             include_spread=False, include_impact=False)
    assert walked == pytest.approx(priced.latency_bps(None, 200.0))
    assert walked < priced.slip_bps(notional=10_000.0, volatility_bps=200.0)


def test_the_description_reports_the_latency_assumptions():
    """So a run's costs can be read back from its own record."""
    described = model(latency_ms=350, adverse=0.5).describe()
    assert described['entry_latency_ms'] == pytest.approx(350.0)
    assert described['latency_adverse_share'] == pytest.approx(0.5)
