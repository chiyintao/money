"""A negative expected edge must never buy a position.

`edge_scale` took `abs(edge_bps)`, so a -10bp prediction was sized exactly like a +10bp
one: the more the model expected to lose, the more of the risk budget the trade received.
The serving gate computes a NET figure that is already negative for any prediction below
the round trip, which makes this reachable on ordinary signals rather than only on
malformed input.
"""
import pytest

from app.trading.risk import RiskEngine


def engine(**overrides):
    defaults = dict(max_risk=.005, max_portfolio_risk=.02, max_symbol_leverage=5,
                    max_gross_leverage=5, target_exposure=1.0)
    defaults.update(overrides)
    return RiskEngine(**defaults)


def plan(entry=100.0, stop=98.0, target=104.0):
    return {'symbol': 'X', 'side': 'LONG', 'entry': entry, 'stop': stop, 'target': target}


def test_a_negative_net_edge_sizes_nothing():
    risk = engine()
    scale, inputs = risk.edge_scale(plan(), edge_bps=-10.0)
    assert scale == pytest.approx(0.0)
    assert inputs['edge_bps'] == -10.0
    assert risk.size(plan(), 10_000.0, edge_bps=-10.0)['quantity'] == 0.0


def test_the_scale_no_longer_symmetrises_the_sign():
    risk = engine()
    negative = risk.edge_scale(plan(), edge_bps=-10.0)[0]
    positive = risk.edge_scale(plan(), edge_bps=10.0)[0]
    assert negative < positive


def test_a_negative_probability_edge_also_sizes_nothing():
    """The other input form describes the same bet and cannot disagree about its sign."""
    risk = engine()
    scale, inputs = risk.edge_scale(plan(), probability_up=0.10)
    assert scale == pytest.approx(0.0)
    assert inputs['kelly_fraction'] < 0


def test_a_short_plan_is_scaled_by_the_sign_of_its_own_edge():
    risk = engine()
    short = {'symbol': 'X', 'side': 'SHORT', 'entry': 100.0, 'stop': 102.0, 'target': 96.0}
    assert risk.edge_scale(short, edge_bps=-10.0)[0] == pytest.approx(0.0)
    assert risk.edge_scale(short, edge_bps=25.0)[0] == pytest.approx(1.0)
