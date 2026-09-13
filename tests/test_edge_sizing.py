"""Sizing that knows how good the signal is.

The audit measured the defect directly: `risk.size()` took entry, stop, equity and the
portfolio caps, and nothing else. A 0.5bp edge and a 50bp edge produced byte-identical
quantities. `probability_up` was computed by the calibrator, carried through the whole
decision path, and then read by no one -- the code even said so in a comment.

The fix has to be one-sided. Every existing cap, limit and test was calibrated against the
old sizes, so edge scaling may only ever shrink a position. These tests pin that, and pin
the other property that makes it usable at all: the two ways of describing an edge -- bps
or a probability -- must agree where they overlap.
"""
import pytest

from app.trading.risk import RiskEngine


def engine(**overrides):
    defaults = dict(max_risk=.005, max_portfolio_risk=.02, max_symbol_leverage=5,
                    max_gross_leverage=5, target_exposure=1.0)
    defaults.update(overrides)
    return RiskEngine(**defaults)


def plan(entry=100.0, stop=98.0, target=104.0, **extra):
    return {'symbol': 'X', 'side': 'LONG', 'entry': entry, 'stop': stop,
            'target': target, **extra}


# ------------------------------------------------------------ the defect
def test_a_weak_edge_and_a_strong_one_no_longer_size_the_same():
    """The exact measurement from the audit: 0.5bp vs 50bp."""
    risk = engine()
    weak = risk.size(plan(), 10_000.0, edge_bps=0.5)
    strong = risk.size(plan(), 10_000.0, edge_bps=50.0)
    assert strong['quantity'] > weak['quantity'] * 10


def test_the_scale_is_monotone_in_the_edge():
    """More edge must never mean a smaller position."""
    risk = engine()
    scales = [risk.size(plan(), 10_000.0, edge_bps=edge)['edge_scale']
              for edge in range(0, 41, 5)]
    assert scales == sorted(scales)
    assert scales[0] == pytest.approx(0.0)
    assert scales[-1] == pytest.approx(1.0)


def test_probability_up_is_actually_used_now():
    """It was computed, transported, and dropped."""
    risk = engine()
    low = risk.size(plan(), 10_000.0, probability_up=0.34)
    high = risk.size(plan(), 10_000.0, probability_up=0.50)
    assert high['quantity'] > low['quantity']
    assert low['implied_probability'] == pytest.approx(0.34)


# ------------------------------------------------------- one-sided safety
def test_scaling_can_only_shrink_a_position():
    """Every cap and test was calibrated against the old sizes."""
    risk = engine()
    baseline = risk.size(plan(), 10_000.0)['quantity']
    for edge in (0, 1, 12.5, 25, 100, 10_000):
        sized = risk.size(plan(), 10_000.0, edge_bps=edge)
        assert sized['quantity'] <= baseline + 1e-9


def test_every_signal_keeps_the_full_budget_at_or_above_the_reference():
    """The status quo is the worst case, not a stretch target."""
    risk = engine()
    baseline = risk.size(plan(), 10_000.0)['quantity']
    for edge in (25.0, 40.0, 250.0):
        assert risk.size(plan(), 10_000.0, edge_bps=edge)['quantity'] == \
            pytest.approx(baseline)


def test_no_edge_information_sizes_exactly_as_before():
    """Every existing caller keeps its behaviour byte for byte."""
    risk = engine()
    plain = risk.size(plan(), 10_000.0)
    assert plain['edge_scale'] == pytest.approx(1.0)
    assert plain['kelly_fraction'] is None
    assert plain['implied_probability'] is None
    assert plain['quantity'] > 0


def test_setting_the_reference_to_zero_disables_scaling():
    """The escape hatch back to the pre-audit behaviour."""
    risk = engine(sizing_reference_edge_bps=0.0)
    none = risk.size(plan(), 10_000.0, edge_bps=0.0)['quantity']
    huge = risk.size(plan(), 10_000.0, edge_bps=10_000.0)['quantity']
    assert none == pytest.approx(huge)
    assert none > 0


# ------------------------------------------------------------ correctness
def test_no_edge_at_all_produces_no_position():
    """A zero edge is break-even, and the growth-optimal bet there is nothing."""
    risk = engine()
    sized = risk.size(plan(), 10_000.0, edge_bps=0.0)
    assert sized['quantity'] == pytest.approx(0.0)
    assert sized['edge_scale'] == pytest.approx(0.0)
    assert sized['binding'] == 'risk_budget_exhausted'
    assert sized['kelly_fraction'] == pytest.approx(0.0)


def test_an_edge_below_cost_breakeven_is_refused_not_merely_shrunk():
    """Negative edge: Kelly is negative, so the scale is zero rather than negative."""
    risk = engine()
    sized = risk.size(plan(), 10_000.0, probability_up=0.10)
    assert sized['quantity'] == pytest.approx(0.0)
    assert sized['edge_scale'] == pytest.approx(0.0)
    assert sized['kelly_fraction'] < 0


def test_the_two_ways_of_stating_an_edge_agree_where_they_overlap():
    """`edge_bps=0` and `probability_up=break_even` describe the same bet.

    They are computed by different arithmetic -- one from a claimed bps figure, the other
    from the model's calibrated probability -- so this pins that the conversion between
    them is consistent. Before the conversion was corrected they disagreed badly: zero bps
    gave a scale of 0.0 while break-even probability gave 0.84.
    """
    risk = engine()
    # Reward:risk is 2:1 here, so break-even is a 1/3 win rate.
    by_edge = risk.size(plan(), 10_000.0, edge_bps=0.0)
    by_probability = risk.size(plan(), 10_000.0, probability_up=1.0 / 3.0)
    assert by_edge['edge_scale'] == pytest.approx(0.0)
    assert by_probability['edge_scale'] == pytest.approx(0.0)
    assert by_edge['implied_probability'] == pytest.approx(
        by_probability['implied_probability'])
    # And the same at a strong edge, stated both ways.
    edge_scale = risk.size(plan(), 10_000.0, edge_bps=25.0)
    implied = edge_scale['implied_probability']
    probability_scale = risk.size(plan(), 10_000.0, probability_up=implied)
    assert probability_scale['edge_scale'] == pytest.approx(edge_scale['edge_scale'])


def test_break_even_moves_with_the_plan_geometry():
    """A 1:1 plan needs a 50% hit rate; a 3:1 plan needs 25%."""
    risk = engine()
    even_odds = risk.size(plan(entry=100.0, stop=98.0, target=102.0), 10_000.0,
                          probability_up=0.50)
    assert even_odds['edge_scale'] == pytest.approx(0.0)
    three_to_one = risk.size(plan(entry=100.0, stop=98.0, target=106.0), 10_000.0,
                             probability_up=0.50)
    assert three_to_one['edge_scale'] == pytest.approx(1.0)


def test_a_wider_stop_does_not_inflate_the_budget_through_the_edge_path():
    """The stop still governs risk, whatever the edge says."""
    risk = engine()
    tight = risk.size(plan(stop=99.0), 10_000.0, edge_bps=25.0)
    wide = risk.size(plan(stop=90.0), 10_000.0, edge_bps=25.0)
    assert tight['risk_cash'] == pytest.approx(wide['risk_cash'])
    assert wide['quantity'] < tight['quantity']


def test_the_scale_is_reported_on_every_zero_quantity_path():
    """The reason a caller sees a refusal has to include what the edge contributed."""
    risk = engine()
    exhausted = risk.size(plan(), 0.0, edge_bps=10.0)
    assert exhausted['binding'] == 'risk_budget_exhausted'
    assert 'edge_scale' in exhausted
    capped = risk.size(plan(), 10_000.0, symbol_notional=1e9, edge_bps=10.0)
    assert capped['binding'] == 'notional_room'
    assert 'edge_scale' in capped


def test_approve_passes_the_edge_through_to_the_size():
    """The live path reaches size() only through approve()."""
    risk = engine()
    weak = risk.approve(plan(), 10_000.0, edge_bps=1.0)
    strong = risk.approve(plan(), 10_000.0, edge_bps=50.0)
    assert weak['approved'] and strong['approved']
    assert strong['quantity'] > weak['quantity'] * 10
    assert strong['edge_scale'] == pytest.approx(1.0)


def test_the_edge_is_read_from_the_plan_when_it_is_not_a_keyword():
    """The live candidate carries its own edge; requiring the caller to unpack it means
    one forgotten argument silently restores flat sizing, which is the bug being fixed."""
    risk = engine()
    weak = risk.approve({**plan(), 'edge_bps': 0.5}, 10_000.0)
    strong = risk.approve({**plan(), 'edge_bps': 50.0}, 10_000.0)
    assert strong['quantity'] > weak['quantity'] * 10
    assert weak['edge_scale'] < 1.0


def test_an_explicit_keyword_wins_over_the_plan_field():
    """So a caller can override what the planner put there."""
    risk = engine()
    decision = risk.approve({**plan(), 'edge_bps': 0.0}, 10_000.0, edge_bps=50.0)
    assert decision['edge_scale'] == pytest.approx(1.0)


def test_the_probability_is_also_read_from_the_plan():
    """`probability_up` is written into the plan by the calibrator."""
    risk = engine()
    low = risk.approve({**plan(), 'probability_up': 0.34}, 10_000.0)
    high = risk.approve({**plan(), 'probability_up': 0.50}, 10_000.0)
    assert high['quantity'] > low['quantity']


def test_a_wide_stop_plan_still_scales_end_to_end():
    """The realistic geometry: a 2% stop on a 50k entry."""
    risk = engine(max_symbol_leverage=3)
    wide = {'symbol': 'BTCUSDT', 'side': 'LONG', 'entry': 50_000.0, 'stop': 49_000.0,
            'take_profit': 52_000.0}
    weak = risk.approve({**wide, 'edge_bps': 0.5}, 10_000.0)
    strong = risk.approve({**wide, 'edge_bps': 25.0}, 10_000.0)
    assert weak['edge_scale'] == pytest.approx(0.02)
    assert strong['edge_scale'] == pytest.approx(1.0)
    assert weak['quantity'] < strong['quantity']


def test_approve_refuses_a_position_whose_edge_does_not_cover_its_cost():
    """Zero quantity is refused with a reason, not silently submitted."""
    risk = engine()
    decision = risk.approve(plan(), 10_000.0, edge_bps=0.0)
    assert decision['approved'] is False
    assert decision['reason'] == 'risk_budget_exhausted'
    assert decision['edge_scale'] == pytest.approx(0.0)
