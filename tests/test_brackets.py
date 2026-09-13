"""Stop and target geometry must be usable before an order is allowed out.

The audit found 16 trades whose take-profit sat on the losing side of the entry, all of
them within a hair of it. That fires on the next tick and books the spread plus two fees:
one symbol produced 37 round trips under ten seconds at a net loss. Nothing checked this.
"""
import pytest

from app.trading.brackets import bracket_problem, is_valid, repair_target
from app.trading.execution import reanchor_plan


def test_a_sound_long_bracket_is_accepted():
    assert is_valid({'side': 'LONG', 'entry': 100.0, 'stop': 98.0, 'take_profit': 105.0})


def test_a_sound_short_bracket_is_accepted():
    assert is_valid({'side': 'SHORT', 'entry': 100.0, 'stop': 102.0, 'take_profit': 95.0})


def test_a_long_target_below_entry_is_rejected():
    problem = bracket_problem({'side': 'LONG', 'entry': 100.0, 'stop': 98.0, 'take_profit': 99.0})
    assert problem == 'target_behind_entry'


def test_a_short_target_above_entry_is_rejected():
    problem = bracket_problem({'side': 'SHORT', 'entry': 100.0, 'stop': 102.0, 'take_profit': 101.0})
    assert problem == 'target_behind_entry'


def test_the_real_egld_plan_that_caused_instant_exits_is_rejected():
    # Taken verbatim from the audit: a SHORT entered at 4.98 with a 5.19 stop, whose
    # target of 4.9822 is on the wrong side and 0.2bps from entry.
    problem = bracket_problem({'side': 'SHORT', 'entry': 4.98, 'stop': 5.1919,
                               'take_profit': 4.9822})
    assert problem == 'target_behind_entry'


def test_the_real_inverted_long_plan_is_rejected():
    # Also from the audit: a LONG whose stop sits above its entry.
    problem = bracket_problem({'side': 'LONG', 'entry': 4.92, 'stop': 5.0339,
                               'take_profit': 5.2721})
    assert problem == 'stop_beyond_entry'


def test_a_target_essentially_at_entry_is_rejected_even_on_the_right_side():
    # 0.2bps of target cannot pay a 12bps round trip, so it is unusable rather than tight.
    problem = bracket_problem({'side': 'LONG', 'entry': 100.0, 'stop': 98.0,
                               'take_profit': 100.002})
    assert problem == 'target_inside_entry'


def test_a_long_stop_above_entry_is_rejected():
    assert bracket_problem({'side': 'LONG', 'entry': 100.0, 'stop': 101.0,
                            'take_profit': 110.0}) == 'stop_beyond_entry'


def test_a_short_stop_below_entry_is_rejected():
    assert bracket_problem({'side': 'SHORT', 'entry': 100.0, 'stop': 99.0,
                            'take_profit': 90.0}) == 'stop_beyond_entry'


def test_missing_levels_are_rejected_rather_than_assumed():
    assert bracket_problem({'side': 'LONG', 'entry': 100.0, 'stop': 98.0}) == 'missing_target'
    assert bracket_problem({'side': 'LONG', 'entry': 100.0,
                            'take_profit': 110.0}) == 'missing_stop'
    assert bracket_problem({'side': 'LONG', 'entry': 0.0, 'stop': 1.0,
                            'take_profit': 2.0}) == 'invalid_entry'


def test_a_nonsense_side_is_rejected():
    assert bracket_problem({'side': 'FLAT', 'entry': 100.0, 'stop': 98.0,
                            'take_profit': 110.0}) == 'invalid_side'


def test_a_trailing_stop_above_entry_is_allowed_when_checking_a_live_position():
    # Once a trailing stop has locked in profit it legitimately sits above entry for a
    # LONG, so re-checking an open position must not tighten the stop requirement.
    live = {'side': 'LONG', 'entry': 100.0, 'stop': 104.0, 'take_profit': 120.0}
    assert is_valid(live, require_stop=False)
    assert bracket_problem(live, require_stop=False) is None
    # But the target still has to be on the profit side.
    bad = {'side': 'LONG', 'entry': 100.0, 'stop': 104.0, 'take_profit': 99.0}
    assert bracket_problem(bad, require_stop=False) == 'target_behind_entry'


def test_repair_mirrors_the_stop_distance_onto_the_target():
    fixed = repair_target({'side': 'LONG', 'entry': 100.0, 'stop': 98.0, 'take_profit': 99.0})
    assert fixed['take_profit'] == pytest.approx(102.0)
    assert is_valid(fixed)
    short = repair_target({'side': 'SHORT', 'entry': 100.0, 'stop': 102.0, 'take_profit': 101.0})
    assert short['take_profit'] == pytest.approx(98.0)
    assert is_valid(short)


def test_repair_refuses_when_there_is_no_usable_stop():
    assert repair_target({'side': 'LONG', 'entry': 100.0, 'take_profit': 99.0}) is None
    assert repair_target({'side': 'LONG', 'entry': 100.0, 'stop': 100.0,
                          'take_profit': 99.0}) is None


# ------------------------------------------------- re-anchoring must not break a bracket

def test_a_large_favourable_gap_does_not_invert_the_bracket():
    # Shifting both levels down by more than the target distance would leave the target
    # behind the fill. Keeping the original levels is the safe outcome.
    plan = {'side': 'LONG', 'entry': 100.0, 'stop': 98.0, 'take_profit': 103.0}
    adjusted = reanchor_plan(plan, 99.0)
    assert is_valid(adjusted)


def test_a_small_gap_still_translates_both_levels():
    plan = {'side': 'LONG', 'entry': 100.0, 'stop': 95.0, 'take_profit': 115.0}
    adjusted = reanchor_plan(plan, 100.5)
    assert adjusted['stop'] == pytest.approx(95.5)
    assert adjusted['take_profit'] == pytest.approx(115.5)
    assert adjusted['entry_deviation_bps'] == pytest.approx(50.0)


def test_reanchor_leaves_a_plan_that_was_already_broken_untouched():
    # The fault is reported by submit_entry rather than masked by restoring bad levels.
    plan = {'side': 'LONG', 'entry': 100.0, 'stop': 101.0, 'take_profit': 99.0}
    adjusted = reanchor_plan(plan, 100.5)
    assert adjusted['stop'] == pytest.approx(101.5)
