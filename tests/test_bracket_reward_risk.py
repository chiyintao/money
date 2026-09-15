"""A plan whose reward is a fraction of its risk is not a plan.

The exchange records contain 36 trades that closed within three seconds of opening, and
41 that closed inside two minutes, for a combined -1.09 against 0.74 of fees. The reason
is visible in the stored levels of one of them:

    SHORT entry 4.98   stop 5.1918 (425 bp)   take_profit 4.9822 (4.4 bp)

A 4.4 bp target against a 425 bp stop. It cannot book a profit worth having -- it is
gross 4.4 bp against an 8 bp round trip, so even a winning touch loses money -- and it
fires on the first tick that moves. The mirror case in the same sample has the target at
5.0025, ABOVE the 4.98 entry of a short: on the losing side.

Nothing in the codebase rejected either shape. \`bracket_problem\` checks the direction of
the levels and that the target is more than MIN_TARGET_BPS from the entry; it has no
opinion on the ratio between the two distances, so a plan risking 425 bp to win 4.4 bp
passes every gate. \`reanchor_plan\` inherits that hole: it reverts a translation only when
the result stops being valid, and "valid" never meant "worth taking".

These tests pin the missing constraint.
"""
import pytest

from app.trading import brackets


def short_plan(entry=4.98, stop=5.1918, target=4.9822):
    return {'side': 'SHORT', 'entry': entry, 'stop': stop, 'take_profit': target}


def long_plan(entry=100.0, stop=98.0, target=104.0):
    return {'side': 'LONG', 'entry': entry, 'stop': stop, 'take_profit': target}


def test_the_stored_short_with_a_4bp_target_is_refused():
    """The exact geometry from the trade record."""
    assert brackets.bracket_problem(short_plan()) is not None


def test_a_target_that_cannot_clear_the_round_trip_is_refused():
    # 4.4 bp of reward against an 8 bp round trip: a winning touch still loses money.
    assert brackets.bracket_problem(short_plan(target=4.9778)) is not None


def test_the_long_mirror_case_is_refused_too():
    # entry 100, stop 96 (400 bp risk), target 100.05 (5 bp reward)
    assert brackets.bracket_problem(long_plan(stop=96.0, target=100.05)) is not None


def test_a_healthy_plan_is_untouched():
    assert brackets.bracket_problem(long_plan()) is None


def test_the_policy_shaped_plan_still_passes():
    """scalp geometry: 30 bp stop floor, 1.6 reward:risk, LONG so the stop sits BELOW."""
    entry = 4.98
    stop = entry * (1 - 0.003)
    target = entry * (1 + 0.003 * 1.6)
    assert brackets.bracket_problem(long_plan(entry=entry, stop=stop, target=target)) is None


def test_the_minimum_ratio_is_stated_and_sane():
    assert 0 < brackets.MIN_REWARD_RISK_BPS <= 1.0


def test_the_target_must_at_least_cover_the_round_trip():
    """The economic floor, independent of the ratio: reward > cost."""
    assert brackets.MIN_TARGET_BPS >= 1.0
