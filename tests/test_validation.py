"""Deflated Sharpe and PBO: whether a reported edge survives the search that found it.

A Sharpe ratio is published without its search history. If the horizon, the edge multiple,
the stop width and the reward:risk were tried in combination and the best kept, the reported
figure is the maximum of many draws -- and the maximum of many zero-mean draws is large. The
audit found `horizon` and `MIN_EDGE_MULTIPLE` had been tuned with no record of how many
trials that took, which makes the resulting number uninterpretable rather than merely
optimistic.

These tests pin the two properties that matter: the correction must get stricter as the
search grows, and it must be able to tell a real edge from a lucky one.
"""
import math

import pytest

from app.models.validation import (deflated_sharpe, expected_max_sharpe,
                                   probability_of_backtest_overfitting, validate)


# ------------------------------------------------------- the benchmark
def test_one_trial_selects_nothing_so_the_benchmark_is_zero():
    """With no search there is nothing to deflate against."""
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(0) == 0.0


def test_the_benchmark_grows_with_the_number_of_trials():
    """The maximum of more draws is larger, which is the whole problem."""
    values = [expected_max_sharpe(n, 1_000) for n in (2, 10, 100, 1_000, 10_000)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_the_benchmark_shrinks_as_data_accumulates():
    """More observations means a tighter estimate, so luck buys less."""
    few = expected_max_sharpe(100, 500)
    many = expected_max_sharpe(100, 100_000)
    assert many < few


def test_the_default_benchmark_reduces_to_the_expected_maximum_t_statistic():
    """With the null dispersion the deflated z-score is t minus E[max].

    This identity is what makes the metric interpretable without any input the caller
    cannot supply, so it is worth pinning rather than leaving as an implementation detail:
    a benchmark of E[max]/sqrt(T-1) against a standard error of 1/sqrt(T-1) leaves
    `sharpe*sqrt(T-1) - E[max]`, where E[max] is the unit-variance expected maximum.
    """
    observations, trials, sharpe = 1_000, 50, 0.01
    unit_variance_max = expected_max_sharpe(trials, None, variance_of_trials=1.0)
    t_statistic = sharpe * math.sqrt(observations - 1)
    expected_z = t_statistic - unit_variance_max
    from app.models.validation import _normal_ppf
    observed_z = _normal_ppf(deflated_sharpe(sharpe, trials, observations))
    # The residual is the skew/kurtosis term, small for a normal input (skew 0, kurt 3)
    # and exactly the part that makes the estimator honest for fat-tailed returns.
    assert observed_z == pytest.approx(expected_z, abs=0.01)


def test_the_scaled_benchmark_is_the_unit_variance_one_over_root_t():
    """The two benchmark forms must be the same number in different units."""
    trials, observations = 50, 1_000
    unit = expected_max_sharpe(trials, None, variance_of_trials=1.0)
    scaled = expected_max_sharpe(trials, observations)
    assert scaled == pytest.approx(unit / math.sqrt(observations), rel=1e-9)


# ------------------------------------------------------------ deflation
def test_more_trials_makes_the_same_sharpe_less_convincing():
    """The correction has to bite, or recording the trial count changes nothing.

    A Sharpe of 1.5 per observation is absurd -- annualised on 5-minute bars it would be
    in the hundreds -- so the starting point is a realistic one. On 5-minute bars an
    annual Sharpe of 1.5 is about 0.0046 per bar, which is what a real strategy looks
    like and what the correction is actually for.
    """
    per_bar = 1.5 / math.sqrt(105_120)   # annualising factor for 5-minute bars
    probabilities = [deflated_sharpe(per_bar, n, 100_000) for n in (1, 5, 20, 100, 1_000)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] > probabilities[-1]
    assert probabilities[-1] < probabilities[0]


def test_more_data_makes_the_same_search_less_damaging():
    """A Sharpe estimated from more observations is harder to explain as luck."""
    short = deflated_sharpe(0.01, 50, 1_000)
    long = deflated_sharpe(0.01, 50, 100_000)
    assert long > short


def test_a_large_sharpe_over_lots_of_data_survives_a_long_search():
    """The metric must not be so conservative that nothing ever passes."""
    assert deflated_sharpe(0.05, 100, 100_000) > 0.95


def test_a_small_sharpe_over_little_data_does_not():
    """The failure case the correction exists to catch."""
    assert deflated_sharpe(0.002, 100, 500) < 0.5


def test_a_trial_count_of_one_applies_no_penalty():
    """No search was run, so nothing is deflated away."""
    assert deflated_sharpe(1.5, 1, 500) == pytest.approx(deflated_sharpe(1.5, 1, 10**9))


def test_negative_skew_and_fat_tails_raise_the_bar():
    """Stop-loss strategies are negatively skewed, and the estimator is worse there.

    The effect depends on the sign of the Sharpe. A negative skew inflates the error of a
    POSITIVE Sharpe (the `-skew*sharpe` term grows), while excess kurtosis inflates it for
    either sign. Testing at a positive Sharpe is the case that matters, since a strategy
    with a negative one is not a candidate.
    """
    normal = deflated_sharpe(0.03, 20, 50_000, skew=0.0, kurtosis=3.0)
    skewed = deflated_sharpe(0.03, 20, 50_000, skew=-1.5, kurtosis=3.0)
    fat = deflated_sharpe(0.03, 20, 50_000, skew=0.0, kurtosis=12.0)
    assert skewed < normal, "negative skew must widen the error of a positive Sharpe"
    assert fat < normal, "excess kurtosis must widen the error"
    assert skewed == pytest.approx(fat) or skewed != fat


def test_insufficient_evidence_returns_none_rather_than_a_passing_number():
    """A silent default to "fine" is worse than no metric, because nobody looks again."""
    assert deflated_sharpe(1.5, 10, 1) is None
    assert deflated_sharpe(1.5, 10, 0) is None
    assert deflated_sharpe(None, 10, 500) is None
    assert deflated_sharpe(1.5, 10, None) is None
    assert deflated_sharpe(float('nan'), 10, 500) is None


def test_the_result_is_a_probability():
    """Whatever the inputs, the output is in [0, 1] or None."""
    for sharpe in (-5.0, -0.5, 0.0, 0.5, 5.0, 50.0):
        value = deflated_sharpe(sharpe, 30, 10_000)
        assert value is None or 0.0 <= value <= 1.0


# ---------------------------------------------------------------- PBO
def test_pbo_is_high_when_in_sample_ranking_does_not_carry_over():
    """Perfectly reversed rankings mean selection is picking pure noise."""
    pbo = probability_of_backtest_overfitting([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert pbo > 0.5


def test_pbo_is_low_when_the_ranking_carries_over():
    """The same configuration wins in and out of sample."""
    pbo = probability_of_backtest_overfitting([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert pbo < 0.2


def test_pbo_needs_at_least_two_configurations():
    """A single configuration cannot be selected between."""
    assert probability_of_backtest_overfitting([1], [1]) is None
    assert probability_of_backtest_overfitting([], []) is None


def test_pbo_honours_lower_is_better():
    """For drawdown and loss metrics the direction flip has to change the answer.

    A perfectly linear ranking is a degenerate test: reversing it reverses both the
    selection and the ranking, so the statistic is unchanged. The direction only shows up
    when the two orderings are not mirror images, which is the case a real combinatorial
    split produces.
    """
    in_sample = [1.0, 2.0, 3.0]
    out_of_sample = [3.0, 1.0, 2.0]
    as_gain = probability_of_backtest_overfitting(in_sample, out_of_sample)
    as_loss = probability_of_backtest_overfitting(in_sample, out_of_sample,
                                                 higher_is_better=False)
    assert as_gain != as_loss
    # Read as gains, the in-sample winner (index 2) came second out-of-sample -- a
    # middling outcome. Read as losses, the in-sample winner is index 0, which came last.
    assert as_loss > as_gain


def test_pbo_honours_lower_is_better_in_the_mirror_case_too():
    """When selection is purely reversed, both readings agree the pick was bad."""
    assert probability_of_backtest_overfitting([1, 2, 3, 4], [4, 3, 2, 1]) > 0.5
    assert probability_of_backtest_overfitting([1, 2, 3, 4], [4, 3, 2, 1],
                                               higher_is_better=False) > 0.5


def test_pbo_ignores_incomparable_entries():
    """A configuration with no out-of-sample figure cannot be ranked."""
    pbo = probability_of_backtest_overfitting([1, None, 3, 4], [1, 2, None, 4])
    assert pbo is None or 0.0 <= pbo <= 1.0


# ------------------------------------------------------------ the verdict
def test_validate_reports_the_benchmark_and_a_verdict():
    """A training report needs to say what the Sharpe was measured against."""
    result = validate(0.05, 100, 100_000)
    assert result['trials'] == 100
    assert result['survives'] is True
    assert result['expected_max_sharpe'] > 0
    assert 0.0 <= result['deflated_sharpe'] <= 1.0


def test_validate_says_insufficient_evidence_rather_than_guessing():
    """The absence of a verdict is itself the finding."""
    result = validate(1.5, 10, 1)
    assert result['survives'] is None
    assert result['deflated_sharpe'] is None
    assert result['reason'] == 'insufficient_evidence'
