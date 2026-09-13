"""How much of a backtest's Sharpe ratio is selection, and how likely it is to be noise.

A Sharpe ratio is reported without its search history. If the horizon, the edge multiple,
the stop width and the reward:risk were tried in some combination and the best one kept,
the reported figure is the maximum of many draws, and the maximum of many draws is large
even when every draw has zero mean. The audit noted that `horizon` and `MIN_EDGE_MULTIPLE`
had been tuned with no record of how many trials that took, which makes the resulting
number uninterpretable rather than merely optimistic.

Two standard corrections, both from Lopez de Prado, *Advances in Financial Machine
Learning*, ch. 8:

* **Deflated Sharpe Ratio (DSR)** -- the probability that the observed Sharpe is genuinely
  above zero, given that it was selected as the best of N trials. It asks a sharper
  question than the raw ratio: not "is this good?" but "is this better than the best of N
  coin flips would look?"
* **Probability of Backtest Overfitting (PBO)** -- the share of combinatorial splits in
  which the configuration that performed best in-sample lands at or below median
  out-of-sample. It measures whether selection is picking signal or noise.

Both are deliberately conservative and both return None rather than a comforting number
when there is not enough evidence to compute them. A metric that silently defaults to a
passing value is worse than no metric, because the operator stops looking.
"""
import math

# Euler-Mascheroni constant, used by the expected-maximum approximation below.
EULER_GAMMA = 0.5772156649015329


def _normal_cdf(value):
    """Standard normal CDF, via the error function. No SciPy in this environment."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_ppf(probability):
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to about 1.15e-9, which is far beyond anything this metric needs, and it
    avoids a dependency for the one place a quantile is required.
    """
    if probability <= 0.0:
        return float("-inf")
    if probability >= 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1.0 - 0.02425
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if probability > high:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = probability - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def expected_max_sharpe(trials, observations=None, variance_of_trials=None):
    """The Sharpe the best of `trials` independent zero-skill strategies would show.

    This is the benchmark the deflated ratio is measured against. With one trial it is
    zero (nothing was selected); it grows like sqrt(2 ln N), so a long search produces a
    respectable-looking number from pure noise.

    `variance_of_trials` is the cross-sectional variance of the per-observation Sharpe
    estimates that were compared. It is usually not known -- a tuning run keeps the winner,
    not the table -- so it defaults to the null dispersion 1/`observations`, which is the
    variance of a single zero-skill Sharpe estimate.

    That default is what makes the result interpretable. Working it through: the benchmark
    is E[max]/sqrt(T) and the standard error is about 1/sqrt(T), so the deflated z-score
    reduces to `sharpe*sqrt(T) - E[max]` -- the ordinary t-statistic of the Sharpe minus the
    expected best of N standard normals, which is `expected_max_sharpe(trials, 1)` and NOT
    the value this function returns for a given `observations`. The correction then reads
    as "how many sigma above what a search of this size would have produced anyway", and
    it needs no assumption the caller cannot supply.

    A fixed default of 1.0 would instead make the benchmark about 2.5 Sharpe units for any
    search larger than a hundred -- which for a per-observation Sharpe is astronomical --
    driving the probability to zero for every strategy and making the metric useless
    rather than conservative.
    """
    trials = int(trials or 0)
    if trials <= 1:
        return 0.0
    if variance_of_trials is None:
        count = int(observations or 0)
        variance = (1.0 / count) if count > 1 else 1.0
    else:
        variance = max(0.0, float(variance_of_trials))
    if variance <= 0:
        return 0.0
    sigma = math.sqrt(variance)
    # E[max of N standard normals], the Gumbel approximation. Exact enough for N in the
    # range anyone actually searches, and it cannot overflow the way a direct maximum of
    # the normal quantiles can.
    quantile = (1.0 - 1.0 / trials)
    first = _normal_ppf(quantile)
    second = _normal_ppf(1.0 - 1.0 / (trials * math.e))
    return sigma * ((1.0 - EULER_GAMMA) * first + EULER_GAMMA * second)


def deflated_sharpe(sharpe, trials, observations, skew=0.0, kurtosis=3.0,
                    variance_of_trials=None):

    """Probability the observed Sharpe is real, given how many trials produced it.

    Returns the DSR in [0, 1], or None when it cannot be computed honestly. A value below
    0.95 means the Sharpe is not distinguishable from the best of `trials` lucky draws at
    the conventional threshold, and the right response is more data or fewer trials -- not
    a lower threshold.

    `sharpe` and the benchmark are both per-observation, not annualised. Mixing the two is
    the easiest way to get a flattering answer, so the caller is required to be explicit
    and this function does no annualising of its own.
    """
    if sharpe is None or observations is None:
        return None
    try:
        sharpe = float(sharpe); observations = int(observations)
        skew = float(skew); kurtosis = float(kurtosis)
    except (TypeError, ValueError):
        return None
    if observations < 2 or not math.isfinite(sharpe):
        return None
    benchmark = expected_max_sharpe(trials, observations, variance_of_trials)
    # Standard error of a Sharpe estimate under non-normal returns. The skew and kurtosis
    # terms matter for the fat-tailed, negatively skewed distributions that stop-loss
    # strategies produce, where the normal case understates the error.
    numerator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if numerator <= 0:
        return None
    standard_error = math.sqrt(numerator / (observations - 1))
    if standard_error <= 0 or not math.isfinite(standard_error):
        return None
    z = (sharpe - benchmark) / standard_error
    if not math.isfinite(z):
        return None
    return _normal_cdf(z)


def probability_of_backtest_overfitting(in_sample, out_of_sample, higher_is_better=True):
    """Share of splits where the in-sample winner underperforms out-of-sample.

    `in_sample` and `out_of_sample` are equal-length sequences of per-configuration
    performance, one entry per configuration, already aggregated across the
    combinatorial splits. The statistic is the rank of the in-sample best within the
    out-of-sample distribution: if selecting the best in-sample configuration tells you
    nothing, its out-of-sample rank is uniform and PBO tends to 0.5.

    Returns None when there are fewer than two configurations, since a single
    configuration cannot be selected between and the question does not arise.
    """
    pairs = [(i, o) for i, o in zip(in_sample or (), out_of_sample or ())
             if i is not None and o is not None
             and math.isfinite(float(i)) and math.isfinite(float(o))]
    if len(pairs) < 2:
        return None
    direction = 1.0 if higher_is_better else -1.0
    ranked = sorted(pairs, key=lambda pair: direction * float(pair[0]))
    best_in_sample = ranked[-1]
    values = sorted(direction * float(o) for _, o in pairs)
    rank = values.index(direction * float(best_in_sample[1]))
    # Fraction of configurations the selected one failed to beat out-of-sample.
    return 1.0 - (rank + 0.5) / len(values)


def validate(sharpe, trials, observations, skew=0.0, kurtosis=3.0, threshold=0.95):
    """The deflated Sharpe with a verdict, for a training report to act on.

    The verdict is advisory: it says whether the Sharpe survives the correction, and it is
    reported rather than enforced, because a model can be worth shipping on grounds the
    Sharpe does not capture and silently refusing to train is worse than a warning.
    """
    dsr = deflated_sharpe(sharpe, trials, observations, skew, kurtosis)
    if dsr is None:
        return {"deflated_sharpe": None, "trials": int(trials or 0),
                "benchmark_sharpe": None, "survives": None,
                "reason": "insufficient_evidence"}
    return {"deflated_sharpe": round(dsr, 6), "trials": int(trials or 0),
            "expected_max_sharpe": round(expected_max_sharpe(trials, observations), 6),
            "observations": int(observations or 0),
            "survives": dsr >= float(threshold), "threshold": float(threshold)}
