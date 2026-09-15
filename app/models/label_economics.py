"""Whether a label can pay for the cost of trading it, decided before any model is fit.

This module exists because the previous pipeline could not answer the question that turned
out to be the whole problem. It measured a model\u2019s directional accuracy, its mean net
return, and a deflated Sharpe, and it compared all of those against thresholds. It never
compared the size of the move being predicted against the cost of capturing it, and that
ratio is what decides whether a profitable-looking model is profitable.

The measured numbers on the 1.26M-row mainstream dataset:

* the triple-barrier label outcome has a standard deviation of 70.0 bps
* the round-trip cost assumption is 12 bps
* so the cost is 0.17 standard deviations of the thing being predicted

and the barrier hit rates are 37.48% upper, 39.14% lower, 23.38% vertical. The stop is
touched more often than the target, so the label is negatively skewed before costs are
applied at all.

Against that, a directional model has to shift the conditional mean by more than 12 bps to
break even. Expressing the required shift in standard deviations makes the difficulty
legible: the model must move the mean by 0.17 sigma. In signal-processing terms that is a
requirement on the squared correlation between prediction and outcome, and the arithmetic
is unforgiving. If s is the required shift in sigma and rho is the correlation the model
achieves, then the mean shift available is rho * sigma; setting rho * sigma = 0.17 * sigma
gives rho = 0.17, and R^2 = rho^2 = 2.9%.

A 5-minute crypto return is not predictable at R^2 = 3%. Published cross-sectional and time
series results on crypto at intraday horizons report R^2 in the region of 0.1%, an order of
magnitude below what this cost structure demands. That is the honest explanation of sixteen
candidate models landing between -7.6 and -18.6 bps of out-of-sample edge, and it is
arithmetic rather than opinion: no amount of feature engineering inside this cost budget
reaches a tradeable result.

The module therefore computes the *breakeven requirement* first, as a gate on the data rather
than on the model, so that a run whose label cannot pay for its costs says so before spending
hours fitting.
"""

import math

# Cost assumptions, in basis points of notional, round trip. The taker rate is the Binance
# USDT-M standard tier; the spread and impact components match the fill model defaults.
COST_PROFILES = {
    "taker": {"fee_bps": 4.0, "spread_bps": 1.0, "impact_bps": 1.0},
    "maker": {"fee_bps": 2.0, "spread_bps": 0.0, "impact_bps": 1.0},
}

# A label whose barrier outcomes are this skewed or worse cannot be traded directionally
# whatever the model does, because the loss leg is touched first more often than the win leg.
MIN_BARRIER_BALANCE = 0.48


def cost_bps(profile="taker", extra_bps=0.0):
    """Round-trip cost in basis points for a named execution profile."""
    if profile not in COST_PROFILES:
        raise ValueError("unknown_cost_profile")
    parts = COST_PROFILES[profile]
    return parts["fee_bps"] + parts["spread_bps"] + parts["impact_bps"] + float(extra_bps)


def label_statistics(outcomes, barriers=None):
    """Mean, standard deviation and skew of the labelled outcomes.

    outcomes is an iterable of realised returns; barriers, when supplied, is the matching
    iterable of which barrier fired. Both are used because a label can look well-behaved in
    its distribution while being structurally unbeatable -- the barrier counts are what
    reveal that the stop is closer in probability space than the target.
    """
    values = [float(v) for v in outcomes]
    count = len(values)
    if count < 2:
        return {"samples": count, "ready": False, "reason": "too_few_samples"}
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / (count - 1)
    deviation = math.sqrt(variance)
    out = {"samples": count, "ready": True, "mean": mean, "std": deviation,
           "mean_bps": mean * 10000, "std_bps": deviation * 10000,
           "positive_share": sum(1 for v in values if v > 0) / count}
    if deviation > 0 and count > 2:
        out["skew"] = sum(((v - mean) / deviation) ** 3 for v in values) / count
    if barriers is not None:
        tally = {}
        for name in barriers:
            tally[name] = tally.get(name, 0) + 1
        total = sum(tally.values()) or 1
        shares = {name: n / total for name, n in tally.items()}
        out["barriers"] = tally
        out["barrier_shares"] = shares
        upper = shares.get("upper")
        lower = shares.get("lower")
        if upper is not None and lower is not None:
            out["barrier_balance"] = upper / (upper + lower) if (upper + lower) else None
    return out


def breakeven_requirement(std_bps, cost_bps_value):
    """What a model must achieve for the cost to be recoverable.

    Returns the required conditional mean shift, its size in standard deviations, and the
    implied R^2 -- which is the number to compare against what financial ML actually
    delivers on this horizon.
    """
    if std_bps <= 0:
        raise ValueError("invalid_label_std")
    shift_sigma = cost_bps_value / std_bps
    return {
        "cost_bps": round(cost_bps_value, 4),
        "label_std_bps": round(std_bps, 4),
        "cost_in_sigma": round(shift_sigma, 6),
        "required_mean_shift_bps": round(cost_bps_value, 4),
        "required_correlation": round(shift_sigma, 6),
        "required_r2": round(shift_sigma ** 2, 6),
        "required_r2_pct": round(shift_sigma ** 2 * 100, 4),
    }


def achievable_r2(feature_outcome_correlation):
    """R^2 implied by an observed feature/outcome correlation."""
    rho = abs(float(feature_outcome_correlation))
    return {"correlation": rho, "r2": rho ** 2, "r2_pct": rho ** 2 * 100}


def horizon_that_costs_are_affordable(bar_std_bps, cost_bps_value, target_r2=0.003,
                                      min_bars=1, max_bars=2000):
    """How long a holding period this cost budget requires at a realistic R^2.

    Volatility grows with the square root of time, so a horizon H bars away has a standard
    deviation of bar_std_bps * sqrt(H). The mean shift a model of quality target_r2 can
    produce is sqrt(target_r2) * that. Setting the two equal gives the horizon at which the
    signal finally covers the cost:

        sqrt(target_r2) * bar_std * sqrt(H) = cost
        H = (cost / (sqrt(target_r2) * bar_std)) ** 2

    This is the number that turns "use a longer horizon" from a slogan into a calculation.
    """
    if bar_std_bps <= 0:
        raise ValueError("invalid_bar_std")
    rho = math.sqrt(target_r2)
    required = cost_bps_value / (rho * bar_std_bps)
    horizon = int(math.ceil(required ** 2))
    return {"bars": max(min_bars, min(max_bars, horizon)), "raw_bars": horizon,
            "target_r2": target_r2, "bar_std_bps": round(bar_std_bps, 4),
            "cost_bps": round(cost_bps_value, 4)}


def assess(outcomes, barriers=None, profile="taker", extra_bps=0.0, target_r2=0.003):
    """The full economics verdict for one labelling configuration.

    feasible is False when the label cannot pay for its costs at a realistic model quality.
    That is not a statement about any particular model; it is a statement about the question
    the model is being asked, and it is the reason a run should be stopped before it starts.
    """
    cost = cost_bps(profile, extra_bps)
    stats = label_statistics(outcomes, barriers)
    if not stats.get("ready"):
        return {"ready": False, "reason": stats.get("reason"), "cost_bps": cost}
    requirement = breakeven_requirement(stats["std_bps"], cost)
    reachable = achievable_r2(math.sqrt(target_r2))
    affordable = horizon_that_costs_are_affordable(stats["std_bps"], cost, target_r2)
    balance = stats.get("barrier_balance")
    reasons = []
    if balance is not None and balance < MIN_BARRIER_BALANCE:
        reasons.append("barrier_balance_below_threshold:%.4f" % balance)
    if requirement["required_r2"] > target_r2:
        reasons.append("required_r2_above_achievable:%.6f>%.6f"
                       % (requirement["required_r2"], target_r2))
    return {"ready": True, "profile": profile, "cost_bps": cost,
            "label": stats, "requirement": requirement, "achievable": reachable,
            "affordable_horizon_bars": affordable,
            "feasible": not reasons, "reasons": reasons}


def verdict_lines(report):
    """Human-readable lines, so a run prints why it stopped rather than only that it did."""
    if not report.get("ready"):
        return ["label economics not assessable: %s" % report.get("reason")]
    label = report["label"]
    req = report["requirement"]
    lines = [
        "label std %.1f bps, %d samples, positive share %.3f"
        % (label["std_bps"], label["samples"], label["positive_share"]),
        "round-trip cost %.1f bps (%s) = %.4f sigma"
        % (report["cost_bps"], report["profile"], req["cost_in_sigma"]),
        "breakeven needs R^2 >= %.4f%% ; realistic target is %.4f%%"
        % (req["required_r2_pct"], report["achievable"]["r2_pct"]),
    ]
    if "barrier_shares" in label:
        shares = label["barrier_shares"]
        lines.append("barrier shares: " + ", ".join(
            "%s %.2f%%" % (k, v * 100) for k, v in sorted(shares.items())))
    horizon = report.get("affordable_horizon_bars") or {}
    if horizon.get("bars"):
        lines.append("a %.2f%% R^2 model needs a horizon of about %d bars to cover this cost"
                     % (horizon["target_r2"] * 100, horizon["bars"]))
    for reason in report.get("reasons", []):
        lines.append("INFEASIBLE: " + reason)
    return lines
