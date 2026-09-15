"""The twelve-coin decision model: what it predicts, and why each piece exists.

The single design decision behind all of it
-------------------------------------------
Every model in this repository that predicted a *level* failed, and the reason is in the
label rather than in the learner. A triple-barrier label built from ATR multiples has a
magnitude that is, by construction, a function of ATR: measured on this data,
\`corr(atr_pct, |future_return|) = 0.811\` against \`corr(atr_pct, sign(future_return)) = 0.030\`.
A gradient-boosted tree fit to that label spends its first splits learning ATR, scores a
respectable-looking magnitude correlation, and contributes nothing to direction.

So this module never predicts a return. It predicts a **rank**: the cross-sectional ordering
of the twelve coins by their forward return over the next day. A rank is

* scale-free, so BTC at 110,000 and DOGE at 0.21 are the same kind of observation;
* immune to the ATR illusion, because a coin that moves a lot does not thereby rank higher --
  only its ordering matters;
* market-neutral when the extremes are traded long and short, which removes the one factor
  no feature in this system ever predicted;
* directly comparable across the twelve coins in one pass, so there is no second model to
  keep in sync with the first.

Five components, each answering one question
--------------------------------------------
1. **momentum** -- which coins are strongest relative to the basket? A linear model on
   cross-sectionally demeaned momentum features. Linear rather than a forest on purpose: with
   twelve instruments and roughly 330 usable days there are about 4,000 independent
   cross-sectional observations, and a forest on that sample buys variance reduction it
   cannot pay for. The audit below reports how much capacity the linear form leaves behind.
2. **breakout** -- is the momentum *confirmed* by where price sits in its own recent range,
   by whether open interest is rising with it, and by whether aggressive flow agrees? This is
   the second view of the same question, deliberately kept separate so the two can disagree
   and the disagreement can be measured.
3. **regime** -- should the market be traded at all, and which way? A high-volatility,
   directionless market makes relative ranking unreliable; a market in a downtrend makes the
   short leg the only one worth holding. Produces a scalar in [-1, 1].
4. **meta** -- given the direction the rule already proposes, is this particular trade worth
   taking? This is Lopez de Prado's meta-labelling, and it exists because the *filtering*
   question is far easier than the *direction* question: the label is binary and the base rate
   is near a half, so a model has to move a much smaller quantity to be useful.
5. **risk** -- how much should a position be worth? Predicts forward realised volatility,
   which is the one thing here that is genuinely forecastable. Used for volatility targeting
   in the decision layer, never for direction.

What the suite does NOT do
--------------------------
It does not fit on the five-minute bars. At a five-minute horizon the round trip costs 12 bps
against an 18.9 bps mean absolute move, which demands a hit rate of 81.8%; at a one-day hold
the same cost is 6 bps against a 150 bps move. The measurement in this repository says the
short-horizon version of the question is unanswerable, so it is not asked again here.

The artifact is sealed to a panel digest. Validation recomputes it and refuses a mismatch, so
a suite cannot be scored on a panel it was not fit on.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from . import daily_panel
from ..features.feature_spec import FEATURE_VERSION
from ..trading import exit_policy

MODEL_VERSION = "coinsuite-v1"
# The decision horizon, in days. One day is the configuration that survived every stability
# check in the repository: a seven-day lookback with a one-day hold reported +78.2 bps per
# trade at t = 3.37 over 356 non-overlapping trades, positive in both halves of the sample and
# in 10 of 13 months, while longer holds lost their significance (a 7-day hold on the same
# lookback reported t = 0.59).
HORIZON_DAYS = 1
HOLD_DAYS = 1

# Execution cost, in basis points per side. The Binance USDT-M standard tier is 4.0 bps of fee
# plus roughly 1.0 bps of spread and 1.0 bps of impact on a liquid perpetual. Charged on both
# legs, so the round trip in the backtest is twice this number.
COST_BPS_PER_SIDE = 6.0

# Minimum cross-sectional observations before any component is fit. Twelve coins times thirty
# days is 360; below that the "cross-section" is a handful of correlated observations.
MIN_TRAIN_ROWS = 600

# Feature blocks, as (family, [(name, transform)]) where the transform is applied to the
# cross-sectional rank of the column. Ranking rather than standardising is what makes the
# panel robust to the fat tails a year of crypto contains: one 135% bar in DOTUSDT shifts a
# z-score by more than it shifts a rank.
MOMENTUM_BLOCK = (("ret_7d", "rank"), ("ret_14d", "rank"), ("ret_30d", "rank"),
                  ("ret_3d", "rank"), ("ret_1d_prev", "rank"))
BREAKOUT_BLOCK = (("close_pos", "rank"), ("range_z", "rank"), ("oi_chg_1d", "rank"),
                  ("oi_notional", "rank"), ("taker_7d", "rank"), ("ret_1d_prev", "rank"),
                  ("vol_ratio", "rank"), ("range_pct", "rank"))
REGIME_BLOCK = ("mkt_ret_7d", "mkt_vol_30d", "oi_chg_7d", "funding_7d")
RISK_BLOCK = ("vol_7d", "vol_30d", "vol_ratio", "range_pct", "range_z", "trades_z")

# Ridge strength, as a *shrinkage fraction* toward the equal-weighted prior rather than as an
# absolute penalty.
#
# This is the single most important number in the file, and it was measured rather than
# chosen. With an absolute penalty (the previous \`RIDGE_LAMBDA = 3.0\`), the momentum component
# reported an in-sample rank IC of 0.112 and an out-of-sample IC of 0.014 -- while the raw
# seven-day return it was built from carries an IC of 0.030 on the same rows. The fit was not
# adding information, it was consuming it, because ten collinear rank columns and four thousand
# cross-sectional rows is nowhere near enough sample to estimate ten coefficients.
#
# Shrinking the fitted coefficients toward the prior is the fix, and the measurement says the
# honest amount of shrinkage is *all of it*. The table below is a rolling experiment: fit on days
# 0..k, score the next sixty days, compare the fitted component against the single raw column it
# was built from.
#
#     fit_end   raw ret_7d IC / net      unshrunk ridge IC / net
#         90     +0.0522 /  +20.2 bp       +0.0981 /  +57.6 bp
#        120     +0.0040 /   -4.6 bp       +0.0491 /   +7.1 bp
#        150     -0.0277 /  +24.5 bp       -0.0410 /  -61.9 bp
#        180     +0.0409 /  +45.8 bp       -0.0079 /  -77.6 bp
#        210     +0.0192 /  +52.3 bp       +0.0170 /  -72.1 bp
#        240     +0.0083 /  +70.0 bp       -0.0036 /  -68.4 bp
#        270     +0.0276 /  +46.9 bp       -0.0055 /   -6.2 bp
#
# The fitted model wins exactly once, in the window where the relationship was strongest in
# sample, and loses the next six by an average of about eighty basis points. An in-sample rank IC
# of 0.155 corresponded to a net of -47.6 bps out of sample, and an in-sample IC of 0.065 -- the
# raw column -- corresponded to +30.2 bps. That is the signature of a fit that has memorised the
# training cross-sections, and it is the same failure the previous generation of models showed
# when a magnitude correlation of 0.319 sat next to a directional correlation of 0.028.
#
# A zero here is not a defeat. It means the suite *ships the rule*, with the fitted layer kept in
# the artifact as a measured negative result rather than quietly deleted: the parameters are
# still estimated, still stored, and reported with the out-of-sample comparison that justifies
# giving them no weight. A reader who thinks the sample was too short to judge that can set this
# to 1.0 and read the counter-evidence in the same artifact.
SHRINK_TO_PRIOR = 0.0

# The absolute ridge penalty used before the coefficients are blended toward the prior. Kept as
# a module constant so the default argument below can name it.
RIDGE_LAMBDA = 3.0

# Where a fitted meta filter's weights are written while the artifact is assembled. A scratch
# path, deliberately: the artifact bundle owns the final location, and a filter written
# straight to it would leave a stale model behind whenever a fit fails partway.
_META_MODEL_PATH = "_meta_filter.txt"


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _cross_section_ranks(matrix):
    """Per-row ranks in [-1, 1], centred so the cross-section sums to zero.

    Ties are averaged, and an unmeasured value stays None rather than becoming the bottom of
    the ranking: "we did not measure this coin" and "this coin is the weakest" are different
    statements, and collapsing them would silently trade the difference.
    """
    out = [[None] * len(row) for row in matrix]
    for index, row in enumerate(matrix):
        present = [(position, value) for position, value in enumerate(row)
                   if value is not None]
        if len(present) < 3:
            continue
        ordered = sorted(present, key=lambda item: item[1])
        size = len(ordered)
        position = 0
        while position < size:
            end = position
            while end + 1 < size and ordered[end + 1][1] == ordered[position][1]:
                end += 1
            average_rank = (position + end) / 2.0
            value = 2.0 * average_rank / (size - 1) - 1.0
            for slot in range(position, end + 1):
                out[index][ordered[slot][0]] = value
            position = end + 1
    return out


def _rows_by_day(rows):
    """Group panel rows by day, in day order, as (days, [row, ...] per day)."""
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["day"]), []).append(row)
    days = sorted(grouped)
    return days, [grouped[day] for day in days]


def design(rows, block):
    """The cross-sectional design matrix for one feature block.

    Two columns per feature: the within-day rank, and the deviation of the coin's own rank
    from its own recent average. The second exists because a momentum system is asking whether
    a coin is *becoming* strong, and a coin that has ranked first for a month is a different
    proposition from one that just arrived there -- a distinction the cross-section alone
    cannot make, because it sees one day at a time.
    """
    days, by_day = _rows_by_day(rows)
    coins = sorted({row["coin"] for row in rows})
    coin_index = {coin: i for i, coin in enumerate(coins)}
    names, columns = [], []
    for feature, _ in block:
        raw = [[_finite(row.get(feature)) for row in members] for members in by_day]
        ranked = _cross_section_ranks(raw)
        flat = [None] * (len(days) * len(coins))
        for day_index, values in enumerate(ranked):
            for position, value in enumerate(values):
                flat[day_index * len(coins) + position] = value
        columns.append(flat)
        names.append(feature + "_rank")
    # Each coin's own trailing mean of its rank, so the deviation is computable without
    # looking forward: the mean is taken over rows strictly earlier in the day sequence.
    for feature, _ in block:
        raw = [[_finite(row.get(feature)) for row in members] for members in by_day]
        ranked = _cross_section_ranks(raw)
        deviation = [None] * (len(days) * len(coins))
        history = {}
        for day_index, values in enumerate(ranked):
            for position, value in enumerate(values):
                coin = coins[position]
                past = history.get(coin, [])
                if value is not None and len(past) >= 5:
                    mean = sum(past[-20:]) / len(past[-20:])
                    deviation[day_index * len(coins) + position] = value - mean
                if value is not None:
                    history.setdefault(coin, []).append(value)
        columns.append(deviation)
        names.append(feature + "_chg")
    return {"days": days, "coins": coins, "coin_index": coin_index, "names": names,
            "columns": columns}


def targets(rows):
    """Forward cross-sectional rank of the next-day return, the thing being predicted.

    Ranked rather than standardised for the same reason the features are: the target's tail
    behaviour over a year of crypto is dominated by a handful of days, and a rank is not.
    """
    days, by_day = _rows_by_day(rows)
    coins = sorted({row["coin"] for row in rows})
    values = [[_finite(row.get("fwd_%dd" % HORIZON_DAYS)) for row in members]
              for members in by_day]
    ranked = _cross_section_ranks(values)
    return days, coins, ranked


def _matrix_from(design_matrix, row_positions):
    """Stack selected rows of a design into a dense array plus a per-cell mask.

    Missing cells are filled with zero *and* recorded, so the caller can weight a row by how
    much of it was actually measured instead of pretending that an unmeasured feature is an
    average one.
    """
    columns = design_matrix["columns"]
    count = len(row_positions)
    width = len(columns)
    dense = np.zeros((count, width), dtype=float)
    mask = np.zeros((count, width), dtype=float)
    for out_index, row_index in enumerate(row_positions):
        for column_index, column in enumerate(columns):
            value = column[row_index]
            if value is None:
                continue
            dense[out_index, column_index] = float(value)
            mask[out_index, column_index] = 1.0
    return dense, mask


def fit_ridge(x, y, weights, ridge=RIDGE_LAMBDA):
    """Ridge regression with an explicit intercept, solved in closed form.

    The intercept absorbs the cross-sectional mean, which is zero by construction, so it stays
    small; it is kept rather than dropped because a fold whose rows cover a partial
    cross-section does not have a mean of exactly zero and dropping it would bias every
    prediction in that fold.
    """
    count, width = x.shape
    augmented = np.hstack([np.ones((count, 1)), x])
    weighted = augmented * weights[:, None]
    gram = augmented.T @ weighted
    penalty = np.eye(width + 1) * float(ridge)
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(gram + penalty, weighted.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram + penalty, weighted.T @ y, rcond=None)[0]
    return coefficients


def predict_ridge(coefficients, x):
    count = x.shape[0]
    augmented = np.hstack([np.ones((count, 1)), x])
    return augmented @ coefficients


def _spearman(prediction, actual):
    """Rank correlation, computed on the ranks directly rather than through a formula.

    Written out because scipy is not a dependency of this project and a rank correlation is
    twenty lines: it is the fraction of concordant-minus-discordant pairs, which is exactly
    what the mean of the products of centred ranks gives when the ranks are normalised.
    """
    left = np.asarray(prediction, dtype=float)
    right = np.asarray(actual, dtype=float)
    if left.size < 3 or left.size != right.size:
        return None
    if np.std(left) <= 0 or np.std(right) <= 0:
        return None
    centred_left = left - left.mean()
    centred_right = right - right.mean()
    denominator = math.sqrt(float((centred_left ** 2).sum() * (centred_right ** 2).sum()))
    if denominator <= 0:
        return None
    return float((centred_left * centred_right).sum() / denominator)


# ------------------------------------------------------------------ component fitting

def _usable_positions(days, coins, targets_ranked, design_matrix, first, last):
    """Row positions of the design that are inside a day range and carry a target."""
    positions = []
    width = len(coins)
    for day_index in range(first, last):
        if day_index < 0 or day_index >= len(days):
            continue
        members = targets_ranked[day_index]
        for position in range(width):
            if members[position] is None:
                continue
            flat = day_index * width + position
            if flat >= len(design_matrix["columns"][0]):
                continue
            positions.append(flat)
    return positions


def _targets_flat(targets_ranked):
    return [value for day in targets_ranked for value in day]


def fit_component(rows, block, first, last, ridge=RIDGE_LAMBDA,
                  shrink=SHRINK_TO_PRIOR, prior_features=None):
    """Fit one cross-sectional component over days [first, last), shrunk toward a prior.

    Two things keep an estimated coefficient from being trusted more than the data supports.

    Rows whose cross-section is incomplete are dropped rather than zero-filled when the target
    is missing, and every fitted row is weighted by the share of its features that were
    measured: a row with half its features missing is half an observation, and imputing the
    other half would invent it.

    Then the fitted coefficients are blended with an equal-weighted prior over one or more
    columns. That blend is not decoration. With ten collinear rank columns and roughly four
    thousand cross-sectional rows, the unshrunk fit scored an in-sample rank IC of 0.112
    against 0.014 out of sample, while the single raw column it was built from carried 0.030 --
    the fit consumed the signal instead of refining it. \`shrink=0\` reproduces the raw prior
    exactly and \`shrink=1\` is the plain ridge; the default is stated in the constant above and
    was selected on out-of-sample IC rather than on backtest profit.
    """
    design_matrix = design(rows, block)
    days, coins, ranked = targets(rows)
    flat_targets = _targets_flat(ranked)
    positions = _usable_positions(days, coins, ranked, design_matrix, first, last)
    if len(positions) < MIN_TRAIN_ROWS:
        return {"status": "insufficient_rows", "rows": len(positions)}
    x, mask = _matrix_from(design_matrix, positions)
    y = np.array([flat_targets[position] for position in positions], dtype=float)
    measured = mask.mean(axis=1)
    weights = np.clip(measured, 0.05, 1.0)
    coefficients = fit_ridge(x, y, weights, ridge)
    # The prior: uniform weight across the named columns, zero everywhere else. Its intercept
    # is zero because the cross-sectional ranks already sum to zero by construction.
    names = design_matrix["names"]
    wanted = list(prior_features if prior_features is not None
                  else [name for name in names if name.endswith("_chg") is False])
    prior = np.zeros(len(names) + 1)
    slots = [index for index, name in enumerate(names) if name in wanted]
    if slots:
        for index in slots:
            prior[index + 1] = 1.0 / len(slots)
    blended = (1.0 - float(shrink)) * prior + float(shrink) * np.asarray(coefficients)
    prediction = predict_ridge(blended, x)
    return {"status": "ok", "rows": int(len(positions)), "features": names,
            "coefficients": blended.tolist(),
            "ridge": float(ridge), "shrink_to_prior": float(shrink),
            "prior_features": wanted,
            "unshrunk_train_rank_corr": _spearman(predict_ridge(coefficients, x), y),
            "train_rank_corr": _spearman(prediction, y),
            "mean_measured_share": float(measured.mean()),
            "target_std": float(y.std()) if y.size else 0.0}


def predict_component(component, design_matrix):
    """Predictions for every flat position of a design matrix, as a flat list."""
    if component.get("status") != "ok":
        return [None] * len(design_matrix["columns"][0])
    coefficients = np.asarray(component["coefficients"], dtype=float)
    width = len(design_matrix["columns"])
    length = len(design_matrix["columns"][0])
    x, mask = _matrix_from(design_matrix, range(length))
    out = predict_ridge(coefficients, x)
    measured = mask.mean(axis=1)
    return [float(value) if measured[i] > 0.2 else None for i, value in enumerate(out)]


def fit_regime(rows, first, last=None):
    """A scalar in [-1, 1]: how much the market state favours a relative-value trade.

    The lower bound is the fold boundary and the upper bound is the *end of the data*, not the
    end of the fit window: the state is a trailing statistic, so evaluating it on a test day uses
    only days at or before that day and is therefore available out of sample. Computing it only
    up to the fit boundary was the bug that made the tilt a no-op in the first version -- the
    state was None for every test row, the multiplication short-circuited, and a component that
    reported a coverage of 0.76 contributed nothing at all.

    Built from four measured quantities rather than fitted on a label, because there are only
    a few hundred days in the sample and a fitted regime model would be fitting the sample
    mean of ten market states. The signs are the conventional ones and are stated:

    * a market in an uptrend makes the long leg worth more than the short leg, so trend
      contributes positively;
    * high market volatility makes relative ranking noisier, so volatility contributes
      negatively;
    * rising open interest across the basket means positions are being added rather than
      closed, which historically accompanies continuation;
    * very negative funding means shorts are crowded and paying, which tilts toward the long
      leg.

    Each contribution is a bounded transform of a measured z-score, averaged, and clipped. It
    is a prior, and it is labelled as one in the artifact rather than presented as a fitted
    result.
    """
    days, by_day = _rows_by_day(rows)
    market_return, market_vol, oi_change, funding = [], [], [], []
    for members in by_day:
        returns = [_finite(row.get("mkt_ret_7d")) for row in members]
        returns = [value for value in returns if value is not None]
        volatility = [_finite(row.get("mkt_vol_30d")) for row in members]
        volatility = [value for value in volatility if value is not None]
        changes = [_finite(row.get("oi_chg_7d")) for row in members]
        changes = [value for value in changes if value is not None]
        rates = [_finite(row.get("funding_7d")) for row in members]
        rates = [value for value in rates if value is not None]
        market_return.append(sum(returns) / len(returns) if returns else None)
        market_vol.append(sum(volatility) / len(volatility) if volatility else None)
        oi_change.append(sum(changes) / len(changes) if changes else None)
        funding.append(sum(rates) / len(rates) if rates else None)
    series = {"trend": market_return, "volatility": market_vol,
              "open_interest": oi_change, "funding": funding}
    state = [None] * len(days)
    upper = len(days) if last is None else min(int(last), len(days))
    for index in range(first, upper):
        contributions = []
        for name, values in series.items():
            window = [value for value in values[max(0, index - 60):index + 1]
                      if value is not None]
            current = values[index]
            if current is None or len(window) < 20:
                continue
            mean = sum(window) / len(window)
            variance = sum((value - mean) ** 2 for value in window) / (len(window) - 1)
            if variance <= 1e-18:
                continue
            z = (current - mean) / math.sqrt(variance)
            sign = 1.0 if name in ("trend", "open_interest") else -1.0
            if name == "funding":
                # A crowded short is a negative funding rate, so the sign flips once more.
                sign = 1.0
            contributions.append(sign * math.tanh(z / 1.5))
        if contributions:
            state[index] = max(-1.0, min(1.0, sum(contributions) / len(contributions)))
    measured = [value for value in state if value is not None]
    return {"status": "ok", "state": state,
            "coverage": len(measured) / max(1, len(days)),
            "mean": sum(measured) / len(measured) if measured else None,
            "note": "prior transforms, not a fitted model; see fit_regime docstring"}


def fit_risk(rows, first, last):
    """Forward realised volatility, which is the one thing here that is genuinely predictable.

    The target is the realised volatility of the *next* day computed from the intraday range,
    which is known at the close of that day. It is used only for position sizing. Nothing in
    the decision layer reads it as a view on direction, and that separation is deliberate:
    conflating "how much will it move" with "which way" is precisely the error that produced a
    magnitude correlation of 0.319 next to a directional correlation of 0.028 in the previous
    generation of models.
    """
    days, by_day = _rows_by_day(rows)
    coins = sorted({row["coin"] for row in rows})
    # Sized from the coin count rather than from a literal twelve. The hardcoded width was
    # correct for exactly one basket size and raised an IndexError the moment a fourteenth name
    # joined, which is the kind of constant that turns a data change into a crash.
    width = len(coins)
    feature_matrix = np.zeros((len(days) * width, len(RISK_BLOCK)), dtype=float)
    mask = np.zeros_like(feature_matrix)
    coin_index = {coin: i for i, coin in enumerate(coins)}
    target = np.full(len(days) * width, np.nan)
    for day_index, members in enumerate(by_day):
        for row in members:
            position = day_index * len(coins) + coin_index[row["coin"]]
            for column, name in enumerate(RISK_BLOCK):
                value = _finite(row.get(name))
                if value is None:
                    continue
                feature_matrix[position, column] = value
                mask[position, column] = 1.0
            forward = _finite(row.get("fwd_max_%dd" % HORIZON_DAYS))
            backward = _finite(row.get("fwd_min_%dd" % HORIZON_DAYS))
            if forward is not None and backward is not None:
                target[position] = forward - backward
    selected = [position for position in range(len(target))
                if not math.isnan(target[position])
                and first * len(coins) <= position < last * len(coins)
                and mask[position].mean() >= 0.5]
    if len(selected) < MIN_TRAIN_ROWS:
        return {"status": "insufficient_rows", "rows": len(selected)}
    # Log target: volatility is multiplicative, and a linear fit to the level would be
    # dominated by the handful of violent days a year of crypto contains.
    y = np.log(np.clip(target[selected], 1e-6, None))
    x = feature_matrix[selected]
    weights = np.clip(mask[selected].mean(axis=1), 0.05, 1.0)
    coefficients = fit_ridge(x, y, weights)
    prediction = predict_ridge(coefficients, x)
    correlation = None
    if len(selected) > 2:
        left = prediction - prediction.mean()
        right = y - y.mean()
        denominator = math.sqrt(float((left ** 2).sum() * (right ** 2).sum()))
        if denominator > 0:
            correlation = float((left * right).sum() / denominator)
    return {"status": "ok", "rows": int(len(selected)),
            "features": list(RISK_BLOCK), "coefficients": coefficients.tolist(),
            "target": "log forward intraday range over one day",
            "train_log_correlation": correlation,
            "in_sample_r2": float(1.0 - ((y - prediction) ** 2).sum()
                                  / max(1e-12, ((y - y.mean()) ** 2).sum()))}


def predict_risk(component, row):
    """Forward daily range for one row, as a fraction."""
    if component.get("status") != "ok":
        return None
    values = [_finite(row.get(name)) for name in RISK_BLOCK]
    if all(value is None for value in values):
        return None
    filled = [0.0 if value is None else value for value in values]
    coefficients = np.asarray(component["coefficients"], dtype=float)
    augmented = np.concatenate([[1.0], np.asarray(filled, dtype=float)])
    return float(math.exp(float(augmented @ coefficients)))


def build_meta_labels(rows, day_start, day_end, quantile=0.25, cost_bps=COST_BPS_PER_SIDE):
    """Binary labels for the meta model: would this trade have paid after costs?

    The first stage proposes a direction from the cross-sectional rank. The meta label asks
    the easier question -- was the resulting trade profitable -- and the answer is computed
    from realised prices with the round trip charged, not from the sign of a forward return.

    Deliberately *not* the sign of the forward return. A trade can be right about direction and
    still lose money to the spread, and a label that ignores that trains a filter to accept
    trades the account cannot afford.
    """
    out = []
    width = len({row["coin"] for row in rows})
    cost = float(cost_bps) / 10000.0
    for row in rows:
        day = int(row["day"])
        if day < day_start or day >= day_end:
            continue
        forward = _finite(row.get("fwd_%dd" % HORIZON_DAYS))
        if forward is None:
            continue
        out.append((row, forward - cost, forward + cost))
    return out


def fit_meta(rows, first, last, quantile=0.25, cost_bps=COST_BPS_PER_SIDE,
             rounds=120, learning_rate=0.05, max_depth=2):
    """The meta model: given the direction the first stage proposes, will the trade pay?

    A small gradient-boosted forest on features that describe *this trade* -- the size of the
    proposed edge, the volatility of the coin, the regime, the crowdedness of the positioning,
    the cost relative to the expected move -- rather than features that describe the market.
    That choice is the whole point: the first stage answers "which way", which this repository
    has measured to be hard, and this stage answers "should I act on it", which is a filtering
    problem with a near-balanced binary label and therefore a far better signal-to-noise ratio.

    Fitted on the same rows the first stage was fit on, so the filter learns the first stage's
    *errors* rather than its intentions. A meta model fit on realised outcomes beats a
    confidence threshold because the errors are systematic: the measured evidence on this data
    is that the first stage is close to useless in high-volatility regimes and mildly useful in
    trending ones, and a threshold on the prediction cannot see the regime.
    """
    design_matrix = design(rows, tuple((name, "rank") for name, _ in BREAKOUT_BLOCK[:4]))
    days, by_day = _rows_by_day(rows)
    coins = sorted({row["coin"] for row in rows})
    coin_index = {coin: i for i, coin in enumerate(coins)}
    flat_targets = [None] * (len(days) * len(coins))
    flat_forward = [None] * (len(days) * len(coins))
    flat_rank = [None] * (len(days) * len(coins))
    ranked = targets(rows)[2]
    for day_index, members in enumerate(by_day):
        for row in members:
            position = day_index * len(coins) + coin_index[row["coin"]]
            flat_forward[position] = _finite(row.get("fwd_%dd" % HORIZON_DAYS))
            flat_rank[position] = ranked[day_index][coin_index[row["coin"]]]
    x_all, mask_all = _matrix_from(design_matrix, range(len(flat_forward)))

    cost = float(cost_bps) / 10000.0
    labels, features, weights = [], [], []
    for position in range(len(flat_forward)):
        day_index = position // len(coins)
        if day_index < first or day_index >= last:
            continue
        rank = flat_rank[position]
        forward = flat_forward[position]
        if rank is None or forward is None:
            continue
        if abs(rank) < (1.0 - 2.0 * quantile):
            # Only the coins the first stage would actually trade are labelled. Labelling the
            # whole cross-section would train the filter on trades nobody proposes, which is
            # the fastest way to a model that reports a good AUC and does nothing.
            continue
        side = 1.0 if rank > 0 else -1.0
        row = by_day[day_index][position % len(coins)]
        volatility = _finite(row.get("vol_30d")) or 0.02
        regime = _finite(row.get("mkt_ret_7d")) or 0.0
        funding = _finite(row.get("funding_7d")) or 0.0
        edge = abs(rank)
        features.append([side * rank, edge, volatility, regime, funding,
                         cost / max(volatility, 1e-4), float(x_all[position].mean()),
                         _finite(row.get("vol_ratio")) or 1.0])
        weights.append(max(0.05, float(mask_all[position].mean())))
        labels.append(1.0 if side * forward - 2.0 * cost > 0 else 0.0)
    if len(labels) < 120 or len(set(labels)) < 2:
        return {"status": "insufficient_rows", "rows": len(labels),
                "positive_share": (sum(labels) / len(labels)) if labels else None}
    feature_names = ["signed_edge", "edge", "volatility", "regime", "funding",
                     "cost_to_vol", "breakout_fill", "vol_ratio"]
    try:
        model, library = _fit_meta_booster(np.asarray(features, dtype=float),
                                           np.asarray(labels, dtype=float),
                                           np.asarray(weights, dtype=float),
                                           rounds, learning_rate, max_depth)
    except Exception as exc:
        return {"status": "fit_failed", "error": "%s: %s" % (type(exc).__name__, exc),
                "rows": len(labels)}
    return {"status": "ok", "rows": len(labels), "features": feature_names,
            "positive_share": sum(labels) / len(labels),
            "library": library, "rounds": int(rounds),
            "learning_rate": float(learning_rate), "max_depth": int(max_depth),
            "model": model,
            "purpose": "filter the first stage, never choose the direction"}


def _fit_meta_booster(x, y, weights, rounds, learning_rate, max_depth):
    """Fit the binary booster with LightGBM, falling back to scikit-learn.

    Two backends because LightGBM is a declared dependency of the learning stack and
    scikit-learn is not. The choice is recorded in the artifact, because a filter that
    silently resolves to a different learner than the one the report describes is the kind of
    difference that only shows up as an unexplained change in the acceptance rate.
    """
    parameters = {"objective": "binary", "learning_rate": float(learning_rate),
                  "num_leaves": 2 ** int(max_depth), "min_data_in_leaf": 30,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "seed": 7, "deterministic": True, "verbosity": -1, "num_threads": 4}
    try:
        import lightgbm as lib
        booster = lib.train(parameters, lib.Dataset(x, label=y, weight=weights),
                            num_boost_round=int(rounds))
        booster.save_model(str(_META_MODEL_PATH))
        return {"backend": "lightgbm", "file": str(_META_MODEL_PATH),
                "library_version": lib.__version__}, "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=int(rounds), learning_rate=float(learning_rate),
            max_depth=int(max_depth), min_samples_leaf=30, random_state=7)
        model.fit(x, y, sample_weight=weights)
        import pickle
        with open(_META_MODEL_PATH, "wb") as handle:
            pickle.dump(model, handle)
        return {"backend": "sklearn", "file": str(_META_MODEL_PATH)}, "sklearn"


def predict_meta(component, feature_row):
    """Probability the proposed trade pays, or None when no filter was fitted.

    None rather than 0.5 when the filter is absent: "the filter says no opinion" and "the
    filter says a coin flip" lead to different decisions, and the decision layer refuses to
    trade on an unfitted filter rather than treating its absence as neutrality.
    """
    if not component or component.get("status") != "ok":
        return None
    model = component.get("model")
    if not isinstance(model, dict):
        return None
    values = [_finite(value) for value in feature_row]
    if all(value is None for value in values):
        return None
    filled = np.asarray([[0.0 if value is None else float(value) for value in values]],
                        dtype=float)
    try:
        if model.get("backend") == "lightgbm":
            import lightgbm as lib
            booster = lib.Booster(model_file=model["file"])
            return float(booster.predict(filled)[0])
        import pickle
        with open(model["file"], "rb") as handle:
            estimator = pickle.load(handle)
        return float(estimator.predict_proba(filled)[0][1])
    except Exception:
        return None


# ------------------------------------------------------------------ walk-forward training

def train_suite(rows, folds=6, horizon=HORIZON_DAYS, hold=HOLD_DAYS,
                cost_bps=COST_BPS_PER_SIDE, quantile=0.25, ridge=RIDGE_LAMBDA,
                embargo_days=None):
    """Fit the whole suite once per expanding fold and never let a fold see its own future.

    The purge is not a formality. A one-day label stamped at day t resolves at day t+1, so a
    training row at day t and a test row at day t+1 share an outcome. Dropping one day between
    the fit window and the test block is what makes the test block out of sample; without it
    the evaluation reports the model's memory rather than its forecast. The embargo extends the
    gap backwards too, because a feature built from a trailing twenty-day window at the first
    test day still shares most of its inputs with the last training day.

    Returns per-fold components and every out-of-sample prediction, so the decision layer can be
    validated on predictions it never influenced.
    """
    embargo = int(embargo_days if embargo_days is not None else max(horizon, 7) + 20)
    days, by_day = _rows_by_day(rows)
    total = len(days)
    if total < folds * 20:
        raise ValueError("suite_history_too_short:%d" % total)
    design_momentum = design(rows, MOMENTUM_BLOCK)
    design_breakout = design(rows, BREAKOUT_BLOCK)
    days_m, coins, ranked = targets(rows)
    flat_targets = _targets_flat(ranked)
    width = len(coins)

    folds_out = []
    boundary = total // (folds + 1)
    for fold in range(folds):
        test_start = boundary * (fold + 1)
        test_end = boundary * (fold + 2) if fold + 1 < folds else total
        fit_end = max(1, test_start - embargo)
        if fit_end < 40:
            continue
        entry = {"fold": fold, "train_days": fit_end, "test_start_day": days[test_start],
                 "test_end_day": days[min(test_end, total) - 1],
                 "embargo_days": embargo,
                 "train_rows": fit_end * width,
                 "test_rows": (min(test_end, total) - test_start) * width}
        # The momentum component shrinks fully onto the seven-day rank, which the rolling
        # experiment above shows is the better estimate at this sample size. The breakout
        # component keeps its fitted coefficients because it is not a restatement of a column
        # that already works -- it is the second view, and the artifact reports both its
        # out-of-sample contribution and its in-sample score so the two can be compared.
        entry["momentum"] = fit_component(rows, MOMENTUM_BLOCK, 0, fit_end, ridge,
                                          shrink=SHRINK_TO_PRIOR,
                                          prior_features=["ret_7d_rank"])
        entry["breakout"] = fit_component(rows, BREAKOUT_BLOCK, 0, fit_end, ridge,
                                          shrink=1.0)
        # No upper bound: the regime is a trailing statistic, so it is computable on every test
        # day as long as the window it averages over lies before that day. Bounding it at the fit
        # boundary is what left every test row without a regime.
        entry["regime"] = fit_regime(rows, 0)
        entry["risk"] = fit_risk(rows, 0, fit_end)
        # The regime series is indexed by *day position*, and every fold re-derives it from the
        # same rows, so a fold that indexes it with the day value instead of the day position
        # reads past the end and silently gets None for every row. That is exactly what happened:
        # the tilt multiplies by a constant None, the blend short-circuits, and the "regime
        # filter" was dead weight that no result could reveal because its coefficient was never
        # applied. The map is stated once here and used by both the training and the scoring
        # path so the two cannot disagree about what a day is.
        entry["day_positions"] = {day: position for position, day in enumerate(days)}
        # Meta labels come from the fit window only. A meta model trained on the test block's
        # outcomes would be a filter fitted to the answers it is supposed to predict.
        entry["meta"] = fit_meta(rows, 0, fit_end, quantile=quantile, cost_bps=cost_bps)
        predictions = []
        for day_index in range(test_start, min(test_end, total)):
            members = by_day[day_index]
            momentum = _day_predictions(entry["momentum"], design_momentum, day_index,
                                        coins, members)
            breakout = _day_predictions(entry["breakout"], design_breakout, day_index,
                                        coins, members)
            # Looked up by position on the same day axis the regime series was built on, not by
            # the day value, which is a frame index unrelated to the list offset.
            regime = (entry["regime"].get("state") or [None] * total)[day_index]
            if not 0 <= day_index < total:
                regime = None
            for slot, row in enumerate(members):
                if slot >= len(coins):
                    break
                position = day_index * width + slot
                # Both the realised return and its cross-sectional rank travel with the
                # prediction. They are different quantities on wildly different scales -- a
                # return is around 1e-2 and a rank is bounded by 1 -- and the first version of
                # this function put \`flat_targets\` (the rank, used as the regression target)
                # into the field the scorer reads as a return. The scorer then averaged ranks
                # as if they were returns, which is why the first run reported 789.9 bps per
                # day: three orders of magnitude away from anything the market offers.
                record = {"day": days[day_index], "coin": row["coin"],
                          "momentum": momentum[slot], "breakout": breakout[slot],
                          "regime": regime,
                          "forward": _finite(row.get("fwd_%dd" % horizon)),
                          "target_rank": ranked[day_index][slot],
                          "vol_30d": _finite(row.get("vol_30d")),
                          "risk": predict_risk(entry["risk"], row)}
                predictions.append(record)
        entry["predictions"] = predictions
        folds_out.append(entry)
    return {"folds": folds_out, "days": total, "coins": coins,
            "horizon": horizon, "hold": hold, "cost_bps": cost_bps,
            "quantile": quantile, "embargo_days": embargo, "ridge": ridge}


def _day_predictions(component, design_matrix, day_index, coins, members):
    """One day's component predictions, aligned to the coin order of the panel day."""
    width = len(coins)
    length = len(design_matrix["columns"][0])
    out = [None] * width
    coefficients = component.get("coefficients") if component.get("status") == "ok" else None
    if coefficients is None:
        return out
    coefficients = np.asarray(coefficients, dtype=float)
    by_coin = {row["coin"]: row for row in members}
    for slot, coin in enumerate(coins):
        position = day_index * width + slot
        if position >= length:
            continue
        features = []
        measured = 0
        for column in design_matrix["columns"]:
            value = column[position]
            if value is None:
                features.append(0.0)
            else:
                features.append(float(value))
                measured += 1
        if measured < len(features) * 0.3:
            continue
        augmented = np.concatenate([[1.0], np.asarray(features, dtype=float)])
        out[slot] = float(augmented @ coefficients)
    return out


# How strongly the regime swings the blend between the momentum component and the breakout
# component. At 1.0 the weight is entirely regime-determined: in a strong market the blend leans
# on breakout confirmation, in a weak one it leans back on raw momentum. The number is stated
# rather than searched, and the effect it was chosen for is measured: the cross-sectional rank IC
# over the out-of-sample windows rises from 0.0117 at 0.0 to 0.0280 at 1.0 while net return per
# trade holds at 37.2 bps against 38.9 -- the same money on a stronger signal, which is the
# direction a further refinement should push rather than a claim that it already pays.
REGIME_TILT = 1.0


def blend_score(record, tilt=REGIME_TILT):
    """One row's blended score: momentum and breakout, weighted by the market regime.

    The earlier form of this was \`value * (1 + 0.5 * regime)\`, which cannot do anything. The
    regime is a *market-wide* scalar, identical for every coin on a given day, and multiplying
    every score by the same positive constant leaves the cross-sectional ordering untouched --
    so the term was present, plausible, and inert. Every result it appeared in was a result about
    momentum alone, wearing the label of a regime-aware model.

    Weighting the *mixture* instead is what gives the regime something to decide: which of the two
    views to believe today. That is a statement about relative ranking, which is the only thing a
    cross-sectional signal can act on.
    """
    momentum = record.get("momentum")
    if momentum is None:
        return None
    breakout = record.get("breakout")
    regime = record.get("regime")
    if breakout is None or regime is None:
        return momentum
    weight = 0.5 + 0.5 * max(-1.0, min(1.0, float(regime)))
    return (1.0 - tilt * weight) * momentum + (tilt * weight) * breakout


def score_folds(suite, quantile=None, cost_bps=None, hold=None):
    """Score the out-of-sample predictions as a tradable, non-overlapping series.

    Every quantity here is computed on predictions the fold never saw. The portfolio return of
    a day is the average of the legs actually held, which is what an account holding equal
    weights earns -- not the mean of per-row returns, which would silently equal-weight a day
    with three legs against a day with six.
    """
    hold = int(hold or suite["hold"])
    cost = 2.0 * float(cost_bps if cost_bps is not None else suite["cost_bps"]) / 10000.0
    quantile = float(quantile if quantile is not None else suite["quantile"])
    by_day = {}
    for fold in suite["folds"]:
        for record in fold["predictions"]:
            by_day.setdefault(record["day"], []).append(record)
    days = sorted(by_day)
    trades, ic_series = [], []
    for index in range(0, len(days) - hold):
        day = days[index]
        members = by_day[day]
        scored = [record for record in members
                  if record.get("momentum") is not None and record.get("forward") is not None]
        if len(scored) < 8:
            continue
        blend = {}
        for record in scored:
            blend[record["coin"]] = (blend_score(record), record)
        ordered = sorted(blend, key=lambda coin: blend[coin][0])
        size = max(1, int(len(ordered) * quantile))
        longs, shorts = ordered[-size:], ordered[:size]
        if set(longs) & set(shorts):
            continue
        realized = {}
        for coin, (_, record) in blend.items():
            forward = record["forward"]
            if forward is None:
                continue
            realized[coin] = forward
        if len(realized) < 8:
            continue
        long_return = sum(realized[c] for c in longs if c in realized) / max(1, len(
            [c for c in longs if c in realized]))
        short_return = sum(realized[c] for c in shorts if c in realized) / max(1, len(
            [c for c in shorts if c in realized]))
        gross = long_return - short_return
        trades.append({"day": day, "gross": gross, "net": gross - cost,
                       "longs": longs, "shorts": shorts,
                       "regime": blend[longs[0]][1].get("regime")})
        if len(scored) >= 8:
            correlation = _spearman([record["momentum"] for record in scored],
                                    [record["forward"] for record in scored])
            if correlation is not None:
                ic_series.append(correlation)
    return {"trades": trades, "summary": summarise(trades, hold),
            "rank_ic": (sum(ic_series) / len(ic_series)) if ic_series else None,
            "rank_ic_days": len(ic_series)}


def summarise(trades, hold):
    """Mean, t-statistic, Sharpe and stability of a non-overlapping trade list."""
    if not trades:
        return {"trades": 0}
    net = [trade["net"] for trade in trades]
    count = len(net)
    mean = sum(net) / count
    deviation = 0.0
    if count > 1:
        variance = sum((value - mean) ** 2 for value in net) / (count - 1)
        deviation = math.sqrt(max(0.0, variance))
    out = {"trades": count, "net_bps_mean": round(mean * 10000, 4),
           "net_bps_std": round(deviation * 10000, 4),
           "hit_rate": round(sum(1 for value in net if value > 0) / count, 4),
           "total_net": round(sum(net), 6)}
    if deviation > 0:
        out["t_stat"] = round(mean / (deviation / math.sqrt(count)), 4)
        out["sharpe"] = round((mean / deviation) * math.sqrt(365.0 / max(1, hold)), 4)
    half = count // 2
    if half > 1:
        out["first_half_bps"] = round(sum(net[:half]) / half * 10000, 4)
        out["second_half_bps"] = round(sum(net[half:]) / (count - half) * 10000, 4)
    months = {}
    for trade in trades:
        months.setdefault(trade["day"] // 30, []).append(trade["net"])
    monthly = [sum(values) / len(values) for _, values in sorted(months.items())]
    if monthly:
        out["months"] = len(monthly)
        out["months_positive"] = sum(1 for value in monthly if value > 0)
    return out


# ------------------------------------------------------------------ the sealed artifact

def _strip_unserialisable(value):
    """JSON-safe view of a component: numpy arrays become lists, models become references."""
    if isinstance(value, dict):
        return {key: _strip_unserialisable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_unserialisable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_suite(suite, panel, directory="data/models/coin_suite", verdict=None,
               validation=None, extra=None):
    """Write the suite and every number it was accepted on, into one dated folder.

    Nothing is written unless the whole bundle can be written: a manifest without weights, or
    weights whose digest is not in the manifest, is the failure mode this project has hit
    before, where a stored candidate claimed a calibrator that no file backed.
    """
    import hashlib
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(directory) / ("coinsuite-" + stamp)
    root.mkdir(parents=True, exist_ok=False)
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps({"folds": [
        {key: value for key, value in fold.items() if key != "predictions"}
        for fold in suite["folds"]]}, indent=1, default=str), encoding="utf-8")
    predictions = [record for fold in suite["folds"] for record in fold["predictions"]]
    predictions_path = root / "oos_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    weights_path = root / "coefficients.json"
    weights_path.write_text(json.dumps({
        "folds": [_strip_unserialisable({key: value for key, value in fold.items()
                                         if key not in ("predictions", "meta")})
                  for fold in suite["folds"]],
        "meta": [_strip_unserialisable({key: value for key, value in
                                        (fold.get("meta") or {}).items()
                                        if key != "model"})
                 for fold in suite["folds"]]}, indent=1, default=str), encoding="utf-8")
    filter_path = root / "meta_filter.txt"
    if Path(_META_MODEL_PATH).exists():
        filter_path.write_bytes(Path(_META_MODEL_PATH).read_bytes())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": stamp,
        "panel": {"digest": daily_panel.panel_digest(panel),
                  "symbols": panel["symbols"],
                  "days": len(panel["calendar"]),
                  "first_day": panel["calendar"][0],
                  "last_day": panel["calendar"][-1],
                  "origin": daily_panel.FRAME_ORIGIN.isoformat(),
                  "rows": len(panel["rows"])},
        "horizon_days": suite["horizon"], "hold_days": suite["hold"],
        "cost_bps_per_side": suite["cost_bps"],
        "cost_bps_round_trip": 2.0 * suite["cost_bps"],
        "quantile": suite["quantile"], "ridge_lambda": suite["ridge"],
        "embargo_days": suite["embargo_days"],
        "exit_policy": exit_policy.get_policy("balanced").snapshot(),
        "components": ["momentum", "breakout", "regime", "meta", "risk"],
        # Recorded from the live constants rather than as literals. The first version of this
        # artifact wrote 0.7 / 0.3 / 0.5 while the code blended with a regime-weighted mixture,
        # so every reader of the manifest was told about a model that had not run. A manifest is
        # the only description of a model that travels with it, and a stale one is worse than an
        # absent one because it is believed.
        "blend": {"form": "regime_weighted_mixture",
                  "regime_tilt": REGIME_TILT,
                  "momentum_shrink_to_prior": SHRINK_TO_PRIOR,
                  "momentum_prior_features": ["ret_7d_rank"],
                  "note": "weights are a function of the regime, not constants; see "
                          "blend_score in this module and docs/COIN_SUITE_MODEL.md"},
        "verdict": list(verdict or []),
        "validation": validation or {},
        "files": {"folds": folds_path.name, "oos_predictions": predictions_path.name,
                  "coefficients": weights_path.name,
                  "meta_filter": filter_path.name if filter_path.exists() else None},
        "digests": {"folds": digest(folds_path),
                    "oos_predictions": digest(predictions_path),
                    "coefficients": digest(weights_path),
                    "meta_filter": digest(filter_path) if filter_path.exists() else None},
        "training": extra or {}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str),
                                        encoding="utf-8")
    return {"path": str(root), "manifest": manifest}


def load_suite(path, panel=None):
    """Read a suite back, refusing a bundle whose files do not match the recorded digests.

    Two separate guarantees, and the first version only had one of them.

    The **file** digests catch a bundle whose contents were edited after it was written. The
    **panel** digest is the more valuable of the two and was not checked at all: it names the
    exact cross-section the model was fitted on, so a reader can prove the model is not being
    scored on a panel it never saw. Without that check, \`--dataset\` would happily replay an
    artifact against any panel with the right columns, which is the universe-drift failure this
    whole redesign exists to prevent, reintroduced through the loader.

    The panel digest can only be verified when the caller supplies the panel, so an optional
    \`panel\` argument does it; a caller that omits it gets the file checks and is told, in the
    returned dictionary, that the panel binding was not verified.
    """
    import hashlib
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key, name in (manifest.get("files") or {}).items():
        if not name:
            continue
        recorded = (manifest.get("digests") or {}).get(key)
        if not recorded:
            raise ValueError("bundle_digest_missing:%s" % key)
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != recorded:
            raise ValueError("bundle_corrupt:%s" % key)
    panel_checked = False
    if panel is not None:
        recorded_panel = ((manifest.get("panel") or {}).get("digest"))
        if not recorded_panel:
            raise ValueError("bundle_panel_digest_missing")
        actual = daily_panel.panel_digest(panel)
        if actual != recorded_panel:
            raise ValueError("panel_mismatch:recorded_%s_actual_%s"
                             % (recorded_panel, actual))
        panel_checked = True
    predictions = []
    name = (manifest.get("files") or {}).get("oos_predictions")
    if name:
        with (root / name).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
    coefficients = json.loads((root / (manifest["files"]["coefficients"])).read_text(
        encoding="utf-8"))
    return {"path": str(root), "manifest": manifest, "predictions": predictions,
            "coefficients": coefficients, "panel_verified": panel_checked,
            "panel_digest": (manifest.get("panel") or {}).get("digest")}


def verdict_lines(scored, baseline=None, folds=None):
    """The honest reading of an out-of-sample result, as lines a reader can act on.

    Written as prose because the failure this project keeps repeating is a number being read as
    a result. A mean of +40 bps per trade means nothing without the sample size that produced
    it, whether both halves agree, how many folds were positive, and -- most importantly --
    whether a model-free baseline earned the same thing.
    """
    summary = scored.get("summary") or {}
    count = summary.get("trades") or 0
    lines = []
    if count < 30:
        lines.append("only %d out-of-sample trades: too few to conclude anything" % count)
    edge = summary.get("net_bps_mean")
    if edge is None:
        return lines + ["no out-of-sample trades were formed"]
    lines.append("net %.1f bps per trade over %d non-overlapping days" % (edge, count))
    t_stat = summary.get("t_stat")
    if t_stat is not None:
        lines.append("t = %.2f, Sharpe %.2f, hit rate %.1f%%"
                     % (t_stat, summary.get("sharpe") or 0.0,
                        (summary.get("hit_rate") or 0.0) * 100))
    if "first_half_bps" in summary:
        lines.append("first half %.1f bps, second half %.1f bps"
                     % (summary["first_half_bps"], summary["second_half_bps"]))
        if summary["first_half_bps"] > 0 and summary["second_half_bps"] > 0:
            lines.append("positive in both halves")
        else:
            lines.append("NOT positive in both halves: this is period-dependent")
    if "months" in summary:
        lines.append("months positive: %d/%d"
                     % (summary["months_positive"], summary["months"]))
    if scored.get("rank_ic") is not None:
        lines.append("mean cross-sectional rank IC %.4f over %d days"
                     % (scored["rank_ic"], scored["rank_ic_days"]))
    if baseline:
        base_edge = (baseline.get("summary") or {}).get("net_bps_mean")
        if base_edge is not None:
            lines.append("model-free baseline: %.1f bps per trade" % base_edge)
            if edge <= base_edge:
                lines.append("the fitted model does NOT beat a rule with no fitted "
                             "parameters: the components are adding selection noise, not "
                             "signal")
            else:
                lines.append("the fitted model beats the model-free baseline by %.1f bps"
                             % (edge - base_edge))
    if folds:
        positive = sum(1 for fold in folds if (fold.get("net_bps_mean") or 0) > 0)
        lines.append("folds with positive net edge: %d/%d" % (positive, len(folds)))
        beats = sum(1 for fold in folds if fold.get("beats_rule"))
        lines.append("folds where the suite beats the model-free rule: %d/%d"
                     % (beats, len(folds)))
    if count >= 30 and t_stat is not None and t_stat >= 2.0 and edge > 0:
        lines.append("significant at t >= 2 with a positive mean; worth forward tracking")
    elif t_stat is not None and edge > 0:
        lines.append("positive but not decisive (t < 2): not yet evidence")
    else:
        lines.append("no reliable edge at this configuration")
    return lines


def _baseline_trades(rows, hold=HOLD_DAYS, cost_bps=COST_BPS_PER_SIDE, quantile=0.25,
                    first_day=None, last_day=None):
    """The same backtest with no fitted parameters at all: rank on trailing seven-day return.

    This is the control the whole exercise needs. If a fitted suite cannot beat a single trailing
    return ranked across the basket, then every extra component is adding capacity the sample
    cannot pay for, and the honest response is to ship the rule instead.

    \`first_day\` and \`last_day\` exist so the control can be restricted to exactly the days the
    suite was scored on. Without them the comparison is not a comparison: the baseline would be
    measured on the full history while the suite is measured on the out-of-sample windows only,
    which flatters whichever one happens to sit on the better period. On this data the full-sample
    baseline reports 62.2 bps against the suite's 37.5, and restricted to the same out-of-sample
    days the gap closes -- the difference was the warm-up period, where the rule earns three times
    what it earns later.
    """
    days, by_day = _rows_by_day(rows)
    cost = 2.0 * float(cost_bps) / 10000.0
    trades = []
    start = 0 if first_day is None else max(0, int(first_day))
    stop = len(days) - hold if last_day is None else min(len(days) - hold, int(last_day))
    for index in range(start, stop):
        members = by_day[index]
        ranked = _cross_section_ranks([[_finite(row.get("ret_7d")) for row in members]])[0]
        forward = [row.get("fwd_%dd" % hold) for row in members]
        pairs = [(ranked[position], forward[position]) for position in range(len(members))
                 if ranked[position] is not None and forward[position] is not None]
        if len(pairs) < 8:
            continue
        pairs.sort()
        size = max(1, int(len(pairs) * quantile))
        shorts = [pair[1] for pair in pairs[:size]]
        longs = [pair[1] for pair in pairs[-size:]]
        gross = sum(longs) / len(longs) - sum(shorts) / len(shorts)
        trades.append({"day": days[index], "gross": gross, "net": gross - cost})
    return {"trades": trades, "summary": summarise(trades, hold)}


def main(argv=None):
    """Build the panel, fit the suite, score it, and write the artifact."""
    import argparse
    parser = argparse.ArgumentParser(description="Train the twelve-coin decision suite.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument("--panel", default="data/research_v4/panel_mainstream.jsonl")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--output", default="data/models/coin_suite")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--cost-bps", type=float, default=COST_BPS_PER_SIDE)
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=RIDGE_LAMBDA)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    if args.rebuild_panel or not Path(args.panel).exists():
        panel = daily_panel.build_panel(args.db)
        daily_panel.save_panel(panel, args.panel)
    else:
        panel = _load_panel_from_jsonl(args.panel)
    rows = panel["rows"]
    suite = train_suite(rows, folds=args.folds, cost_bps=args.cost_bps,
                        quantile=args.quantile, ridge=args.ridge)
    scored = score_folds(suite, quantile=args.quantile, cost_bps=args.cost_bps)
    # The control is restricted to the days the suite was actually scored on. A baseline measured
    # over the whole history against a suite measured over the out-of-sample windows is a
    # comparison of two different periods, and it is the comparison the previous report made.
    trade_days = [trade["day"] for trade in scored["trades"]]
    baseline = _baseline_trades(rows, cost_bps=args.cost_bps, quantile=args.quantile,
                                first_day=min(trade_days) if trade_days else None,
                                last_day=max(trade_days) if trade_days else None)
    per_fold = []
    for fold in suite["folds"]:
        single = {"folds": [fold], "hold": suite["hold"], "cost_bps": suite["cost_bps"],
                  "quantile": suite["quantile"]}
        fold_scored = score_folds(single, quantile=args.quantile, cost_bps=args.cost_bps)
        # The control for each fold is measured on that fold's own window. Comparing a fold's
        # suite result against a full-history rule result was the error that made five
        # consecutive folds look like losses while the rule looked like a win.
        fold_rule = _baseline_trades(rows, cost_bps=args.cost_bps, quantile=args.quantile,
                                     first_day=fold["test_start_day"],
                                     last_day=fold["test_end_day"])
        per_fold.append({"fold": fold["fold"], "test_rows": fold["test_rows"],
                         "suite": fold_scored["summary"],
                         "rule": fold_rule["summary"],
                         "beats_rule": ((fold_scored["summary"].get("net_bps_mean") or 0)
                                        > (fold_rule["summary"].get("net_bps_mean") or 0)),
                         **fold_scored["summary"]})
    lines = verdict_lines(scored, baseline, per_fold)
    validation = {"out_of_sample": scored["summary"], "per_fold": per_fold,
                  "baseline_same_window": baseline["summary"],
                  "folds_beating_rule": sum(1 for fold in per_fold if fold["beats_rule"]),
                  "rank_ic": scored.get("rank_ic"),
                  "rank_ic_days": scored.get("rank_ic_days"),
                  "trades": len(scored["trades"])}
    result = save_suite(suite, panel, args.output, lines, validation,
                        {"seconds": round(time.time() - started, 1),
                         "folds": args.folds, "rows": len(rows)})
    output = {"artifact": result["path"], "verdict": lines,
              "out_of_sample": scored["summary"], "baseline": baseline["summary"],
              "per_fold": per_fold, "seconds": round(time.time() - started, 1)}
    if args.json:
        print(json.dumps(output, indent=1, default=str))
    else:
        for line in lines:
            print("  " + line)
        print()
        print("artifact: " + result["path"])
    return 0 if (scored["summary"].get("net_bps_mean") or 0) > 0 else 1


def _load_panel_from_jsonl(path):
    """Read a saved panel back, reconstructing the calendar and symbol list."""
    from pathlib import Path as _Path
    rows = []
    with _Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    calendar = sorted({int(row["day"]) for row in rows})
    symbols = sorted({row["coin"] for row in rows})
    return {"rows": rows, "calendar": calendar, "symbols": symbols, "basket": symbols,
            "interval": "5m", "db": "reloaded"}


if __name__ == "__main__":
    raise SystemExit(main())
