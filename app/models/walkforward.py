"""Walk-forward evaluation for the tabular models.

A single chronological split reports one number from one period, which cannot separate
edge from luck. This module retrains on an expanding window and evaluates on each
successive out-of-sample block, then reports the distribution across folds.

Two details matter for honesty:

* a purge gap of `horizon` bars is dropped between train and test, because a label made
  at the end of the training window resolves inside the test window;
* the same gap is dropped between the fit set and the early-stopping set. Without it the
  iteration count is chosen on rows whose labels overlap the rows the model just fitted,
  so early stopping optimises a number that is partly memorised. Measured on a real
  dataset the boundary was adjacent (0 bars) while labels reached 12 bars forward;
* overlapping labels are not independent, so the confidence interval uses a moving-block
  bootstrap rather than an i.i.d. one, which would understate the uncertainty.
"""
import bisect

import numpy as np

from ..features.feature_spec import FEATURES
from .fit_model import fit
from .tabular_model import evaluate


def walk_forward(rows, train_size=40, test_size=10, step=10):
    """Rolling-window walk-forward for the linear baseline.

    Kept alongside walk_forward_tabular: it is the cheap sanity check used by the
    research report, while the tabular version is what the candidates are judged on.
    """
    if min(train_size, test_size, step) <= 0:
        raise ValueError('invalid_window_sizes')
    rows = sorted(rows, key=lambda row: (row['timestamp'], row['symbol']))
    times = sorted({row['timestamp'] for row in rows})
    reports = []
    start = 0
    while start + train_size + test_size <= len(times):
        train_times = set(times[start:start + train_size])
        test_times = set(times[start + train_size:start + train_size + test_size])
        train = [row for row in rows if row['timestamp'] in train_times]
        test = [row for row in rows if row['timestamp'] in test_times]
        validation_size = max(5, train_size // 5)
        total = train_size + test_size
        result = fit(train + test, (train_size - validation_size) / total,
                     validation_size / total, output=None)
        reports.append({'start': start, 'train_rows': len(train), 'test_rows': len(test),
                        'model': result})
        start += step
    return {'windows': len(reports), 'reports': reports}


def feature_matrix(rows, features=FEATURES):
    """Build the design matrix once; slicing it per fold beats rebuilding it."""
    values = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non_finite_features")
    return values


def time_blocks(rows, folds):
    """Split the distinct timestamps into `folds` contiguous blocks.

    Splitting on time rather than rows keeps every symbol in the same fold, so the
    evaluation never trains on the same period it is tested on.
    """
    if folds < 1:
        raise ValueError("invalid_fold_count")
    times = sorted({row["timestamp"] for row in rows})
    if len(times) < folds * 2:
        raise ValueError("too_few_time_groups")
    size = len(times) // folds
    return [times[i * size:(i + 1) * size] for i in range(folds)]


def _train(backend, parameters, x, y, vx, vy, rounds):
    if backend == "lightgbm":
        import lightgbm as lib
        return lib.train(parameters, lib.Dataset(x, label=y), num_boost_round=rounds,
                         valid_sets=[lib.Dataset(vx, label=vy)],
                         callbacks=[lib.early_stopping(20, verbose=False)])
    import catboost as lib
    model = lib.CatBoostRegressor(**parameters)
    model.fit(x, y, eval_set=(vx, vy), early_stopping_rounds=20)
    return model


def parameters_for(backend):
    if backend == "lightgbm":
        return {"objective": "regression", "learning_rate": .03, "num_leaves": 15,
                "min_data_in_leaf": 10, "seed": 42, "num_threads": 2,
                "deterministic": True, "force_col_wise": True, "verbosity": -1}
    return {"iterations": 200, "depth": 5, "learning_rate": .03,
            "loss_function": "RMSE", "random_seed": 42, "thread_count": 2,
            "verbose": False, "allow_writing_files": False}


def _label_span_bars(rows, times, step=None):
    """The longest label, in bars, or None when the dataset has no label end times.

    The median is the wrong statistic here: one label that reaches past the gap is enough to
    leak, so the gap has to be built from the maximum. Returns None rather than a guess when
    the intervals are absent, because inventing a span would silently under-purge.
    """
    if len(times) < 2:
        return None
    if step is None:
        step = times[1] - times[0]
    if step <= 0:
        return None
    longest = 0.0
    seen = False
    for row in rows:
        end = row.get("label_end_time")
        if end is None:
            continue
        seen = True
        span = (float(end) - float(row["timestamp"])) / step
        if span > longest:
            longest = span
    if not seen:
        return None
    # Rounded up: a label spanning 12.4 bars still reaches into the 13th.
    return int(np.ceil(longest))


def _purged_split(train_idx, rows, by_time, times, purge_bars, validation_fraction=0.1):
    """Split a training window into (fit, validation) with a purge gap between them.

    The gap is removed from the END of the fit set rather than the start of the validation
    set, so the validation window stays as close to the test block as it was before and the
    early-stopping signal remains representative of what the model will face. Rows inside
    the gap are dropped from training entirely for that fold; on an expanding window that
    costs a few hundred rows out of hundreds of thousands.

    Falls back to an adjacent split when there is not enough room to purge, rather than
    failing: a fold with too little history to hold back a validation set is a fold whose
    numbers are weak anyway, and refusing to run would hide that.
    """
    guard = max(1, int(len(train_idx) * validation_fraction))
    valid = train_idx[-guard:]
    valid_start = min(rows[i]["timestamp"] for i in valid)
    # The same arithmetic the train/test split uses: count the timestamps strictly before
    # the validation window, step back `purge_bars` of them, and cut there. Working in whole
    # timestamps rather than milliseconds matters because a bar holds one row per symbol --
    # cutting by row index, which is what this used to do, split a single bar's symbols
    # across fit and validation, so the model trained on half a bar and validated on the
    # other half.
    cutoff = bisect.bisect_left(times, valid_start)
    cut = cutoff - purge_bars
    if cut <= 0:
        # Too little history to hold back both a validation set and a gap. Fall back to an
        # adjacent split rather than refusing, so the fold still reports something.
        return train_idx[:-guard] or train_idx, valid
    fit = [i for stamp in times[:cut] for i in by_time[stamp]]
    return (fit or train_idx[:-guard]), valid


def walk_forward_tabular(rows, backend="lightgbm", folds=12, cost_bps=12, rounds=200,
                         horizon=12, purge_bars=None, features=FEATURES, return_oos=False):
    """Expanding-window walk-forward; returns per-fold and aggregate out-of-sample metrics.

    With return_oos the individual out-of-sample predictions come back too. They are what a
    portfolio replay needs: this function scores each prediction on its own, which cannot
    see capital being shared, positions being concurrent, or a drawdown accumulating.
    """
    if backend not in ("lightgbm", "catboost"):
        raise ValueError("unsupported_backend")
    rows = sorted(rows, key=lambda row: (row["timestamp"], row["symbol"]))
    blocks = time_blocks(rows, folds)
    x = feature_matrix(rows, features)
    y = np.array([float(row["future_return"]) for row in rows])
    # Group row indices by timestamp once: rescanning every row per fold is O(folds*rows).
    by_time = {}
    for index, row in enumerate(rows):
        by_time.setdefault(row["timestamp"], []).append(index)
    times = sorted(by_time)
    purge = horizon if purge_bars is None else int(purge_bars)
    # How far a label reaches past the bar it is stamped on, in bars. Taken from the data
    # when the dataset carries label end times, because the true span is not always the
    # nominal horizon: a triple-barrier label resolves early when a barrier is touched. The
    # gap below has to cover the longest label in the data, not the average one.
    label_span = _label_span_bars(rows, times, step=None) or purge

    per_fold = []
    oos = []
    collected_actual, collected_side, collected_net = [], [], []
    for fold, block in enumerate(blocks):
        first = times.index(block[0])
        cut = max(0, first - purge)
        train_idx = [i for stamp in times[:cut] for i in by_time[stamp]]
        test_idx = [i for stamp in block for i in by_time[stamp]]
        if not train_idx or not test_idx:
            continue
        # The early-stopping set has to be separated from the fit set by the same reasoning
        # that separates train from test: a label stamped in the fit window resolves inside
        # the validation window when the two are adjacent, so the iteration count chosen on
        # it is chosen partly on rows the model already memorised. Measured on a real
        # dataset, the boundary was at 0 bars while labels reached 12, so the stopping point
        # was being picked on information from the training rows.
        fit_idx, valid_idx = _purged_split(train_idx, rows, by_time, times, label_span)
        model = _train(backend, parameters_for(backend), x[fit_idx], y[fit_idx],
                       x[valid_idx], y[valid_idx], rounds)
        prediction = np.asarray(model.predict(x[test_idx]))
        actual = y[test_idx]
        stats = evaluate(actual, prediction, cost_bps)
        cost = cost_bps / 10000
        side = np.where(prediction > cost, 1, np.where(prediction < -cost, -1, 0))
        active = side != 0
        collected_actual.append(actual[active])
        collected_net.append(side[active] * actual[active] - cost)
        per_fold.append({"fold": fold, "test_start": block[0], "train_rows": len(fit_idx),
                         "test_rows": len(test_idx), **stats})
        if return_oos:
            for offset, index in enumerate(test_idx):
                row = rows[index]
                oos.append({"fold": fold, "timestamp": int(row["timestamp"]),
                            "symbol": row.get("symbol"),
                            "predicted_return": float(prediction[offset]),
                            "actual_return": float(actual[offset]),
                            "side": int(side[offset])})

    result = {"folds": len(per_fold), "backend": backend, "cost_bps": cost_bps,
              "purge_bars": purge, "label_span_bars": label_span, "per_fold": per_fold,
              "aggregate": aggregate(per_fold),
              "bootstrap": bootstrap_edge(np.concatenate(collected_net) if collected_net
                                          else np.array([]))}
    if return_oos:
        result["oos"] = oos
    return result


def aggregate(per_fold):
    if not per_fold:
        return {"folds": 0}
    rows = sum(f["rows"] for f in per_fold)
    active = sum(f["active_samples"] for f in per_fold)
    weighted_edge = ([f["active_net_edge_bps"] * f["active_samples"] for f in per_fold
                      if f["active_samples"]])
    weights = [f["active_samples"] for f in per_fold if f["active_samples"]]
    edges = [f["active_net_edge_bps"] for f in per_fold]
    return {"folds": len(per_fold), "rows": rows, "active_samples": active,
            "active_share_pct": 100.0 * active / rows if rows else 0.0,
            "dir_acc_pct": sum(f["directional_accuracy_pct"] * f["rows"] for f in per_fold) / rows if rows else 0.0,
            "net_edge_bps": sum(weighted_edge) / sum(weights) if weights else 0.0,
            "folds_with_activity": len(weights),
            "folds_positive_edge": sum(1 for e in edges if e > 0),
            "folds_negative_edge": sum(1 for e in edges if e < 0)}


def bootstrap_edge(net_returns, iterations=2000, block=None, seed=12345):
    """Moving-block bootstrap of the mean net return per active trade.

    Overlapping labels make neighbouring trades dependent, so an i.i.d. bootstrap would
    report a confidence interval that is too narrow. Resampling contiguous blocks keeps
    some of that dependence and gives the conservative interval.
    """
    net_returns = np.asarray(net_returns, dtype=float)
    count = net_returns.size
    if count < 2:
        return {"samples": int(count), "mean_bps": float(net_returns.mean() * 10000) if count else 0.0,
                "low_bps": 0.0, "high_bps": 0.0, "prob_positive": 0.0,
                "block": 0, "iterations": 0}
    size = int(block or max(2, int(np.sqrt(count))))
    size = min(size, count)
    starts = np.arange(count - size + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    needed = int(np.ceil(count / size))
    for step in range(iterations):
        picked = rng.choice(starts, size=needed, replace=True)
        sample = np.concatenate([net_returns[s:s + size] for s in picked])[:count]
        means[step] = sample.mean()
    return {"samples": count, "mean_bps": float(net_returns.mean() * 10000),
            "low_bps": float(np.percentile(means, 2.5) * 10000),
            "high_bps": float(np.percentile(means, 97.5) * 10000),
            "prob_positive": float(np.mean(means > 0)),
            "block": size, "iterations": iterations}
