"""Combinatorial purged cross-validation, and the overfitting probability it measures.

The existing walk-forward produces four folds on this dataset, and four folds is not enough
to decide anything. The measured evidence says so: the four-fold bootstrap interval on the
1.26M-row set ran from about -38 bp to +16 bp, and an interval that wide cannot separate
"there is no signal" from "there is a signal and this sample cannot see it". Every candidate
model was reported with P(edge>0) between 0.08 and 0.27 -- a number as consistent with a real
edge as with noise, and it was therefore acted on as though it meant the latter.

Lopez de Prado combinatorial purged CV fixes the sample-size problem directly. Instead of
one chronological ordering -- which yields N folds and therefore one path -- split the
timeline into N disjoint groups, then hold out every combination of k groups. That produces
C(N, k) test sets and, after reassembling them in the original order, C(N, k) * k / N
complete backtest paths. With N=8, k=2 that is 28 test sets and 7 full paths from the same
data and the same labels: seven times the evidence about the only question that matters,
which is whether an edge survives across different market regimes rather than across one
lucky ordering.

Two things make it honest rather than merely larger:

* **Purging.** A label stamped in the training set resolves at a bar that may fall inside a
  test group. Those rows are dropped from the fit.
* **Embargo.** Even after purging, serial correlation leaks across a boundary. A short
  embargo after every test group removes rows whose features were computed from bars inside
  it. The purge covers the label forward reach; the embargo covers the feature backward
  reach. Doing only one of the two is the standard way to get an optimistic number while
  believing the result is purged.

The module also computes the probability of backtest overfitting (PBO) by the
combinatorially symmetric cross-validation of Bailey et al.: for each split, rank the N
candidate configurations in-sample, take the best, and record where that configuration
lands out-of-sample. PBO is the share of splits where the in-sample winner is below the
out-of-sample median. It answers a question a single deflated Sharpe cannot: not "is this
one number significant" but "does the selection procedure that produced it generalise".
"""
import itertools
import math

import numpy as np

# Rows closer than this many bars are dropped between a test group and any training row.
# Expressed in bars; the caller defaults it to the label horizon.
DEFAULT_EMBARGO_BARS = 12


def group_bounds(times, groups):
    """Split sorted timestamps into `groups` contiguous blocks of near-equal size."""
    if groups < 2:
        raise ValueError("invalid_group_count")
    if len(times) < groups * 2:
        raise ValueError("too_few_timestamps")
    size = len(times) // groups
    bounds = []
    for index in range(groups):
        start = index * size
        stop = len(times) if index == groups - 1 else (index + 1) * size
        bounds.append((start, stop))
    return bounds


def combinations(groups, k):
    """Every choice of k held-out groups from N, ordered for reproducibility."""
    if k < 1 or k > groups:
        raise ValueError("invalid_test_groups")
    return list(itertools.combinations(range(groups), k))


def path_count(groups, k):
    """How many complete backtest paths C(N,k) splits can be assembled into."""
    return int(math.comb(groups, k) * k // groups)


def _index_by_time(rows):
    by_time = {}
    for index, row in enumerate(rows):
        by_time.setdefault(row["timestamp"], []).append(index)
    return by_time


def _median_step(test_sorted):
    """Median spacing between consecutive test timestamps, in milliseconds."""
    if len(test_sorted) <= 1:
        return 300_000
    spacings = [b - a for a, b in zip(test_sorted, test_sorted[1:]) if b > a]
    if not spacings:
        return 300_000
    return sorted(spacings)[len(spacings) // 2]


def _purge(train_times, test_times, purge_bars, embargo_bars):
    """Keep the training timestamps separated from the whole test region.

    The two guards are not the same guard applied twice: purge_bars removes training rows
    whose label reaches forward into the test region, and embargo_bars removes rows whose
    features reach backward out of it. A row must be clear on both sides, which is why this
    compares each candidate against the test region as a whole rather than only its edges.
    """
    if not test_times:
        return set(train_times)
    test_sorted = sorted(test_times)
    low, high = test_sorted[0], test_sorted[-1]
    pad = (purge_bars + embargo_bars) * _median_step(test_sorted)
    return {stamp for stamp in train_times if stamp < low - pad or stamp > high + pad}


def cpcv_splits(rows, groups=8, k=2, purge_bars=12, embargo_bars=None):
    """The (train_idx, test_idx) pairs of a combinatorial purged CV.

    Pure index arithmetic: no model is fit here, so the split structure can be inspected and
    unit-tested on its own.
    """
    if embargo_bars is None:
        embargo_bars = DEFAULT_EMBARGO_BARS
    ordered = sorted(rows, key=lambda row: (row["timestamp"], row.get("symbol") or ""))
    by_time = _index_by_time(ordered)
    times = sorted(by_time)
    bounds = group_bounds(times, groups)
    splits = []
    for combo in combinations(groups, k):
        test_times = set()
        for group in combo:
            start, stop = bounds[group]
            test_times.update(times[start:stop])
        # Candidates exclude the test groups entirely, so a train set can never contain a
        # test bar even before the guard is applied.
        outside = [stamp for stamp in times if stamp not in test_times]
        kept = _purge(outside, test_times, purge_bars, embargo_bars)
        train_idx = [i for stamp in outside if stamp in kept for i in by_time[stamp]]
        test_idx = [i for stamp in times if stamp in test_times for i in by_time[stamp]]
        if not train_idx or not test_idx:
            continue
        splits.append({"groups": combo, "train_idx": train_idx, "test_idx": test_idx,
                       "test_start": min(test_times), "test_end": max(test_times),
                       "train_rows": len(train_idx), "test_rows": len(test_idx)})
    return splits


def _summarise(per_split, paths):
    total_rows = sum(s["test_rows"] for s in per_split)
    total_active = sum(s["active_samples"] for s in per_split)
    weighted = sum(s["net_edge_bps"] * s["active_samples"] for s in per_split)
    edges = [s["net_edge_bps"] for s in per_split]
    out = {
        "splits": len(per_split),
        "rows": total_rows,
        "active_samples": total_active,
        "active_share_pct": round(100.0 * total_active / total_rows, 6) if total_rows else 0.0,
        "net_edge_bps": round(weighted / total_active, 4) if total_active else 0.0,
        "splits_positive_edge": sum(1 for e in edges if e > 0),
        "splits_negative_edge": sum(1 for e in edges if e < 0),
    }
    path_edges = np.array([p["net_bps"] for p in paths], dtype=float)
    if path_edges.size:
        std = float(path_edges.std(ddof=1)) if path_edges.size > 1 else 0.0
        out.update({
            "path_net_bps_mean": round(float(path_edges.mean()), 4),
            "path_net_bps_std": round(std, 4),
            "path_net_bps_min": round(float(path_edges.min()), 4),
            "path_net_bps_max": round(float(path_edges.max()), 4),
            "paths_positive": int((path_edges > 0).sum()),
            "paths_negative": int((path_edges < 0).sum()),
            "path_sharpe": round(float(path_edges.mean()) / std, 4) if std > 0 else 0.0,
        })
    return out


def assemble_paths(by_row, splits, k):
    """Recombine per-split outcomes into complete, non-overlapping backtest paths.

    A row is observed in several splits; a path is a choice of one split per row such that
    every timestamp is covered once. Splits are assigned to paths by position, which is the
    standard reconstruction and is what makes the paths independent enough to read as
    separate backtests.
    """
    paths = []
    count = max(1, len(splits) // max(1, k))
    for path in range(count):
        members = set(range(path * k, min(len(splits), (path + 1) * k)))
        nets, sides = [], []
        for outcomes in by_row.values():
            for number, side, net in outcomes:
                if number in members:
                    nets.append(net)
                    sides.append(side)
                    break
        if nets:
            paths.append({"path": path, "samples": len(nets),
                          "net_bps": round(float(np.mean(nets)) * 10000, 4),
                          "active_samples": int(sum(1 for s in sides if s != 0))})
    return paths


def cpcv_evaluate(rows, fit_predict, groups=8, k=2, purge_bars=12, embargo_bars=None,
                  cost_bps=12.0, features=None):
    """Run fit_predict over every CPCV split and summarise the resulting paths.

    fit_predict(train_rows, test_rows) -> predicted_return is supplied by the caller so this
    module stays free of any particular learner. Predictions are collected per split and then
    reassembled into complete paths, so the caller gets both per-split evidence and the
    path-level distribution the p-value is read from.
    """
    if features is None:
        from ..features.feature_spec import FEATURES as features  # noqa: N813
    cost = cost_bps / 10000.0
    splits = cpcv_splits(rows, groups=groups, k=k, purge_bars=purge_bars,
                         embargo_bars=embargo_bars)
    by_row = {}
    per_split = []
    for number, split in enumerate(splits):
        train_rows = [rows[i] for i in split["train_idx"]]
        test_rows = [rows[i] for i in split["test_idx"]]
        predicted = np.asarray(fit_predict(train_rows, test_rows, features), dtype=float)
        actual = np.array([float(row["future_return"]) for row in test_rows])
        side = np.where(predicted > cost, 1, np.where(predicted < -cost, -1, 0))
        net = side * actual - np.where(side != 0, cost, 0.0)
        for offset, row in enumerate(test_rows):
            key = (int(row["timestamp"]), row.get("symbol"))
            by_row.setdefault(key, []).append((number, float(side[offset]), float(net[offset])))
        active = side != 0
        per_split.append({
            "split": number, "groups": list(split["groups"]),
            "test_rows": len(test_rows), "train_rows": len(train_rows),
            "active_samples": int(active.sum()),
            "active_share_pct": round(100.0 * float(active.sum()) / len(test_rows), 6),
            "dir_acc_pct": round(100.0 * float((np.sign(predicted) == np.sign(actual)).mean()), 4),
            "net_edge_bps": round(float(net[active].mean()) * 10000, 4) if active.any() else 0.0,
        })
    paths = assemble_paths(by_row, splits, k)
    out = _summarise(per_split, paths)
    out.update({"combinations": len(splits), "paths": len(paths), "groups": groups, "k": k,
                "cost_bps": cost_bps, "per_split": per_split, "path_detail": paths})
    return out


def probability_of_backtest_overfitting(matrix, rounds=None):
    """PBO of Bailey et al. from an (observations x configurations) performance matrix.

    matrix holds one out-of-sample performance number per observation per candidate
    configuration. For each random half the configurations are ranked in-sample, the best is
    selected, and its rank out-of-sample is recorded. PBO is the fraction of splits in which
    the in-sample winner lands below the out-of-sample median -- the probability that the
    selection procedure picks a configuration that does not survive.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix_must_be_2d")
    observations, configs = matrix.shape
    if configs < 2 or observations < 4:
        return {"pbo": None, "reason": "insufficient_data", "configurations": configs,
                "observations": observations}
    rng = np.random.default_rng(20240914)
    rounds = 200 if rounds is None else int(rounds)
    half = observations // 2
    logits = []
    for _ in range(rounds):
        order = rng.permutation(observations)
        in_sample = matrix[order[:half]]
        out_sample = matrix[order[half:]]
        means = out_sample.mean(axis=0)
        best = int(np.argmax(in_sample.mean(axis=0)))
        rank = float((means < means[best]).sum() + 0.5 * (means == means[best]).sum()) / configs
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": round(float((logits <= 0).mean()), 4), "rounds": rounds,
            "configurations": configs, "observations": observations}
