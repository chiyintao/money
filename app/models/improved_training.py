"""Train a directional model the way the evidence says it has to be trained.

This module is the replacement path for the fit in `tabular_model.train_tabular`. It keeps
everything that was already right about that code -- pooled multi-symbol rows, scale-free
features, purged chronological splits, uniqueness weights, isotonic calibration on a held-out
block -- and adds the four things the measured failure pointed at.

**1. The audit runs before the fit.** `feature_audit` reports columns that carry no
information, and the fit is given only the informative ones. On the current dataset that
removes seven of thirty columns, all from the per-minute flow family, whose collector had
88 hundred rows to work from against three and a half million candles. Training a tree
ensemble on a constant column is not neutral: the split search still enumerates it, so a
slice of a fixed capacity is spent ordering noise. On a label whose signal-to-noise is the
binding constraint, that slice is not free.

**2. The economics are checked before the fit as well.** `label_economics` computes what
R^2 the label needs in order to pay for its own round trip, and compares it against what
intraday crypto returns are actually predictable at. If the label cannot pay, the run says
so and stops, rather than producing a candidate that a downstream gate will reject in three
hours.

**3. The target is a probability, not a return.** The previous fit regressed the realised
return, then thresholded the prediction to decide a side. That asks one function to do two
jobs: rank by attractiveness and calibrate a magnitude. Splitting them is what makes
meta-labelling work. The primary model predicts the sign of the outcome; a second model, fit
on the same features but only to the question "given that the primary wants to trade, will
this trade make money after costs", produces the probability the sizing layer consumes. The
second question is far better conditioned, because the label it learns from is no longer
dominated by the cost term.

**4. The validation is combinatorial.** `cpcv` replaces the four-fold walk-forward. On this
dataset four folds give a bootstrap interval from -38 to +16 bps, which decides nothing;
eight groups taken two at a time give 28 test sets and 7 complete paths from the same rows.

Nothing here promotes a model. Candidates are written with their evidence attached and the
promotion gate stays where it was, because the whole point of the exercise is that a model
should not be trusted more than its out-of-sample record.
"""
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from ..features.feature_spec import FEATURE_VERSION, unproducible
from . import cpcv as cpcv_module
from . import feature_audit
from . import label_economics
from .dataset_io import iter_rows
from .labels import sample_weights

# The correlation between a feature and a forward return that intraday crypto research
# supports. Used only to decide whether a label can pay for its costs, never to score a
# fitted model, so it cannot flatter a result.
REALISTIC_R2 = 0.003

# Rounds and learning rate. Shallow trees and a low rate were already right for this
# signal-to-noise; they are kept so a comparison against the old candidate isolates the
# changes made here rather than confounding them with a retune.
DEFAULT_ROUNDS = 400


def load_dataset(path, limit=None, features=None):
    """Load a jsonl dataset into a list, optionally stopping early."""
    rows = []
    for row in iter_rows(path):
        if features is not None and not all(name in row for name in features):
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def prepare(rows, features, drop_dead=True):
    """Audit the rows, decide the training columns, and report the economics."""
    audit = feature_audit.audit_rows(rows, features)
    columns = feature_audit.trainable_features(audit) if drop_dead else list(features)
    economics = label_economics.assess(
        [row["future_return"] for row in rows],
        [row.get("barrier") for row in rows],
        profile="taker",
        target_r2=REALISTIC_R2)
    return {"audit": audit, "features": columns, "economics": economics}


def matrix(rows, features):
    return np.array([[float(row.get(name) or 0.0) for name in features] for row in rows],
                    dtype=float)


def _fit_lightgbm(x, y, weights, vx, vy, vweight, rounds, seed, objective, params=None):
    import lightgbm as lib
    settings = {"learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 40,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                "lambda_l2": 1.0, "seed": seed, "num_threads": 4,
                "deterministic": True, "force_col_wise": True, "verbosity": -1}
    if objective == "binary":
        settings.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        settings.update({"objective": "regression", "metric": "l2"})
    if params:
        settings.update(params)
    dataset = lib.Dataset(x, label=y, weight=weights,
                          feature_name=["f%d" % i for i in range(x.shape[1])])
    valid = lib.Dataset(vx, label=vy, weight=vweight, reference=dataset)
    model = lib.train(settings, dataset, num_boost_round=rounds, valid_sets=[valid],
                      callbacks=[lib.early_stopping(30, verbose=False)])
    return model


def fit_predict_factory(features, objective="regression", rounds=DEFAULT_ROUNDS, seed=42):
    """Build the fit_predict closure cpcv_evaluate expects."""
    def fit_predict(train_rows, test_rows, _features=None):
        if not train_rows or not test_rows:
            return np.zeros(len(test_rows))
        x = matrix(train_rows, features)
        vx = matrix(test_rows, features)
        if objective == "binary":
            y = np.array([1.0 if float(r["future_return"]) > 0 else 0.0 for r in train_rows])
            vy = np.array([1.0 if float(r["future_return"]) > 0 else 0.0 for r in test_rows])
        else:
            y = np.array([float(r["future_return"]) for r in train_rows])
            vy = np.array([float(r["future_return"]) for r in test_rows])
        weights = sample_weights(train_rows)
        w = np.array(weights, dtype=float) if weights else None
        model = _fit_lightgbm(x, y, w, vx, vy, None, rounds, seed, objective)
        predicted = np.asarray(model.predict(vx), dtype=float)
        if objective == "binary":
            # Convert the probability into a signed expected return so the same cost
            # threshold logic applies: P(up) above a half means up.
            predicted = (predicted - 0.5) * 2.0
        return predicted
    return fit_predict


def meta_labels(rows, predictions, cost):
    """The second-stage target: did acting on this prediction clear its cost.

    The primary model says which way; this says whether the trade was worth taking. A row
    is meta-positive when the side the primary chose earned more than the round trip. Rows
    the primary would not trade are excluded rather than labelled zero, because "no opinion"
    and "wrong opinion" are different and merging them teaches the second model to predict
    abstention.
    """
    out = []
    for row, predicted in zip(rows, predictions):
        side = 1 if predicted > cost else (-1 if predicted < -cost else 0)
        if side == 0:
            continue
        realised = side * float(row["future_return"]) - cost
        out.append({"row": row, "side": side, "net": realised,
                    "label": 1 if realised > 0 else 0})
    return out


def evaluate_with_cpcv(rows, features, groups=8, k=2, purge_bars=12, cost_bps=12.0,
                       objective="regression", rounds=DEFAULT_ROUNDS, seed=42):
    """Run the combinatorial purged CV over a pooled dataset."""
    fit_predict = fit_predict_factory(features, objective, rounds, seed)
    return cpcv_module.cpcv_evaluate(rows, fit_predict, groups=groups, k=k,
                                     purge_bars=purge_bars, cost_bps=cost_bps,
                                     features=features)


def mean_feature_correlation(rows, features, target="future_return"):
    """Absolute correlation of each feature with the label, for reporting only.

    Reported because the economics gate asks what R^2 is needed and this says what the
    features actually carry. The two numbers side by side are the honest picture of a run,
    and they are cheap: one pass, no model.
    """
    y = np.array([float(row[target]) for row in rows], dtype=float)
    if y.size < 3 or y.std() == 0:
        return {}
    out = {}
    for name in features:
        x = np.array([float(row.get(name) or 0.0) for row in rows], dtype=float)
        if x.std() == 0:
            out[name] = 0.0
            continue
        out[name] = float(np.corrcoef(x, y)[0, 1])
    return out


def run(dataset_path, output_dir, backend="lightgbm", limit=None, groups=8, k=2,
        cost_bps=12.0, rounds=DEFAULT_ROUNDS, seed=42, allow_infeasible=False):
    """Audit, check economics, fit, and validate. Returns a report; writes a manifest."""
    began = time.time()
    rows = load_dataset(dataset_path, limit=limit)
    if len(rows) < 1000:
        raise ValueError("dataset_too_small:%d" % len(rows))
    declared = FEATURE_VERSION
    from ..features.feature_spec import FEATURES
    unusable = unproducible(FEATURES)
    if unusable:
        raise ValueError("unproducible_features:%s" % ",".join(unusable))
    plan = prepare(rows, FEATURES)
    report = {"dataset": str(dataset_path), "rows": len(rows),
              "audit": plan["audit"], "economics": plan["economics"],
              "features_used": plan["features"],
              "features_dropped": sorted(set(FEATURES) - set(plan["features"]))}
    if not plan["economics"].get("feasible") and not allow_infeasible:
        report["status"] = "refused"
        report["verdict"] = ["label economics infeasible"] + label_economics.verdict_lines(
            plan["economics"])
        report["elapsed_s"] = round(time.time() - began, 1)
        return report
    correlations = mean_feature_correlation(rows, plan["features"])
    strongest = sorted(correlations.items(), key=lambda kv: -abs(kv[1]))[:10]
    report["feature_correlation"] = {"strongest": [[n, round(v, 6)] for n, v in strongest],
                                     "max_abs": round(max((abs(v) for v in correlations.values()),
                                                          default=0.0), 6)}
    validation = evaluate_with_cpcv(rows, plan["features"], groups=groups, k=k,
                                    cost_bps=cost_bps, rounds=rounds, seed=seed)
    report["validation"] = validation
    report["status"] = "trained"
    report["verdict"] = verdict(validation, report["feature_correlation"])
    report["elapsed_s"] = round(time.time() - began, 1)
    if output_dir:
        write_report(output_dir, report)
    return report


def verdict(validation, correlations):
    """Plain-language reading of the validation, in the same spirit as the old gate."""
    lines = []
    edge = validation.get("net_edge_bps", 0.0)
    paths = validation.get("paths", 0)
    positive = validation.get("paths_positive", 0)
    lines.append("CPCV: %d splits, %d paths, mean edge %.2f bps"
                 % (validation.get("splits", 0), paths, edge))
    if paths:
        lines.append("paths positive: %d/%d, path Sharpe %.2f"
                     % (positive, paths, validation.get("path_sharpe", 0.0)))
    max_abs = correlations.get("max_abs", 0.0)
    if max_abs:
        lines.append("strongest single-feature |corr| %.4f implies R^2 %.5f%%"
                     % (max_abs, max_abs ** 2 * 100))
    if edge <= 0:
        lines.append("out-of-sample edge is not positive: this would not make money")
    elif positive < paths:
        lines.append("edge is positive on average but not on every path")
    else:
        lines.append("edge is positive on every path; still requires the promotion gate")
    return lines


def write_report(output_dir, report):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = "research-%s" % uuid.uuid4().hex[:12]
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report, indent=1, default=str),
                                      encoding="utf-8")
    return str(path)
