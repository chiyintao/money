"""Feature audit: refuse to train on columns that carry no information.

The previous training path checked that every declared feature was *present* and that the
dataset was large enough. It never asked whether a feature varied. Those are different
questions, and the second one is the one that costs money.

The measured failure this module exists for: of the 30 features in `features-v4`, seven --
`flow_delta_ratio`, `flow_delta_z`, `flow_trade_intensity_z`, `flow_largest_trade_z`,
`liquidation_pressure`, `liquidation_share`, `liquidation_z` -- were constant 0.0 across
the entire 1.26M-row dataset. The `flow` table had 8,854 rows against 3.5M candles, so the
family had almost nothing to compute from and every value fell back to the default. Nothing
noticed: a constant column is inside every out-of-range bound, produces finite gradients, and
makes a tree learner look like it works.

A constant column is not merely useless. It is actively harmful on a low signal-to-noise
problem, which is what a 5-minute return label is: the split search in LightGBM and the
feature-combination search in CatBoost still enumerate it, so a slice of a fixed model
capacity is spent ranking noise. With 23 of 30 columns carrying signal, that slice is real.

The gate reports three distinct defects, because they need different responses:

* **constant** -- no variance at all. Never trainable. Remove the feature or fix its
  collector; there is no third option.
* **degenerate** -- varies, but one value covers almost everything (the "constant with a
  few typos" case, e.g. a liquidation field that is 0.0 on 99.99% of bars). Not
  automatically fatal, because a rare-event feature can still be informative, but it must be
  declared rather than discovered later.
* **unmeasured** -- the column is absent or non-numeric in a large share of rows. Distinct
  from constant: a column that is missing half the time is a collection problem with a
  collection fix.

Everything here reads a single streaming pass and allocates a fixed amount of memory per
feature, so it composes with `dataset_io.stream_profile` on a 1.5 GB dataset.
"""
from collections import Counter

from ..features.feature_spec import FEATURES, FAMILY_FEATURES

# A column whose distinct-value count is at or below this is constant for practical
# purposes. Exactly 1 is a true constant; the gate uses the observed value count rather
# than a float tolerance because these features are already clipped to a fixed grid, so
# genuine variation produces many distinct values and a dead column produces one.
CONSTANT_MAX_DISTINCT = 1

# Share of rows a single value must cover before a column is called degenerate. Set high
# on purpose: a rare-event feature (a liquidation that fires on 2% of bars) is legitimate
# and this must not reject it. It rejects the 99.9%-one-value case, which is the shape a
# broken collector actually produces.
DEGENERATE_SHARE = 0.999

# Share of rows a feature may be missing before the audit calls it unmeasured.
MAX_MISSING_SHARE = 0.02

# Cap on the distinct-value counter per feature. Beyond this the column is certainly not
# constant, and the exact count is no longer interesting, so memory stays bounded no
# matter how many rows are streamed.
DISTINCT_CAP = 4096


class FeatureAudit:
    """Running per-feature statistics: distinct values, mode share, and missingness."""

    def __init__(self, features=FEATURES):
        self.features = tuple(features)
        self.rows = 0
        self.missing = {name: 0 for name in self.features}
        self.counts = {name: Counter() for name in self.features}
        self.capped = {name: False for name in self.features}

    def observe(self, row):
        self.rows += 1
        for name in self.features:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.missing[name] += 1
                continue
            value = float(value)
            # NaN and the infinities are treated as missing rather than as values: a NaN
            # mode would otherwise be reported as "the most common value", which is a
            # sentence that cannot be acted on.
            if value != value or value in (float("inf"), float("-inf")):
                self.missing[name] += 1
                continue
            if self.capped[name]:
                continue
            counter = self.counts[name]
            counter[value] += 1
            if len(counter) > DISTINCT_CAP:
                self.capped[name] = True
                # The column is provably not constant; keep the counter small.
                self.counts[name] = Counter()

    def verdicts(self):
        """Per-feature classification, worst defect first."""
        out = []
        for name in self.features:
            missing = self.missing[name]
            missing_share = missing / self.rows if self.rows else 0.0
            counter = self.counts[name]
            if self.capped[name]:
                distinct = DISTINCT_CAP + 1
                mode_share = 0.0
                mode = None
            elif not counter:
                distinct = 0
                mode_share = 1.0
                mode = None
            else:
                distinct = len(counter)
                mode, hits = counter.most_common(1)[0]
                mode_share = hits / max(1, self.rows - missing)
            if missing_share > MAX_MISSING_SHARE:
                status = "unmeasured"
            elif distinct <= CONSTANT_MAX_DISTINCT:
                status = "constant"
            elif mode_share >= DEGENERATE_SHARE:
                status = "degenerate"
            else:
                status = "ok"
            out.append({"feature": name, "status": status, "distinct": distinct,
                        "mode": mode, "mode_share": round(mode_share, 6),
                        "missing_share": round(missing_share, 6),
                        "family": _family_of(name)})
        order = {"constant": 0, "unmeasured": 1, "degenerate": 2, "ok": 3}
        out.sort(key=lambda item: (order[item["status"]], item["feature"]))
        return out

    def report(self):
        verdicts = self.verdicts()
        by_status = Counter(item["status"] for item in verdicts)
        dead = [item["feature"] for item in verdicts
                if item["status"] in ("constant", "unmeasured")]
        degenerate = [item["feature"] for item in verdicts if item["status"] == "degenerate"]
        informative = [item["feature"] for item in verdicts if item["status"] == "ok"]
        return {"rows": self.rows, "features": len(self.features),
                "counts": dict(by_status), "informative": informative,
                "dead": dead, "degenerate": degenerate,
                "wasted_share": round(len(dead) / len(self.features), 4) if self.features else 0.0,
                "verdicts": verdicts}


def _family_of(name):
    for family, names in FAMILY_FEATURES.items():
        if name in names:
            return family
    return "unknown"


def audit_rows(rows, features=FEATURES):
    """Audit an iterable of dataset rows."""
    audit = FeatureAudit(features)
    for row in rows:
        audit.observe(row)
    return audit.report()


def audit_dataset(path, features=FEATURES):
    """Audit a jsonl dataset in one streaming pass."""
    from .dataset_io import iter_rows
    return audit_rows(iter_rows(path), features)


def trainable_features(report, drop=("constant", "unmeasured")):
    """The features a fit should actually be given, given an audit report.

    Degenerate columns are deliberately kept. The gate's job is to make the decision
    visible, not to make it: a feature that is 99.95% zero may be exactly the rare
    liquidation signal the strategy wants, and silently dropping it would remove the only
    column that describes the event. Constant and unmeasured columns have no such reading --
    they are a collector that did not run -- so those go.
    """
    drop = set(drop)
    return [item["feature"] for item in report["verdicts"] if item["status"] not in drop]


def blocked(report, allow_degenerate=False):
    """Whether the audit should stop a training run, and why.

    Only *dead* columns block, and only as a report: the caller decides whether to drop
    them (the useful default) or fail. Degenerate columns never block. This is deliberately
    not a hard failure inside the audit, because the honest response to seven dead columns
    is to train without them, and a gate that refuses to continue teaches people to disable
    gates.
    """
    reasons = []
    if report["dead"]:
        reasons.append("dead_features:%s" % ",".join(report["dead"]))
    if report["degenerate"] and not allow_degenerate:
        reasons.append("degenerate_features:%s" % ",".join(report["degenerate"]))
    return reasons
