"""Feature drift detection.

The previous version was thirteen lines: a population stability index computed against
min/max of the reference sample, with a hardcoded 0.2 threshold. Three problems, in
increasing order of severity.

First, using min and max as the histogram range means a single outlier in the reference
sample stretches every bin, and one outlier in the current sample lands in the last bin --
so the index mostly measures outliers, not drift. Second, the threshold was a module
constant, so every feature was judged against the same number regardless of how many
observations backed it: PSI is a sample statistic and its noise floor scales with 1/n.
Third, and decisively, nothing ever called it.

The version here bins on quantiles of the reference sample, reports the per-feature index
with the share of current rows that fell outside the reference range, and applies a
threshold that widens with small samples. It also compares a serving population against
the training population, which is what makes it useful: the question is not whether the
market changed, it is whether the inputs the model is being asked about look like the
inputs it was fitted on.
"""
import bisect
import math

DEFAULT_BINS = 10
DEFAULT_THRESHOLD = 0.2
# PSI on 10 bins with a few hundred rows is noisy, so the threshold widens as the reference
# sample shrinks. A fixed constant treats a 200-row comparison and a 200000-row one alike.
SMALL_SAMPLE_PENALTY = 0.02
SMALL_SAMPLE_UNIT = 1000


def _quantile_edges(values, bins):
    """Bin edges at approximately equal counts of the reference sample.

    Equal-count rather than equal-width. Equal-width bins over min/max let one outlier
    compress every other observation into a single bin, at which point the index is
    measuring the outlier rather than the distribution.
    """
    ordered = sorted(values)
    size = len(ordered)
    if size < bins:
        return None
    edges = []
    for index in range(1, bins):
        position = index * size / bins
        low = ordered[int(position) - 1]
        high = ordered[min(int(position), size - 1)]
        edges.append((low + high) / 2 if low != high else low)
    # Collapse duplicates: a constant feature yields identical edges and zero-width bins.
    unique = []
    for edge in edges:
        if not unique or edge > unique[-1]:
            unique.append(edge)
    return unique or None


def _distribution(values, edges, bins):
    counts = [0] * bins
    outside = 0
    low, high = (edges[0], edges[-1]) if edges else (None, None)
    for value in values:
        if edges and (value <= low or value > high):
            outside += 1
        counts[bisect.bisect_right(edges, value)] += 1
    total = len(values) or 1
    # Laplace smoothing keeps an empty bin from dividing by zero and keeps a bin that is
    # empty on one side from contributing an unbounded term.
    return [(count + 1e-6) / (total + bins * 1e-6) for count in counts], outside


def psi(reference, current, bins=DEFAULT_BINS):
    """Population stability index of current against reference."""
    reference = [float(value) for value in (reference or []) if value is not None]
    current = [float(value) for value in (current or []) if value is not None]
    if len(reference) < bins or not current:
        return 0.0
    edges = _quantile_edges(reference, bins)
    if not edges:
        return 0.0
    width = len(edges) + 1
    expected, _ = _distribution(reference, edges, width)
    actual, _ = _distribution(current, edges, width)
    return sum((y - x) * math.log(y / x) for x, y in zip(expected, actual) if x > 0 and y > 0)


def feature_drift(reference, current, name=None, bins=DEFAULT_BINS,
                  threshold=DEFAULT_THRESHOLD, reference_size=None):
    """One feature index plus the evidence behind the verdict."""
    size = reference_size if reference_size is not None else len(reference or [])
    adjusted = threshold
    if size and size < SMALL_SAMPLE_UNIT:
        adjusted = threshold + SMALL_SAMPLE_PENALTY * (1 - size / float(SMALL_SAMPLE_UNIT))
    index = psi(reference, current, bins)
    edges = _quantile_edges([float(value) for value in (reference or []) if value is not None],
                            bins)
    outside = 0
    clean_current = [float(value) for value in (current or []) if value is not None]
    if edges and clean_current:
        _dist, outside = _distribution(clean_current, edges, len(edges) + 1)
    return {"name": name, "psi": index, "threshold": adjusted, "drifted": index >= adjusted,
            "reference_rows": size, "current_rows": len(clean_current),
            "outside_pct": (outside / len(clean_current) * 100) if clean_current else 0.0}


def drift_report(reference, current, threshold=DEFAULT_THRESHOLD, bins=DEFAULT_BINS,
                 names=None, limit=None):
    """Per-feature drift with an overall verdict.

    Both arguments map a feature name to a list of observed values. The names argument
    restricts the comparison, so a caller can check only the features a given model
    actually consumes rather than every column in the file.
    """
    keys = sorted(set(reference or {}) & set(current or {}))
    if names is not None:
        wanted = set(names)
        keys = [key for key in keys if key in wanted]
    features = {}
    for key in keys:
        sample = current.get(key) or []
        if limit:
            sample = list(sample)[-int(limit):]
        features[key] = feature_drift(reference.get(key), sample, name=key,
                                      bins=bins, threshold=threshold)
    drifted = sorted(key for key, value in features.items() if value["drifted"])
    return {"features": features, "drifted": drifted,
            "status": "drift" if drifted else "stable",
            "threshold": threshold, "bins": bins, "compared": len(keys)}


def drift_from_rows(reference_rows, current_rows, features, threshold=DEFAULT_THRESHOLD,
                    bins=DEFAULT_BINS):
    """Compare two row sets feature by feature.

    Rows are dicts, which is how both the training dataset and the decision records are
    stored, so a training sample and live feature values can be compared without either
    side being reshaped.
    """
    reference = {}
    current = {}
    for name in features or ():
        reference[name] = [row[name] for row in (reference_rows or [])
                           if isinstance(row.get(name), (int, float))]
        current[name] = [row[name] for row in (current_rows or [])
                         if isinstance(row.get(name), (int, float))]
    return drift_report(reference, current, threshold=threshold, bins=bins, names=features)
