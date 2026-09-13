"""Streaming access to training datasets.

A pooled multi-symbol dataset is far too large to load, sort and re-serialize in
memory just to fingerprint it, which is what the previous implementation did on every
live start. These helpers make a single streaming pass and return everything the model
runtime needs: the dataset digest, the feature bounds and the covered symbols.

The digest reproduces the historical value exactly --
``sha256(json.dumps(sorted(rows, key=(timestamp, symbol)), sort_keys=True))`` -- so
existing model manifests stay verifiable. Rows are therefore re-serialized with sorted
keys rather than hashed from the raw line, which is what makes the equality hold
regardless of how the file was written.
"""
import hashlib
import json
import time
from pathlib import Path

from ..features.feature_spec import FEATURE_CLIP, FEATURES

# Bins used for the robust range estimate. 512 over a bounded feature resolves the tails
# far more finely than the decision needs, and the memory is fixed regardless of dataset
# size: ten features, one list of counts each.
HISTOGRAM_BINS = 512
# Tail fraction discarded from each end before the bound is read off.
#
# Chosen deliberately small. This is a *blocking* gate, so every basis point of tail it
# trims is a basis point of ordinary bars it will refuse; at 0.1% per end it refuses about
# 0.2% of in-sample bars, which is the price of a gate that can fire at all. It is still
# enough to matter: on the 1.26M-row dataset a thousand-odd rows saturate each clipped
# feature, and trimming them is exactly what pulls the bound inside the clip range.
BOUND_TAIL = 0.001


def _histogram(limits):
    """Counting bins spanning a feature's a-priori clip range."""
    low, high = limits
    width = (high - low) / HISTOGRAM_BINS
    return {"low": low, "high": high, "width": width, "counts": [0] * HISTOGRAM_BINS,
            "below": 0, "above": 0}


def _observe(hist, value):
    if value < hist["low"]:
        hist["below"] += 1
        return
    if value > hist["high"]:
        hist["above"] += 1
        return
    index = int((value - hist["low"]) / hist["width"])
    if index >= HISTOGRAM_BINS:
        index = HISTOGRAM_BINS - 1
    hist["counts"][index] += 1


def _quantile(hist, fraction):
    """Edges of the bin holding a tail fraction of the observed distribution.

    Returns (low_edge, high_edge) of that bin, or None when nothing was observed -- which
    the caller reports as an unverifiable bound rather than substituting the clip range.
    Edges rather than centres so the caller can clamp the estimate against the smallest and
    largest values actually seen and never produce a bound outside the data.
    """
    total = sum(hist["counts"]) + hist["below"] + hist["above"]
    if total <= 0:
        return None
    target = fraction * total
    seen = 0
    for index, count in enumerate(hist["counts"]):
        if count and seen + count >= target:
            low = hist["low"] + index * hist["width"]
            return low, low + hist["width"]
        seen += count
    return None


def iter_rows(path):
    """Yield each dataset row, skipping blank lines."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_profile(path, features=FEATURES):
    """Digest, feature bounds, symbols and row count in one pass.

    Returns a dict with keys: rows, sha256, bounds (name -> [min, max]), symbols.
    """
    digest = hashlib.sha256()
    digest.update(b"[")
    hists = {}
    observed = set()
    extremes = {}
    symbols = set()
    rows = 0
    first = True  # the separator goes between rows, so it precedes every row but the first
    for row in iter_rows(path):
        if not first:
            digest.update(b", ")
        first = False
        digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8"))
        rows += 1
        symbol = row.get("symbol")
        if isinstance(symbol, str) and symbol:
            symbols.add(symbol)
        for name in features:
            value = row.get(name)
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                continue
            observed.add(name)
            bounds = extremes.get(name)
            if bounds is None:
                extremes[name] = [value, value]
            else:
                if value < bounds[0]:
                    bounds[0] = value
                if value > bounds[1]:
                    bounds[1] = value
            limits = FEATURE_CLIP.get(name)
            if limits is None:
                continue
            hist = hists.get(name)
            if hist is None:
                hist = hists[name] = _histogram(limits)
            _observe(hist, value)
    digest.update(b"]")
    # Robust tail bounds, not min and max. The previous bounds were the extremes of the
    # data, and because the features had already been clipped to FEATURE_CLIP before being
    # written, the extremes *were* FEATURE_CLIP for every feature that ever touched its
    # limit. The stored profile documents this exactly: return_10, funding_z, mark_basis
    # and rsi all had bounds equal to the clip range, so the serving gate could only fire
    # on a value outside the range its own inputs were clipped to -- which never happens.
    # The gate was structurally incapable of refusing anything.
    bounds = {}
    for name in sorted(observed):
        hist = hists.get(name)
        lowest, highest = extremes[name]
        quoted = None
        if hist is not None:
            first, last = _quantile(hist, BOUND_TAIL), _quantile(hist, 1.0 - BOUND_TAIL)
            if first is not None and last is not None:
                # Clamped against the values actually seen. Without this a small dataset
                # reads its bound off a bin edge that sits just outside its own range, and
                # a row from the training set is then reported as out of distribution by
                # the model that was fitted on it.
                low = max(first[0], lowest)
                high = min(last[1], highest)
                if low < high:
                    quoted = [low, high]
        bounds[name] = quoted if quoted is not None else [lowest, highest]
    return {"rows": rows, "sha256": digest.hexdigest(), "bounds": bounds,
            "extremes": {name: extremes[name] for name in sorted(extremes)},
            "symbols": sorted(symbols),
            "bounds_method": "tail_%g" % BOUND_TAIL,
            "features_missing": sorted(set(features) - observed)}


def stream_digest(path):
    return stream_profile(path)["sha256"]


def profile_cache_path(path):
    return Path(str(path) + ".profile.json")


def cached_profile(path, features=FEATURES, use_cache=True):
    """Profile the dataset, reusing a sidecar cache while the file is unchanged.

    Fingerprinting 400MB of rows costs about half a minute, which would otherwise be
    paid on every live start. The cache is keyed on file size and mtime and is discarded
    whenever either changes, so a replaced dataset can never be served stale bounds.
    The tradeoff is explicit: a file edited in place while keeping its size and mtime
    would not be re-read.
    """
    path = Path(path)
    stat = path.stat()
    cache_path = profile_cache_path(path)
    if use_cache and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if (cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and cached.get("features") == list(features)):
            # The whole stored profile, not a subset of keys. Returning a subset made a
            # cache hit a different shape from a cache miss, so a caller reading any of the
            # newer keys saw None on the second run and a value on the first.
            return {key: value for key, value in cached.items()
                    if key not in ("size", "mtime_ns", "features", "cached_at")}
    profile = stream_profile(path, features)
    payload = dict(profile, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   features=list(features), cached_at=int(time.time() * 1000))
    try:
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
    return profile


def bounds_are_informative(bounds, features=FEATURES, extremes=None):
    """Whether a set of serving bounds can refuse anything at all.

    A bound equal to the a-priori clip range is not a bound: every value the pipeline can
    produce after clipping is inside it. Reported rather than fixed silently, because the
    right response is to rebuild the dataset profile, and a gate that quietly repaired
    itself would hide that the stored one is stale.
    """
    degenerate = []
    for name in features:
        limits = bounds.get(name)
        clip = FEATURE_CLIP.get(name)
        if limits is None or clip is None:
            continue
        if limits[0] <= clip[0] and limits[1] >= clip[1]:
            degenerate.append(name)
    return {"informative": not degenerate, "degenerate": degenerate,
            "checked": [name for name in features if name in bounds],
            "unbounded": [name for name in features if name not in bounds]}


def rows_digest(rows):
    """Digest for rows already in memory, without joining them into one huge string.

    A pooled dataset serializes to hundreds of megabytes; building that string only to
    hash it doubles peak memory for no reason.
    """
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, row in enumerate(rows):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()
