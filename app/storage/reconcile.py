"""Candle-series data quality: where a stored series has holes, and merging new rows in.

A candle series with a missing bar is not visibly broken. The indicators computed over it
are simply wrong -- a 20-bar average that spans 22 bars of elapsed time is not a 20-bar
average -- and nothing in the pipeline can tell, because every downstream function sees a
list of rows and has no idea one is absent. find_gaps is how a caller finds out.

Note on provenance: this module was overwritten by mistake during a refactor and these two
functions were reconstructed from their test contract, because the repository has no version
control and no other caller to recover them from. The semantics below are stated precisely
and covered by tests, but the original implementation is gone. The account-level
reconciliation that replaced it now lives in account_reconcile.py, which is where it should
have gone to begin with -- the names were similar enough that a file was clobbered.
"""


def _open_time(row):
    try:
        return int(row.get("open_time", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def find_gaps(rows, interval_ms):
    """The bars in hand that are followed by a discontinuity.

    Returns the open_time of each row whose successor is more than one interval away -- so
    the value returned is the last bar *before* the hole, which is the point a backfill
    would resume from. An empty list means the series is contiguous.

    Only forward gaps are reported. A series that goes backwards (a re-sorted input, or two
    intervals interleaved) is a different defect and is not silently folded into this one.
    """
    try:
        step = int(interval_ms)
    except (TypeError, ValueError):
        return []
    if step <= 0:
        return []
    ordered = sorted((_open_time(row) for row in rows or () if row is not None))
    gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        if current - previous > step:
            gaps.append(previous)
    return gaps


def merge_events(rows, incoming, key="open_time"):
    """Merge new rows into a series, one row per key, oldest first.

    Rows already present are updated rather than duplicated; keys not present are appended.
    The point is idempotence: the live loop re-reads the bar it is still forming on every
    tick, and appending would grow the series by one row per tick while every indicator
    computed over it silently used the oldest copy of each bar.

    Later sources win on conflicting fields, which is what makes a fresh REST response
    correct an earlier partial one.
    """
    merged = {}
    order = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        marker = row.get(key)
        if marker is None:
            continue
        if marker not in merged:
            order.append(marker)
            merged[marker] = dict(row)
        else:
            merged[marker].update(row)
    for row in incoming or ():
        if not isinstance(row, dict):
            continue
        marker = row.get(key)
        if marker is None:
            continue
        if marker not in merged:
            order.append(marker)
            merged[marker] = dict(row)
        else:
            merged[marker].update(row)
    return [merged[marker] for marker in sorted(order)]
