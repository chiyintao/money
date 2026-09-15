"""Gross-expectation test of the ranking signal on the full 894-day panel.

This answers one question and refuses to answer any other: if you had traded the signal
that the decision layer actually ranks on, taking the top and bottom cross-section each
day and holding one day, what would you have earned BEFORE any costs?

Costs are excluded deliberately. Including them would let a negative result be blamed on
the fee assumption, and the previous finding -- that these signals were negative even
with fees stripped out -- makes that the wrong thing to hide behind. If the gross number
is negative, no cost model rescues it and no sizing rule changes the sign.
"""
import collections
import json
import math
import statistics

PANEL = "data/research_v4/panel_mainstream.jsonl"


def load():
    rows = []
    with open(PANEL, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    by_day = collections.defaultdict(list)
    for row in rows:
        by_day[row["day"]].append(row)
    return rows, by_day


def rank(values):
    """Average-rank, so ties do not depend on input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def signal(rows, lookback=7, field="ret_7d"):
    """The cross-sectional momentum rank the model-free baseline uses.

    The fitted suite is not tested here. It lost to this rule on the windows it was
    measured on, and re-fitting it on the full panel to then judge it on that same panel
    would be in-sample and would not be evidence.
    """
    values = [r.get(field) for r in rows]
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    return rank(values)


def run(by_day, lookback=7, top=2, bottom=2, field="ret_7d"):
    days = sorted(by_day)
    deltas = []
    per_day = []
    for day in days:
        rows = [r for r in by_day[day]
                if r.get("fwd_1d") is not None and math.isfinite(r["fwd_1d"])]
        if len(rows) < 8:
            continue
        scores = signal(rows, lookback, field)
        if scores is None:
            continue
        order = sorted(range(len(rows)), key=lambda i: scores[i])
        longs = order[-top:]
        shorts = order[:bottom]
        long_ret = sum(rows[i]["fwd_1d"] for i in longs) / len(longs)
        short_ret = sum(rows[i]["fwd_1d"] for i in shorts) / len(shorts)
        # Dollar-neutral: the edge per unit of gross exposure is (long - short) / 2,
        # because one unit of gross exposure is carried as one unit long and one short.
        edge = (long_ret - short_ret) / 2.0
        deltas.append(long_ret - short_ret)
        per_day.append({"day": day, "edge": edge, "long": long_ret, "short": short_ret})
    return deltas, per_day


def stats(values):
    n = len(values)
    if n < 2:
        return {"n": n}
    mean = sum(values) / n
    sd = statistics.stdev(values)
    stderr = sd / math.sqrt(n)
    t = mean / stderr if stderr else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "t": t,
            "mean_bps": mean * 10000, "t_annual": t * math.sqrt(365) if n else float("nan")}


if __name__ == "__main__":
    rows, by_day = load()
    deltas, per_day = run(by_day)
    result = stats(deltas)
    print("GROSS EXPECTATION, full panel, no costs at all")
    print("  days traded      : %d" % result["n"])
    print("  mean daily spread: %+.6f  (%+.2f bps of gross)" % (result["mean"], result["mean_bps"]))
    print("  daily sd         : %.6f" % result["sd"])
    print("  t-statistic      : %.3f" % result["t"])
    print()
    # Split the history in half. A result that lives in one half only is a regime, not an
    # edge -- that distinction is the whole reason the earlier 63 bps figure collapsed.
    half = len(deltas) // 2
    for label, part in (("first half", deltas[:half]), ("second half", deltas[half:])):
        s = stats(part)
        print("  %-12s n=%3d  %+8.2f bps  t=%6.2f" % (label, s["n"], s["mean_bps"], s["t"]))
    print()
    # Year by year, so a single good stretch cannot carry the average.
    by_year = collections.defaultdict(list)
    for entry in per_day:
        by_year[entry["day"] // 365].append(entry["edge"])
    print("  by panel-year (day/365):")
    for y in sorted(by_year):
        s = stats(by_year[y])
        print("    year %-3d n=%3d  %+8.2f bps  t=%6.2f" % (y, s["n"], s["mean_bps"], s["t"]))
