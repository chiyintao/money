"""Sweep every feature in the panel through the calibrated gate.

The momentum family is dead, but the panel carries 55 columns and the previous work only
ever ranked one of them. This is a search, so it carries a multiple-testing problem: with
enough candidates one will clear t=2 by luck. The handling is to report every candidate
including the failures, and to require a candidate to hold up on a held-out half before
anything is built on it.

The gate itself was calibrated first (scripts/gate_calibration.py): it fails five shuffled
nulls and passes a stale oracle, so a rejection here means something.
"""
import collections
import json
import math
import statistics
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe

rows, by_day = pe.load()
days = sorted(by_day)

# Features worth testing. Excluded: anything derived from the forward return (that would
# be the oracle), and identifiers. Each is a cross-sectional rank, which is what the
# decision layer does -- so no fitting is needed and no parameters are chosen here.
CANDIDATES = [
    "ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d",           # momentum, both signs
    "vol_7d", "vol_30d", "vol_ratio", "atr_pct", "range_pct",     # volatility
    "close_pos", "range_z",                                        # position in range
    "buy_vol_share", "taker_7d", "taker_daily",                    # flow
    "trades_z", "avg_trade_size",                                  # activity
    "mkt_ret_1d",                                                  # market state
]


def spread(field, clip=None, top=2, bottom=2, sign=1, exclude=()):
    """Daily dollar-neutral spread from ranking `field` cross-sectionally.

    `clip` bounds each trade's forward return, which is how tail dependence is measured.
    `exclude` drops coins before ranking, which is how single-coin dependence is measured.
    Both are needed: a candidate that only survives untested is not a candidate.
    """
    out = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])
              and x["coin"] not in exclude]
        if len(rr) < 8:
            continue
        values = [r.get(field) for r in rr]
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        # A constant column carries no cross-sectional information. Ranking it would
        # produce ties and a meaningless spread, so it is skipped rather than reported as
        # a zero result.
        if len(set(values)) < 3:
            return None
        scores = pe.rank([sign * v for v in values])
        order = sorted(range(len(rr)), key=lambda i: scores[i])

        def value(i):
            v = rr[i]["fwd_1d"]
            return max(-clip, min(clip, v)) if clip else v

        out.append(sum(value(i) for i in order[-top:]) / top
                   - sum(value(i) for i in order[:bottom]) / bottom)
    return out


def halved(values):
    h = len(values) // 2
    return pe.stats(values[:h]), pe.stats(values[h:])


def main():
    print("FEATURE SWEEP -- every candidate, both directions, through the gate")
    print("=" * 96)
    results = []
    for field in CANDIDATES:
        for sign, label in ((1, "+"), (-1, "-")):
            values = spread(field, sign=sign)
            if not values:
                continue
            raw = pe.stats(values)
            # Clip the per-trade forward return, NOT the daily spread. Clipping the
            # spread after the legs have already been averaged leaves a single extreme
            # coin able to move its day by at most 10% of the spread, which is a far
            # weaker test: it removes the largest days but keeps the largest coins on
            # ordinary days. The first version of this file did exactly that and reported
            # 82% retention on a signal whose true retention is 50%. The bars below are
            # only meaningful against the per-trade construction.
            clipped_values = spread(field, sign=sign, clip=0.10)
            clip = pe.stats(clipped_values)
            retained = clip["mean"] / raw["mean"] if raw["mean"] else 0.0
            first, second = halved(values)
            ok = (raw["t"] >= 2.0 and clip["t"] >= 2.0 and retained >= 0.5
                  and first["mean"] > 0 and second["mean"] > 0)
            results.append({
                "field": field, "sign": label, "raw_bps": raw["mean_bps"], "t": raw["t"],
                "clip_bps": clip["mean_bps"], "clip_t": clip["t"], "retained": retained,
                "first_bps": first["mean_bps"], "second_bps": second["mean_bps"],
                "passed": ok,
            })
    results.sort(key=lambda r: -abs(r["t"]))
    print("  %-16s %-3s %10s %7s %10s %7s %6s %9s %9s %s" % (
        "field", "dir", "raw bps", "t", "clip bps", "clip t", "ret", "1st half", "2nd half", ""))
    for r in results:
        print("  %-16s %-3s %10.2f %7.2f %10.2f %7.2f %5.0f%% %9.2f %9.2f %s" % (
            r["field"], r["sign"], r["raw_bps"], r["t"], r["clip_bps"], r["clip_t"],
            100 * r["retained"], r["first_bps"], r["second_bps"],
            "PASS" if r["passed"] else ""))
    print()
    passed = [r for r in results if r["passed"]]
    print("  candidates tested: %d   passed all bars: %d" % (len(results), len(passed)))
    if not passed:
        print("  No feature in this panel carries a tradeable cross-sectional edge under")
        print("  the gate. That is a finding about the panel, not about the search.")
    with open("data/research_v4/feature_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("  full results written to data/research_v4/feature_sweep.json")


if __name__ == "__main__":
    main()
