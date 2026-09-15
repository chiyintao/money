"""Rolling out-of-sample check on close_pos.

Everything so far is a single number over one 894-day stretch. The momentum result also
looked fine as a single number and fell apart the moment it was sliced. So this slices
close_pos the same way, on the two axes that exposed the previous failure: time, and the
identity of the coins carrying the result.

The signal is fixed here -- rank on close_pos, hold one day, 4 legs, ZEC excluded, tails
clipped -- so nothing is chosen per window and the comparison across windows is fair.
"""
import json
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread, days as all_days

rows, by_day = pe.load()
DAYS = sorted(by_day)

CONFIG = dict(field="close_pos", sign=1, top=4, bottom=4, clip=0.10)


def window_values(start, end):
    """Daily spreads over a day-index range, using the same fixed rule."""
    from scripts import feature_sweep as fs
    out = []
    for day in DAYS:
        if day < start or day > end:
            continue
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None
              and x["coin"] != "ZECUSDT"]
        if len(rr) < 8:
            continue
        values = [r.get(CONFIG["field"]) for r in rr]
        if any(v is None for v in values):
            continue
        if len(set(values)) < 3:
            continue
        scores = pe.rank(values)
        order = sorted(range(len(rr)), key=lambda i: scores[i])

        def clip(i):
            v = rr[i]["fwd_1d"]
            return max(-0.10, min(0.10, v))

        out.append(sum(clip(i) for i in order[-4:]) / 4
                   - sum(clip(i) for i in order[:4]) / 4)
    return out


def main():
    values = window_values(DAYS[0], DAYS[-1])
    print("ROLLING SUB-PERIOD TEST: close_pos, 4 legs, ZEC excluded, clipped")
    print("=" * 84)
    print("  full period: %+8.2f bps  t=%5.2f  n=%d"
          % (pe.stats(values)["mean_bps"], pe.stats(values)["t"], len(values)))
    print()
    print("  every 120-day window:")
    win, step = 120, 60
    counts = {"positive": 0, "significant": 0, "total": 0}
    for start in range(0, len(values) - win, step):
        part = values[start:start + win]
        s = pe.stats(part)
        counts["total"] += 1
        if s["mean"] > 0:
            counts["positive"] += 1
        if s["t"] >= 2.0:
            counts["significant"] += 1
        print("     days %3d-%3d  %+8.2f bps  t=%5.2f" % (
            start, start + win, s["mean_bps"], s["t"]))
    print()
    print("  windows positive        : %d of %d" % (counts["positive"], counts["total"]))
    print("  windows with t >= 2     : %d of %d" % (counts["significant"], counts["total"]))
    print()
    cut = len(values) // 2
    for label, part in (("first half", values[:cut]), ("second half", values[cut:])):
        s = pe.stats(part)
        print("  %-12s %+8.2f bps  t=%5.2f" % (label, s["mean_bps"], s["t"]))


if __name__ == "__main__":
    main()
