"""Would close_pos have paid after costs, and is it distinct from momentum?

Two things decide whether this is worth building, and neither has been checked.

First, cost. A daily-rebalanced cross-section pays the round trip on whatever it turns
over. Every previous candidate was quoted gross, and the one that reached production lost
money. The size of this edge -- 13 to 20 bps gross -- is the same order as the fee.

Second, redundancy. close_pos correlates with where the day closed, and a day that closes
high has usually gone up. If it is mostly momentum in disguise it inherits momentum's
problems and should be dropped now rather than after it is built.
"""
import math
import statistics
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread

rows, by_day = pe.load()
days = sorted(by_day)
FEE_PER_SIDE_BPS = 4.0   # fee_rate=0.0004 in the live configuration


def legs_and_turnover(field, top, bottom, exclude=("ZECUSDT",)):
    """Mean daily turnover: the share of the four legs that changes each day."""
    previous = set()
    turnover = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])
              and x["coin"] not in exclude]
        if len(rr) < 8:
            continue
        values = [r.get(field) for r in rr]
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        if len(set(values)) < 3:
            continue
        scores = pe.rank(values)
        order = sorted(range(len(rr)), key=lambda i: scores[i])
        current = {rr[i]["coin"] for i in list(order[-top:]) + list(order[:bottom])}
        if previous:
            turnover.append(len(current - previous) / len(current))
        previous = current
    return sum(turnover) / len(turnover) if turnover else 0.0


def correlation(field_a, field_b):
    """Cross-sectional correlation of the two ranks, averaged over days."""
    out = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])]
        if len(rr) < 8:
            continue
        a = [r.get(field_a) for r in rr]
        b = [r.get(field_b) for r in rr]
        if any(v is None or not math.isfinite(v) for v in a + b):
            continue
        ra, rb = pe.rank(a), pe.rank(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
        sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
        if sa and sb:
            out.append(cov / (sa * sb))
    return sum(out) / len(out) if out else float("nan")


def main():
    print("IS close_pos TRADEABLE AFTER COSTS?  (ZECUSDT excluded throughout)")
    print("=" * 84)
    for legs in (2, 3, 4, 5):
        turn = legs_and_turnover("close_pos", legs, legs)
        gross = pe.stats(spread("close_pos", sign=1, top=legs, bottom=legs,
                                clip=0.10, exclude=("ZECUSDT",)))
        # Cost is charged on the legs that actually change. Each changed leg pays a full
        # round trip on 1/(2*legs) of the book, because the spread is normalised to half
        # the gross exposure.
        cost = turn * 2 * FEE_PER_SIDE_BPS * 2 / (2 * legs) * legs
        print("  %d legs: gross %+7.2f bps  turnover %4.1f%%  cost %5.2f bps  net %+7.2f bps"
              % (legs, gross["mean_bps"], 100 * turn, cost, gross["mean_bps"] - cost))
    print()
    print("REDUNDANCY WITH MOMENTUM")
    for f in ("ret_1d", "ret_7d", "ret_14d"):
        print("  rank correlation close_pos vs %-8s : %+.3f" % (f, correlation("close_pos", f)))
    print()
    print("  A low correlation means this is not momentum wearing a different name.")


if __name__ == "__main__":
    main()
