"""Is the momentum edge a strategy or one coin's story?

The full-panel test gave +49.93 bps/day at t=3.19, but removing ZECUSDT alone cut it to
+22.68 at t=1.60. Half the result living in one of fourteen coins is the signature of a
single stretched move being caught by a rank, not of an edge that repeats.

This asks where ZEC's contribution actually sits: is it spread evenly across its 499
appearances, or concentrated in one stretch of days? Those have opposite implications.
Evenly spread means the coin is unusually trendy and the strategy is picking that up.
Concentrated means the average is being carried by an event.
"""
import collections
import math
import statistics
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe

rows, by_day = pe.load()
days = sorted(by_day)


def daily_with_coins(top=2, bottom=2, field="ret_7d"):
    """Per-day spread plus which coins sat in each leg, so contribution is attributable."""
    out = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])]
        if len(rr) < 8:
            continue
        scores = pe.signal(rr, 7, field)
        if scores is None:
            continue
        order = sorted(range(len(rr)), key=lambda i: scores[i])
        longs = order[-top:]
        shorts = order[:bottom]
        out.append({
            "day": day,
            "spread": (sum(rr[i]["fwd_1d"] for i in longs) / top
                       - sum(rr[i]["fwd_1d"] for i in shorts) / bottom),
            "longs": [rr[i]["coin"] for i in longs],
            "shorts": [rr[i]["coin"] for i in shorts],
            # Contribution of each leg to the spread, so a coin's share can be summed.
            "legs": ([(rr[i]["coin"], rr[i]["fwd_1d"] / top) for i in longs]
                     + [(rr[i]["coin"], -rr[i]["fwd_1d"] / bottom) for i in shorts]),
        })
    return out


def main():
    daily = daily_with_coins()
    print("WHERE ZEC'S CONTRIBUTION SITS")
    print("=" * 72)
    total = sum(d["spread"] for d in daily)
    n = len(daily)
    print("  days: %d   total spread summed: %+.4f   mean %+.2f bps"
          % (n, total, 10000 * total / n))
    print()

    # ZEC's own contribution, day by day.
    zec_days = [(d["day"], sum(v for c, v in d["legs"] if c == "ZECUSDT"))
                for d in daily]
    zec_days = [(day, v) for day, v in zec_days if v != 0.0]
    print("  days on which ZECUSDT sat in a leg: %d of %d" % (len(zec_days), n))
    if zec_days:
        contrib = [v for _, v in zec_days]
        print("  ZEC total contribution : %+.4f  (%.1f%% of the total spread)"
              % (sum(contrib), 100 * sum(contrib) / total))
        contrib_sorted = sorted(contrib, reverse=True)
        print("  its best single day    : %+.4f" % contrib_sorted[0])
        print("  its best 5 days        : %+.4f  (%.1f%% of the total spread)"
              % (sum(contrib_sorted[:5]), 100 * sum(contrib_sorted[:5]) / total))
        print()
        # Concentration: what share of the total comes from ZEC's top few days?
        top5_share = 100 * sum(contrib_sorted[:5]) / total
        if top5_share > 60:
            print("  VERDICT: the edge is carried by a handful of ZEC days. This is an")
            print("           event, not a repeating edge.")
        else:
            print("  VERDICT: ZEC's contribution is spread across many days.")
        print()
        # When were those days?
        best = sorted(zec_days, key=lambda kv: -kv[1])[:8]
        print("  ZEC's eight most profitable days (panel day index):")
        for day, v in best:
            print("     day %5d   %+9.4f" % (day, v))


if __name__ == "__main__":
    main()
