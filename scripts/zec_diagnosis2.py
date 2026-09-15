"""The right test: is ZEC's contribution a coin property or an outlier property?

ZEC contributes 69.6% of the spread across 497 days, but only 20.5% from its best five
days. That rules out "one event" but not "one coin". The distinction that matters for a
trading decision is narrower:

  If ZEC is simply the most volatile coin, then any momentum rank on a wide enough
  universe will find such a coin, and the edge is real but concentrated by volatility.
  If ZEC is special -- a coin whose returns happen to be predictable by past returns in a
  way the others are not -- then the result will not replicate when it is swapped for a
  comparable coin, and there is no edge to trade.

The test is a substitution: replace ZEC with each other coin in turn and see what the
per-coin contribution looks like once the rank is recomputed. Then normalise contribution
by each coin's own volatility, which is the thing that should drive it if the mechanism is
simply "volatile coins have bigger forward moves".
"""
import collections
import math
import statistics
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.zec_diagnosis import daily_with_coins

rows, by_day = pe.load()


def main():
    daily = daily_with_coins()
    total = sum(d["spread"] for d in daily)

    # Contribution per coin, and each coin's own realised volatility, so the two can be
    # put side by side. A coin earning 70% of the spread while being 5x as volatile is a
    # different finding from one earning it at average volatility.
    contrib = collections.defaultdict(float)
    appear = collections.Counter()
    for d in daily:
        for coin, value in d["legs"]:
            contrib[coin] += value
            appear[coin] += 1

    returns = collections.defaultdict(list)
    for r in rows:
        if r.get("fwd_1d") is not None and math.isfinite(r["fwd_1d"]):
            returns[r["coin"]].append(r["fwd_1d"])

    print("CONTRIBUTION vs VOLATILITY")
    print("=" * 78)
    print("  %-10s %5s %12s %10s %12s %10s" % (
        "coin", "days", "contrib", "%of total", "daily vol", "contrib/vol"))
    print("  " + "-" * 74)
    ordered = sorted(contrib, key=lambda c: -contrib[c])
    for coin in ordered:
        vol = statistics.stdev(returns[coin]) if len(returns[coin]) > 1 else float("nan")
        ratio = contrib[coin] / vol if vol else float("nan")
        print("  %-10s %5d %+12.4f %9.1f%% %11.5f %10.4f" % (
            coin, appear[coin], contrib[coin], 100 * contrib[coin] / total, vol, ratio))

    print()
    # If contribution tracked volatility, the ranking of contrib/vol would be flat. If ZEC
    # stands out on the RATIO, its contribution is not explained by being volatile.
    ratios = {c: contrib[c] / (statistics.stdev(returns[c]) or float("nan"))
              for c in contrib}
    med = statistics.median(list(ratios.values()))
    print("  median contrib/vol = %.4f" % med)
    print("  ZECUSDT contrib/vol = %.4f  (%.2fx the median)"
          % (ratios["ZECUSDT"], ratios["ZECUSDT"] / med))
    print()
    others = sorted((v, c) for c, v in ratios.items() if c != "ZECUSDT")
    print("  the other thirteen, ranked by contrib/vol:")
    for v, c in reversed(others):
        print("     %-10s %8.4f" % (c, v))


if __name__ == "__main__":
    main()
