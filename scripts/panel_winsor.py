"""The mechanism, stated plainly, and the test that separates it from an edge.

ZECUSDT went from 18.28 to a peak of 1243.68 inside this panel -- 68x. Its own-return
autocorrelation is +0.061, the highest of any coin here. So the strategy's ZEC profit is
not a mystery and not a data error: a coin that trends for months is exactly what a
momentum rank is built to ride.

The question a trader has to answer is different from the question a statistician does.
Riding a 68x trend is profitable in hindsight and tells you nothing about the next coin,
because the rank has no way to know in advance which of fourteen coins will be the one
that runs. The test for that is to remove the TREND, not the coin: winsorise each coin's
forward return to a plausible daily move, and see whether the spread survives. If the edge
lives in the tails of one coin's trend, winsorising kills it and the coin-dropout result
was the honest one.
"""
import math
import statistics
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe

rows, by_day = pe.load()
days = sorted(by_day)


def run_winsorised(clip, top=2, bottom=2, field="ret_7d"):
    """Same rank as before, but the forward return of each leg is clipped to +/-clip.

    Clipping is applied to the OUTCOME, never the signal. The signal still sees the full
    past return -- that is what a live system would see -- so this measures how much of the
    profit came from a handful of outsized forward moves rather than from being right on
    ordinary days.
    """
    out = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])]
        if len(rr) < 8:
            continue
        # The legs are chosen from the RAW signal and only then is the outcome clipped.
        # An earlier version clipped before ranking, which let the selection see the
        # clipped future -- look-ahead. It happened to give the same headline number as
        # the correct version here, which is luck rather than validation, and it would not
        # have been safe on a signal that responded to tails.
        scores = pe.signal(rr, 7, field)
        if scores is None:
            continue
        order = sorted(range(len(rr)), key=lambda i: scores[i])
        longs = order[-top:]
        shorts = order[:bottom]

        def clipped(i):
            return max(-clip, min(clip, rr[i]["fwd_1d"]))

        out.append(sum(clipped(i) for i in longs) / top
                   - sum(clipped(i) for i in shorts) / bottom)
    return out


def main():
    print("WINSORISED FORWARD RETURNS -- how much lives in the tails?")
    print("=" * 72)
    base = pe.stats(pe.run(by_day)[0])
    print("  unclipped          : %+8.2f bps  t=%5.2f" % (base["mean_bps"], base["t"]))
    for clip in (0.15, 0.10, 0.07, 0.05, 0.03):
        s = pe.stats(run_winsorised(clip))
        keep = 100 * s["mean_bps"] / base["mean_bps"]
        print("  clipped to +/-%4.0f%%: %+8.2f bps  t=%5.2f   (%.0f%% of the raw spread)"
              % (clip * 100, s["mean_bps"], s["t"], keep))
    print()
    print("  Reading: if the spread collapses toward zero as the clip tightens, the profit")
    print("  was carried by a few outsized days, and no sizing rule or cost assumption can")
    print("  make that repeatable. If it holds most of its value at a tight clip, the rank")
    print("  is picking direction on ordinary days, which is what an edge looks like.")


if __name__ == "__main__":
    main()
