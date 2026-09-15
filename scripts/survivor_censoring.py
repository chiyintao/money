"""Apply the censoring test to the three features that survived the sweep.

The sweep only applied two of the gate's bars -- significance and tail retention. The
third, single-coin dependence, is what killed momentum, and it has to be applied to the
survivors before any of them is believed. Momentum's raw t fell from 3.19 to 1.60 when one
coin was removed; a candidate that does the same is the same trap.

The sweep also tests 34 candidates, so at t>=2 about two rejections of the null are
expected by chance alone. That is why a survivor has to clear the coin test as well: it is
the one bar a lucky draw is least likely to clear by accident.
"""
import json
import math
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread
from scripts.zec_diagnosis import daily_with_coins

rows, by_day = pe.load()
COINS = sorted({r["coin"] for r in rows})

SURVIVORS = [("ret_14d", 1), ("ret_1d", 1), ("taker_7d", 1)]


def contribution(field, sign, top=2, bottom=2, clip=None):
    """Attribute each day's spread to the coins in its legs."""
    daily = daily_with_coins(top=top, bottom=bottom, field=field)
    tally = {}
    for day in daily:
        legs = [(c, v) for c, v in day["legs"]]
        if sign < 0:
            legs = [(c, -v) for c, v in legs]
        for coin, value in legs:
            tally[coin] = tally.get(coin, 0.0) + value
    return tally


def main():
    print("SINGLE-COIN CENSORING TEST ON THE SURVIVORS")
    print("=" * 88)
    verdicts = {}
    for field, sign in SURVIVORS:
        tally = contribution(field, sign)
        biggest = max(tally, key=lambda c: tally[c])
        base = pe.stats(spread(field, sign=sign))
        without = pe.stats(spread(field, sign=sign, exclude=(biggest,)))
        clip = pe.stats(spread(field, sign=sign, clip=0.10))
        clip_without = pe.stats(spread(field, sign=sign, clip=0.10, exclude=(biggest,)))
        ok = without["t"] >= 2.0 and clip_without["t"] >= 2.0
        verdicts[field] = ok
        print("  %-12s" % field)
        print("     biggest contributor : %-10s (%.0f%% of the spread)"
              % (biggest, 100 * tally[biggest] / sum(tally.values())))
        print("     raw                 : %+8.2f bps  t=%5.2f" % (base["mean_bps"], base["t"]))
        print("     clipped +/-10%%      : %+8.2f bps  t=%5.2f" % (clip["mean_bps"], clip["t"]))
        print("     without %-10s : %+8.2f bps  t=%5.2f" % (biggest, without["mean_bps"], without["t"]))
        print("     without + clipped   : %+8.2f bps  t=%5.2f" % (
            clip_without["mean_bps"], clip_without["t"]))
        print("     VERDICT: %s" % ("survives" if ok else "REJECTED -- single-coin dependence"))
        print()
    print("  summary: %s" % ", ".join(
        "%s=%s" % (f, "pass" if v else "fail") for f, v in verdicts.items()))
    with open("data/research_v4/survivor_censoring.json", "w", encoding="utf-8") as fh:
        json.dump(verdicts, fh, indent=1)


if __name__ == "__main__":
    main()
