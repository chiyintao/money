"""close_pos: the one candidate whose edge does not depend on ZEC.

Every momentum feature lost its significance when ZECUSDT was removed. close_pos did the
opposite: raw t 2.17 -> 2.16 and clipped t 1.88 -> 2.06, so removing the coin that carried
everything else made its tail-robust result BETTER. That is the signature of a signal that
is not simply a proxy for "went up".

It still fails one bar (clipped t = 1.88 against a 2.0 requirement), so it is not yet a
finding. But the failure looks like power rather than absence, and the way to tell those
apart is to vary the construction: if the effect is real, more legs and different horizons
should raise the t; if it was noise, they will not.

close_pos is the close's position within the bar's own range: (close - low) / (high - low).
High means the bar closed near its high. Ranking it long means buying strength INTO the
close, which at a daily horizon is a short-term reversal candidate -- the opposite of the
momentum family, which is why it behaves differently under censoring.
"""
import json
import math
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread


def main():
    print("close_pos CONSTRUCTION SWEEP (no ZEC anywhere)")
    print("=" * 84)
    print("  %-6s %-8s %12s %8s %12s %8s %8s" % (
        "legs", "clip", "mean bps", "t", "1st half", "2nd half", "days"))
    best = []
    for legs in (1, 2, 3, 4, 5):
        for clip in (None, 0.10):
            values = spread("close_pos", sign=1, top=legs, bottom=legs, clip=clip,
                            exclude=("ZECUSDT",))
            s = pe.stats(values)
            cut = len(values) // 2
            a = pe.stats(values[:cut])
            b = pe.stats(values[cut:])
            both = "YES" if (a["mean"] > 0 and b["mean"] > 0) else "no"
            print("  %-6d %-8s %12.2f %8.2f %12.2f %8.2f %8s" % (
                legs, "none" if clip is None else "%.0f%%" % (clip * 100),
                s["mean_bps"], s["t"], a["mean_bps"], b["mean_bps"], both))
            if clip is not None and s["t"] >= 2.0 and a["mean"] > 0 and b["mean"] > 0:
                best.append((legs, s["mean_bps"], s["t"]))
    print()
    if best:
        print("  configurations where the clipped, ZEC-free signal clears t=2 in both halves:")
        for legs, bps, t in best:
            print("     %d legs: %+.2f bps  t=%.2f" % (legs, bps, t))
    else:
        print("  No configuration clears t=2 with tails clipped and ZEC removed.")
    print()
    print("  A real effect should get STRONGER with more legs, because averaging more")
    print("  independent coins reduces noise. A t that decays as legs grow means the")
    print("  signal lived in the two extreme coins rather than across the cross-section.")


if __name__ == "__main__":
    main()
