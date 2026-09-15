"""A gate for proposed signals: would this edge survive the two tests that killed the last one?

The momentum result passed every robustness check that had been run on it -- split halves,
leg counts, lookbacks, autocorrelation, survivorship. It failed two that had not been run:
clipping the forward return, and removing the single largest contributor. Both are cheap
and both are decisive, so they belong in front of any future signal rather than in a
post-mortem of a losing one.

Usage:
    python scripts/edge_gate.py                       # the momentum baseline
    python scripts/edge_gate.py --field ret_14d --top 3 --bottom 3
"""
import argparse
import collections
import math
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe

# A signal has to clear all three. They are deliberately strict: the cost of rejecting a
# real edge is another week of research, and the cost of accepting a false one is capital.
MIN_T_RAW = 2.0
MIN_T_CLIPPED = 2.0
MIN_RETAINED = 0.50     # share of the raw spread that survives a +/-10% clip
MIN_T_NO_BEST_COIN = 2.0


def spread(days_by_coin, top, bottom, field, clip=None, exclude=()):
    out = []
    for day in sorted(days_by_coin):
        rows = [r for r in days_by_coin[day]
                if r.get("fwd_1d") is not None and math.isfinite(r["fwd_1d"])
                and r["coin"] not in exclude]
        if len(rows) < 8:
            continue
        scores = pe.signal(rows, 7, field)
        if scores is None:
            continue
        order = sorted(range(len(rows)), key=lambda i: scores[i])

        def value(i):
            v = rows[i]["fwd_1d"]
            return max(-clip, min(clip, v)) if clip else v

        out.append(sum(value(i) for i in order[-top:]) / top
                   - sum(value(i) for i in order[:bottom]) / bottom)
    return out


def contribution_by_coin(results):
    """Attribute each day's spread back to the coins that produced it."""
    tally = collections.defaultdict(float)
    for day in results:
        for coin, value in day["legs"]:
            tally[coin] += value
    return tally


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gate a candidate signal.")
    parser.add_argument("--field", default="ret_7d")
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--clip", type=float, default=0.10)
    args = parser.parse_args(argv)

    rows, by_day = pe.load()

    from scripts.zec_diagnosis import daily_with_coins
    daily = daily_with_coins(top=args.top, bottom=args.bottom, field=args.field)
    raw = pe.stats([d["spread"] for d in daily])
    # Per-trade clipping, via the same helper the gate's sibling sweep uses. Clipping the
    # daily spread instead would understate the tail dependence badly -- it measured 82%
    # retention on a signal whose true figure is 50%.
    clipped = pe.stats(spread(by_day, args.top, args.bottom, args.field, args.clip))

    tally = contribution_by_coin(daily)
    biggest = max(tally, key=lambda c: tally[c])
    without = pe.stats(spread(by_day, args.top, args.bottom, args.field, None, (biggest,)))

    retained = clipped["mean_bps"] / raw["mean_bps"] if raw["mean_bps"] else 0.0

    print("EDGE GATE  field=%s  legs=%dx%d" % (args.field, args.top, args.bottom))
    print("=" * 68)
    checks = [
        ("raw significance", raw["mean_bps"], raw["t"], MIN_T_RAW,
         "mean %+.2f bps  t=%.2f" % (raw["mean_bps"], raw["t"])),
        ("tail robustness", clipped["mean_bps"], clipped["t"], MIN_T_CLIPPED,
         "clipped +/-%.0f%%: %+.2f bps  t=%.2f  retained %.0f%%"
         % (args.clip * 100, clipped["mean_bps"], clipped["t"], 100 * retained)),
        ("single-coin dependence", without["mean_bps"], without["t"], MIN_T_NO_BEST_COIN,
         "without %s: %+.2f bps  t=%.2f" % (biggest, without["mean_bps"], without["t"])),
    ]
    passed = True
    for name, _, t, floor, detail in checks:
        ok = t >= floor
        passed = passed and ok
        print("  [%s] %-24s %s" % ("PASS" if ok else "FAIL", name, detail))
    print("  [%s] %-24s retained %.0f%% (need %.0f%%)"
          % ("PASS" if retained >= MIN_RETAINED else "FAIL", "tail retention",
             100 * retained, 100 * MIN_RETAINED))
    passed = passed and retained >= MIN_RETAINED
    print()
    print("VERDICT: %s" % ("passes the gate" if passed else "REJECTED"))
    if not passed:
        print("  A signal that fails here has no tradeable edge at this sample size, and no")
        print("  sizing rule, cost assumption or model architecture downstream can create")
        print("  one. Fix the signal, not the layers above it.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
