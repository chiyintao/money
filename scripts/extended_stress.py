"""Stress the extended-panel close_pos result before believing it.

The shape is now what a real effect looks like: t is flat around 2.2-2.4 from a 120-day
window all the way to 2154, rather than plateauing near 1 and jumping at the end. That is
the opposite of the 894-day failure.

But five of the sixteen configurations were tested, the panel was chosen after seeing that
the 14-coin basket could not be extended, and ZEC was excluded by hand. Each of those is a
place a false positive could enter, so each is checked here rather than assumed harmless.
"""
import sys
sys.path.insert(0, ".")
from scripts.extended_close_pos import load, spread, stats

_, by_day = load()


def main():
    print("STRESS TESTS ON THE EXTENDED close_pos RESULT")
    print("=" * 84)

    # 1. Is the result an artifact of excluding ZEC? On the 894-day panel they survived
    #    only because they excluded it; the question is whether ZEC is needed here.
    print("  1. WITH ZEC REINSTATED")
    for clip in (None, 0.10):
        v = spread(by_day, "close_pos", legs=4, clip=clip, exclude=())
        s = stats(v)
        cut = len(v) // 2
        a, b = stats(v[:cut]), stats(v[cut:])
        print("     clip %-5s %+8.2f bps  t=%5.2f   halves %+.2f / %+.2f" % (
            "none" if clip is None else "%.0f%%" % (clip * 100),
            s["mean_bps"], s["t"], a["mean_bps"], b["mean_bps"]))
    print()

    # 2. Every coin dropped in turn. The 894-day result died on this test; if this one
    #    survives it, the effect is not a single series.
    print("  2. LEAVE-ONE-OUT (4 legs, clipped)")
    base = stats(spread(by_day, "close_pos", legs=4, clip=0.10, exclude=("ZECUSDT",)))
    print("     all coins (ex ZEC): %+8.2f bps  t=%5.2f" % (base["mean_bps"], base["t"]))
    coins = sorted({r["coin"] for r in by_day[sorted(by_day)[0]]})
    worst = None
    for coin in coins:
        if coin == "ZECUSDT":
            continue
        v = spread(by_day, "close_pos", legs=4, clip=0.10, exclude=("ZECUSDT", coin))
        s = stats(v)
        mark = ""
        if s["t"] < 2.0:
            mark = "  <-- drops below 2"
        print("     without %-10s %+8.2f bps  t=%5.2f%s" % (coin, s["mean_bps"], s["t"], mark))
        if worst is None or s["t"] < worst[1]:
            worst = (coin, s["t"])
    print()

    # 3. A shuffled control on the same panel, to confirm the machinery reports nothing
    #    when there is nothing.
    import random
    print("  3. NULL CONTROL (close_pos shuffled within each day)")
    for trial in range(3):
        rng = random.Random(1000 + trial)
        values = []
        for day in sorted(by_day):
            rr = [x for x in by_day[day]
                  if x.get("fwd_1d") is not None and x["coin"] != "ZECUSDT"]
            if len(rr) < 6:
                continue
            scores = [r.get("close_pos") for r in rr]
            if any(s is None for s in scores):
                continue
            rng.shuffle(scores)
            order = sorted(range(len(rr)), key=lambda i: scores[i])
            f = lambda i: max(-0.10, min(0.10, rr[i]["fwd_1d"]))
            values.append(sum(f(i) for i in order[-4:]) / 4 - sum(f(i) for i in order[:4]) / 4)
        s = stats(values)
        print("     trial %d: %+8.2f bps  t=%5.2f" % (trial, s["mean_bps"], s["t"]))
    print()
    if worst:
        print("  closest call on leave-one-out: %s -> t=%.2f" % worst)


if __name__ == "__main__":
    main()
