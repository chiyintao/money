"""Re-test close_pos on 2154 days instead of 894.

The 894-day panel gave t=2.07 in full but only 1.03 over its first 600 days, and no
individual 120-day window cleared 2. The claim under test is that this was a power problem:
close_pos is a small real effect that 894 days could not resolve. 2.4x the history is the
cheapest available way to find out, and it needs no new assumptions -- the same rule, the
same legs, the same clip.

The 11-coin universe is deliberately the long-history one rather than the 14-coin trading
basket, because the two coins that were dropped (SUI, WLD, ENA) do not exist before 2024 and
including them would cap the sample at the very 894 days being tested.

ZECUSDT is excluded throughout, as it was for the censoring test: on this panel it is again
the largest mover, and its presence would let a single series carry the result.
"""
import collections
import json
import math
import sys

PANEL = "data/research_v4/panel_extended.jsonl"


def load(path=PANEL):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    by_day = collections.defaultdict(list)
    for row in rows:
        by_day[row["day"]].append(row)
    return rows, by_day


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def stats(values):
    n = len(values)
    if n < 2:
        return {"n": n, "mean": 0.0, "mean_bps": 0.0, "t": 0.0, "sd": 0.0}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    stderr = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "mean_bps": mean * 10000, "sd": sd,
            "t": mean / stderr if stderr else 0.0}


def spread(by_day, field, legs=4, clip=None, exclude=("ZECUSDT",)):
    out = []
    for day in sorted(by_day):
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and x["coin"] not in exclude]
        if len(rr) < 6:
            continue
        values = [r.get(field) for r in rr]
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        if len(set(values)) < 3:
            continue
        scores = rank(values)
        order = sorted(range(len(rr)), key=lambda i: scores[i])

        def value(i):
            v = rr[i]["fwd_1d"]
            return max(-clip, min(clip, v)) if clip else v

        out.append(sum(value(i) for i in order[-legs:]) / legs
                   - sum(value(i) for i in order[:legs]) / legs)
    return out


def main():
    _, by_day = load()
    print("close_pos ON THE EXTENDED PANEL (2154 days, 11 coins, ZEC excluded)")
    print("=" * 82)
    print("  %-10s %-8s %10s %8s %10s %10s" % ("legs", "clip", "mean bps", "t", "1st half", "2nd half"))
    for legs in (2, 3, 4, 5):
        for clip in (None, 0.10):
            v = spread(by_day, "close_pos", legs=legs, clip=clip)
            s = stats(v)
            cut = len(v) // 2
            a, b = stats(v[:cut]), stats(v[cut:])
            print("  %-10d %-8s %10.2f %8.2f %10.2f %10.2f" % (
                legs, "none" if clip is None else "%.0f%%" % (clip * 100),
                s["mean_bps"], s["t"], a["mean_bps"], b["mean_bps"]))
    print()
    print("  t AS A FUNCTION OF WINDOW LENGTH (4 legs, clipped)")
    v = spread(by_day, "close_pos", legs=4, clip=0.10)
    print("  %-10s %10s %8s %12s" % ("window", "mean bps", "t", "windows"))
    for win in (120, 240, 365, 600, 900, 1200, 1800, len(v)):
        if win > len(v):
            continue
        n_win = len(v) // win
        part = v[:n_win * win]
        s = stats(part)
        print("  %-10d %10.2f %8.2f %12d" % (win, s["mean_bps"], s["t"], n_win))
    print()
    print("  rolling 365-day windows:")
    win = 365
    ts = []
    for start in range(0, len(v) - win + 1, win):
        part = v[start:start + win]
        s = stats(part)
        ts.append(s["t"])
        print("     days %4d-%4d  %+8.2f bps  t=%5.2f" % (
            start, start + win, s["mean_bps"], s["t"]))
    if ts:
        positive = sum(1 for t in ts if t > 0)
        sig = sum(1 for t in ts if t >= 2.0)
        print()
        print("  %d of %d windows positive, %d reach t>=2" % (positive, len(ts), sig))
        print("  median t = %.2f" % sorted(ts)[len(ts) // 2])


if __name__ == "__main__":
    main()
