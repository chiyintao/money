"""Economics of the extended-panel close_pos signal, and its behaviour over time.

A t of 2.4 is a statement about a mean, not about a strategy. Two things decide whether it
is worth building: whether it survives the costs a daily-rebalanced book actually pays, and
whether the windows a live account would live through are tolerable. The previous candidate
failed on the second of those even though it passed the first.
"""
import math
import sys

sys.path.insert(0, ".")
from scripts.extended_close_pos import load, spread, stats

_, by_day = load()
DAYS = sorted(by_day)
FEE_PER_SIDE_BPS = 4.0


def turnover_excluding(legs, exclude=("ZECUSDT",)):
    previous = set()
    rates = []
    for day in DAYS:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and x["coin"] not in exclude]
        if len(rr) < 6:
            continue
        values = [r.get("close_pos") for r in rr]
        if any(v is None for v in values) or len(set(values)) < 3:
            continue
        order = sorted(range(len(rr)), key=lambda i: values[i])
        current = {rr[i]["coin"] for i in list(order[-legs:]) + list(order[:legs])}
        if previous:
            rates.append(len(current - previous) / len(current))
        previous = current
    return sum(rates) / len(rates) if rates else 0.0


def main():
    print("ECONOMICS ON THE EXTENDED PANEL (ZEC excluded, 4 legs, clipped)")
    print("=" * 82)
    turn = turnover_excluding(4)
    gross = stats(spread(by_day, "close_pos", legs=4, clip=0.10))["mean_bps"]
    print("  turnover             : %.1f%% of legs replaced per day" % (100 * turn))
    print("  gross (clipped tails): %+.2f bps/day" % gross)
    print()
    print("  %-10s %10s %12s" % ("fee/side", "cost", "net bps/day"))
    for fee in (2.0, 4.0, 6.0, 10.0, 15.0):
        cost = turn * 2 * fee * 2 / 2
        print("  %-10.1f %10.2f %12.2f" % (fee, cost, gross - cost))
    print()
    print("  annualised at the live 4 bps/side, assuming ~100%% gross exposure:")
    net = gross - turn * 2 * 4.0 * 2 / 2
    print("     %.2f bps/day x 365 = %.1f%% per year on gross exposure" % (net, net * 365 / 10000 * 100))
    print()
    print("  WORST WINDOWS A LIVE ACCOUNT WOULD HAVE LIVED THROUGH")
    v = spread(by_day, "close_pos", legs=4, clip=0.10)
    win = 180
    parts = [v[i:i + win] for i in range(0, len(v) - win + 1, win)]
    stats_list = [stats(p) for p in parts]
    neg = [s for s in stats_list if s["mean"] < 0]
    print("     %d windows of %d days: %d positive, %d negative" % (
        len(stats_list), win, len(stats_list) - len(neg), len(neg)))
    print("     worst window: %+.2f bps  t=%.2f" % (
        min(s["mean_bps"] for s in stats_list), min(s["t"] for s in stats_list)))
    print("     best window : %+.2f bps  t=%.2f" % (
        max(s["mean_bps"] for s in stats_list), max(s["t"] for s in stats_list)))
    # Equity path if the book earned the spread each day. The spread is already a fraction
    # (0.001 = 10 bps) and is already normalised to half the gross exposure, so it is used
    # directly -- an earlier version divided by 10000 a second time and reported a drawdown
    # of -0.0%, which was the arithmetic equivalent of never losing money.
    equity = 1.0
    peak = 1.0
    worst_dd = 0.0
    for s in v:
        equity *= (1 + s)
        peak = max(peak, equity)
        worst_dd = min(worst_dd, equity / peak - 1)
    years = len(v) / 365.0
    cagr = equity ** (1 / years) - 1 if years > 0 else 0.0
    print("     compounding the daily spread over %d days:" % len(v))
    print("       final equity   : %.3fx" % equity)
    print("       CAGR           : %+.1f%%" % (100 * cagr))
    print("       max drawdown   : %.1f%%" % (100 * worst_dd))
    print()
    print("     NOTE: this is the GROSS spread compounded with no fees and no slippage.")
    print("     At 4 bps/side the daily cost is %.2f bps, which reduces the mean from %.2f"
          % (turn * 2 * 4.0 * 2 / 2, gross))
    print("     to %.2f bps. The drawdown figure above is the more robust of the two."
          % (gross - turn * 2 * 4.0 * 2 / 2))


if __name__ == "__main__":
    main()
