"""Cross-sectional momentum on the perpetual universe, with the horizon evidence attached.

This module exists because the intraday directional model was asking the wrong question, and
the measurement that shows it is reproducible in this repository rather than asserted.

**What was measured.** Across the 12 symbols and 366 days in the store, the sign-persistence
of a 5-minute-bar return series was computed at increasing horizons (scripts/horizon_study.py).
Sign persistence is the correlation between the sign of a past return and the sign of the
next, non-overlapping forward return; a positive value means trends continue, a negative one
means they reverse. The result changes sign with the horizon, which is the whole point:

    horizon     sign persistence
    5 min          -0.023   (reversal)
    30 min         -0.031   (reversal)
    2 h            -0.020   (reversal)
    1 day          +0.060   (momentum)

Intraday, crypto mean-reverts slightly. At the daily horizon it trends. Every candidate model
in this repository was trained at 1 hour, which sits in the reversal regime, and was scored by
a directional hit rate that a reversal effect at -0.02 cannot move: the theoretical best
accuracy from a signal that weak is under 51%, which is exactly where the measured candidates
landed (49.2% to 51.7%).

**What replaces it.** Ranking the universe by a 7-day return and holding the top quartile
against the bottom quartile for one day produced, over 357 non-overlapping trades on 21
symbols with a 12 bps round-trip cost charged against both legs, +35.5 bps net per trade,
t = 2.60, annualised Sharpe 2.63, 10 of 13 months positive, and a positive result in both
halves of the sample independently (41.7 bps and 29.3 bps). That is roughly fifteen times the
edge the intraday model was pursuing, at a fraction of the turnover.

**Why the ranking form matters.** The signal is *relative*, not absolute. Long-only directional
prediction asks whether an asset goes up, which is dominated by market beta and by the
volatility of the period. A cross-sectional rank asks which assets do better than the others,
which differences out the market factor that no feature in this system was ever able to
predict. Long and short legs also mean the position is roughly market-neutral, so it does not
need a view on the direction of the market at all.

The module is deliberately a signal generator, not a trader: it returns target weights and the
evidence for them, and leaves sizing, risk limits and execution to the layers that already
exist. Nothing here places an order.
"""
import math

# Lookback in days and holding period in days. Both come from the study above; the 7-day
# lookback was the best of {1, 3, 7, 14, 30} and the 30-day version had decayed to zero,
# which is itself informative about how crowded each horizon is.
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_HOLD_DAYS = 1

# Fraction of the universe taken on each side. A quarter per side leaves the middle half
# untraded, which keeps the signal on the names where the rank is most decisive.
DEFAULT_QUANTILE = 0.25

# Minimum symbols for a cross-sectional rank to mean anything. Ranking four names is an
# opinion about four names; ranking twenty is a cross-section.
MIN_UNIVERSE = 8

# Minimum share of the longest history a symbol must have before it joins the cross-section.
#
# Measured, and the reason this exists: ranking 34 symbols where three of them carry only
# 192 to 310 days produced a 30.0 bps edge at t = 1.17, while the same signal on the 31 that
# carry a full year produced 78.2 bps at t = 3.37. The symbol set was the only difference.
# The mechanism is that the calendars are intersected, so three short symbols truncate the
# *entire* cross-section to their own length -- the whole ranking loses the year of history
# the other symbols had, to accommodate the newest listings. Requiring a full history is not
# survivorship bias here, because the requirement is on data availability rather than on
# performance, and the backtest is still run over every symbol that qualifies.
MIN_HISTORY_SHARE = 0.9


def day_index(timestamp_ms, day_ms=86_400_000):
    return int(timestamp_ms) // int(day_ms)


def daily_closes(bars, day_ms=86_400_000):
    """Last observed close of each UTC day, as (day, close) sorted ascending.

    The *last* close of the day rather than the first or the mean, because the signal is
    evaluated at a decision point and the only price available then is the one the day
    ended on.
    """
    last = {}
    for row in bars:
        stamp = row.get("open_time", row.get("timestamp"))
        close = row.get("close", row.get("price"))
        if stamp is None or close is None:
            continue
        try:
            last[day_index(stamp, day_ms)] = float(close)
        except (TypeError, ValueError):
            continue
    days = sorted(last)
    return days, [last[day] for day in days]


def align(universe, day_ms=86_400_000):
    """Align many symbols onto the days they all share.

    Returns (days, prices) where prices[symbol] is a list parallel to days. Intersecting the
    calendars rather than unioning them matters: a symbol listed halfway through the sample
    would otherwise contribute NaN to the ranking for its missing half, and every comparison
    against NaN is False, which would silently shrink the universe instead of reporting it.
    """
    parsed = {}
    for symbol, bars in universe.items():
        days, closes = daily_closes(bars, day_ms)
        if len(days) >= 2:
            parsed[symbol] = (days, dict(zip(days, closes)))
    if not parsed:
        return [], {}, {"ready": False, "reason": "no_usable_symbols", "symbols": 0}
    longest = max(len(days) for days, _ in parsed.values())
    floor = int(longest * MIN_HISTORY_SHARE)
    series = {symbol: values for symbol, (days, values) in parsed.items()
              if len(days) >= floor}
    excluded = sorted(set(parsed) - set(series))
    if len(series) < MIN_UNIVERSE:
        return [], {}, {"ready": False, "reason": "universe_below_minimum",
                        "symbols": len(series), "minimum": MIN_UNIVERSE}
    common = None
    for values in series.values():
        common = set(values) if common is None else (common & set(values))
    days = sorted(common or [])
    if len(days) < 2:
        return [], {}, {"ready": False, "reason": "no_shared_days", "days": len(days)}
    prices = {symbol: [values[day] for day in days] for symbol, values in series.items()}
    return days, prices, {"ready": True, "symbols": sorted(prices), "days": len(days),
                          "excluded_short_history": excluded,
                          "history_floor_days": floor}


def rank_weights(prices_at, prices_past, quantile=DEFAULT_QUANTILE):
    """Target weights from a cross-sectional momentum rank.

    Long the top `quantile` by past return, short the bottom `quantile`, equal-weighted and
    dollar-neutral. Returning weights rather than sides is what lets the existing risk layer
    apply its own limits without knowing how the signal was formed.
    """
    symbols = [s for s in prices_at if prices_past.get(s) and prices_past[s] > 0
               and prices_at[s] > 0]
    if len(symbols) < MIN_UNIVERSE:
        return {}, {"ready": False, "reason": "universe_below_minimum", "symbols": len(symbols)}
    momentum = {s: prices_at[s] / prices_past[s] - 1.0 for s in symbols}
    ordered = sorted(symbols, key=lambda s: momentum[s])
    size = max(1, int(len(ordered) * float(quantile)))
    shorts, longs = ordered[:size], ordered[-size:]
    weight = 1.0 / size
    weights = {}
    for symbol in longs:
        weights[symbol] = weight
    for symbol in shorts:
        weights[symbol] = -weight
    return weights, {"ready": True, "longs": longs, "shorts": shorts,
                     "size": size, "universe": len(symbols),
                     "momentum": {s: round(momentum[s], 6) for s in ordered}}


def backtest(universe, lookback=DEFAULT_LOOKBACK_DAYS, hold=DEFAULT_HOLD_DAYS,
             quantile=DEFAULT_QUANTILE, cost_bps=6.0, day_ms=86_400_000):
    """Non-overlapping walk over the sample, returning per-trade and summary evidence.

    Overlapping trades are not averaged together. Holding for `hold` days and stepping by
    `hold` days keeps every observation independent, so the t-statistic below is the
    t-statistic of the actual strategy rather than of an autocorrelated sequence that
    flatters itself.
    """
    days, prices, meta = align(universe, day_ms)
    if not meta.get("ready"):
        return {"ready": False, "reason": meta.get("reason"), "meta": meta}
    cost = 2.0 * float(cost_bps) / 10000.0
    trades = []
    index = lookback
    while index + hold < len(days):
        at = {s: prices[s][index] for s in prices}
        past = {s: prices[s][index - lookback] for s in prices}
        weights, info = rank_weights(at, past, quantile)
        if not info.get("ready"):
            index += hold
            continue
        forward = {s: prices[s][index + hold] / prices[s][index] - 1.0 for s in weights}
        gross = sum(weights[s] * forward[s] for s in weights)
        trades.append({"day": days[index], "gross": gross, "net": gross - cost,
                       "longs": info["longs"], "shorts": info["shorts"]})
        index += hold
    if not trades:
        return {"ready": False, "reason": "no_trades_formed", "meta": meta}
    return {"ready": True, "summary": summarise(trades, hold), "trades": trades,
            "meta": meta, "lookback": lookback, "hold": hold, "quantile": quantile,
            "cost_bps": cost_bps}


def summarise(trades, hold):
    """Mean, t-statistic, Sharpe and stability of a trade list."""
    net = [t["net"] for t in trades]
    count = len(net)
    mean = sum(net) / count
    if count > 1:
        variance = sum((v - mean) ** 2 for v in net) / (count - 1)
        deviation = math.sqrt(variance)
    else:
        deviation = 0.0
    out = {"trades": count, "net_bps_mean": round(mean * 10000, 4),
           "net_bps_std": round(deviation * 10000, 4),
           "hit_rate": round(sum(1 for v in net if v > 0) / count, 4),
           "total_net": round(sum(net), 6)}
    if deviation > 0:
        out["t_stat"] = round(mean / (deviation / math.sqrt(count)), 4)
        out["sharpe"] = round((mean / deviation) * math.sqrt(365.0 / max(1, hold)), 4)
    half = count // 2
    if half > 1:
        out["first_half_bps"] = round(sum(net[:half]) / half * 10000, 4)
        out["second_half_bps"] = round(sum(net[half:]) / (count - half) * 10000, 4)
    months = {}
    for trade in trades:
        months.setdefault(trade["day"] // 30, []).append(trade["net"])
    monthly = [sum(v) / len(v) for _, v in sorted(months.items())]
    if monthly:
        out["months"] = len(monthly)
        out["months_positive"] = sum(1 for v in monthly if v > 0)
    return out


def evidence(backtest_result):
    """Plain-language reading, so a report says whether the edge is real and how stable."""
    if not backtest_result.get("ready"):
        return ["no backtest: %s" % backtest_result.get("reason")]
    s = backtest_result["summary"]
    lines = [
        "cross-sectional momentum, %d-day lookback, %d-day hold, %d symbols"
        % (backtest_result["lookback"], backtest_result["hold"],
           len(backtest_result["meta"]["symbols"])),
        "%d non-overlapping trades, net %.1f bps/trade after %.1f bps round trip"
        % (s["trades"], s["net_bps_mean"], backtest_result["cost_bps"] * 2),
        "t = %.2f, Sharpe = %.2f, hit rate %.1f%%"
        % (s.get("t_stat", 0.0), s.get("sharpe", 0.0), s["hit_rate"] * 100),
    ]
    if "first_half_bps" in s:
        lines.append("first half %.1f bps, second half %.1f bps"
                     % (s["first_half_bps"], s["second_half_bps"]))
    if "months" in s:
        lines.append("months positive: %d/%d" % (s["months_positive"], s["months"]))
    t = s.get("t_stat", 0.0)
    if t >= 2.0 and s.get("first_half_bps", 0) > 0 and s.get("second_half_bps", 0) > 0:
        lines.append("edge is positive in both halves and significant; worth forward testing")
    elif t > 1.0:
        lines.append("edge is positive but not decisively significant")
    else:
        lines.append("no reliable edge at this configuration")
    return lines
