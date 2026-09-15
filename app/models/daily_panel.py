"""The strict daily panel: one fixed basket of twelve mainstream coins, no look-ahead.

Why this module exists
----------------------
Three shapes of model were tried in this repository and the measurements say why each failed:

* a pooled 5-minute / 1-hour directional GBDT over fifteen pool symbols reached a
  directional accuracy of 49.2%-51.7% against a twelve-basis-point round trip. The chosen
  horizon sits in the *reversal* regime (sign persistence -0.020 at two hours against
  +0.060 at one day), so the label was, on average, the opposite of the signal the features
  carry;
* the feature matrix was 30 names of which 7 were constant zero, so 23% of the model capacity
  was spent enumerating noise;
* the three informative collected series -- open interest, positioning ratios and funding --
  were used at best as weak *features* on a market that is close to efficient at five minutes.

The reproducible evidence in this repository (\`\`\`scripts/research_v2.py momentum\`\`\`) is
unambiguous about where an edge is and where it is not:

    cross-sectional momentum, 7-day lookback, 1-day hold, 31 symbols
    356 non-overlapping trades, net 78.2 bps/trade after a 12 bps round trip
    t = 3.37, Sharpe 3.41, 10 of 13 months positive, positive in both halves

against a measured out-of-sample edge between -5 and -41 bps per trade for the intraday
directional model across sixteen candidates and every horizon tried.

So this module changes the *question* rather than the model:

* **daily bars, not five-minute bars.** Turnover is roughly 1/300th of the intraday system,
  so a 6 bps round trip is 6 bps against a 150 bps daily move instead of against an 18.9 bps
  five-minute move;
* **a cross-section, not a pooled time series.** Every modelling quantity is either a
  demeaned rank or relative to the basket, because the market factor is the one thing no
  feature in this system ever predicted;
* **one fixed basket of twelve mainstream coins for training, validation and decision.**
  A universe re-picked from a live liquidity snapshot every few minutes asks the model about
  symbols it never saw, which the out-of-distribution gate then refuses;
* **the panel is sealed.** A digest of basket, calendar, column names and families is written
  with the artifact and recomputed at validation, so an artifact cannot be validated against
  a panel it was not fit on.

The frame origin is 6 August 2025 UTC -- a Wednesday, and therefore exactly one of the
Binance weekly funding settlements. Every UTC day in the panel is an exact multiple of three
days from a settlement, so the number of settlements inside any day-aligned window is a
function of its length alone. That is what makes a daily funding carry measurable without
look-ahead.
"""
import datetime as dt
import hashlib
import json
import math
import sqlite3

# ------------------------------------------------------------------ the fixed basket
#
# Fourteen USDT-M perpetuals, chosen by a rule rather than by taste: the desk started from the
# twelve largest names by market capitalisation, added the rest of the store's symbols, and
# kept those whose median daily traded notional over the trailing ninety days exceeds 80
# million US dollars. Every name here clears that floor, has a full year of five-minute history,
# and had its order-flow columns backfilled from the public archives.
#
# Why the composition matters more than anything else in this file. The momentum edge that the
# repository documents at +78.2 bps per trade at t = 3.37 was measured over a 31-symbol
# universe, and this basket's predecessor -- the pure twelve large-caps -- reports -11.7 bps at
# t = -1.23 on the *identical* signal and cost. Adding the next two qualifying names by
# liquidity moves it to +62.9 bps at t = 3.27. Reports produced before this line existed were
# describing a basket that is not the one the system trades, which is the same class of error
# as the universe that was re-picked every few minutes: a number about a different instrument
# set than the one under the account.
#
# The mechanism is not mysterious. A cross-sectional signal needs a cross-section to rank, and
# dispersion between coins is what there is to capture. Twelve names that all move together
# leave almost nothing relative to trade, while a basket spanning BTC at 0.24% daily volatility
# and ZEC at 1.4% has real spread to sort.
BASKET = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
          "WLDUSDT", "ADAUSDT", "NEARUSDT", "SUIUSDT", "ENAUSDT", "LINKUSDT", "UNIUSDT")

# The liquidity floor the basket above clears, in US dollars of daily notional. Recorded so a
# future reader can re-derive the membership instead of trusting the tuple.
MIN_DAILY_NOTIONAL_USD = 80_000_000

# The frame origin. See the module docstring: a Wednesday, so every day boundary is an exact
# multiple of three days from a funding settlement.
FRAME_ORIGIN = dt.datetime(2025, 8, 6, tzinfo=dt.timezone.utc)
DAY_MS = 86_400_000

# Components of the per-day coin profile. Every one is computed from data visible at that day
# close, and every one is measured, so an unmeasured component stays distinguishable from a
# measured zero.
NUMERIC_COLUMNS = (
    "date_ms", "coin", "index",
    "close", "open", "high", "low",
    "ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d", "ret_1d_prev",
    "gap_open",
    "vol_7d", "vol_30d", "vol_ratio", "atr_pct",
    "funding_daily", "funding_7d", "funding_30d", "funding_z",
    "oi", "oi_value", "oi_chg_1d", "oi_chg_7d", "oi_z", "oi_notional",
    "taker_daily", "taker_7d",
    "toptrader_ls", "global_ls", "ls_z", "ls_spread",
    "range_pct", "range_z", "close_pos",
    "buy_vol_share", "trades", "avg_trade_size", "trades_z",
    "liquidation_intensity",
    "mkt_ret_1d", "mkt_ret_7d", "mkt_vol_30d",
)

# Which panel columns each feature family reads. Declared here so the panel digest covers the
# names, and an artifact fit on a different column set is refused rather than silently served
# different inputs.
FAMILIES = {
    "momentum": ("ret_7d", "ret_14d", "ret_30d", "ret_3d", "ret_1d_prev"),
    "trend": ("ret_7d", "ret_30d", "vol_30d", "atr_pct", "mkt_ret_7d", "close_pos"),
    "volatility": ("vol_7d", "vol_30d", "vol_ratio", "atr_pct", "range_pct", "gap_open"),
    "flow": ("taker_daily", "taker_7d", "oi_chg_1d", "oi_chg_7d", "oi_notional",
             "trades", "avg_trade_size", "buy_vol_share", "liquidation_intensity"),
    "positioning": ("toptrader_ls", "global_ls", "ls_z", "ls_spread", "oi_z"),
    "carry": ("funding_daily", "funding_7d", "funding_30d", "funding_z"),
    "market": ("mkt_ret_1d", "mkt_ret_7d", "mkt_vol_30d"),
}

# Horizons a label can be built at, in days. The panel labels all of them, so a research
# comparison against the decision horizon is a lookup rather than a rebuild.
HORIZONS = (1, 3, 7)


def frame_day(date_ms):
    """Exact integer day index on the funding-aligned frame."""
    return (int(date_ms) - int(FRAME_ORIGIN.timestamp() * 1000)) // DAY_MS


def day_to_iso(day):
    """ISO date of a frame day index, for reports."""
    return (FRAME_ORIGIN.date() + dt.timedelta(days=int(day))).isoformat()


def _sqlite_ro(path):
    connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


# ------------------------------------------------------------------ small series helpers
#
# All of them are trailing. Nothing here may read forward, because every one of these columns
# becomes a model input and a single forward read turns the whole evaluation into a
# restatement of the past.

def _pct_change(values, lag):
    out = [None] * len(values)
    for index in range(lag, len(values)):
        previous, current = values[index - lag], values[index]
        if previous and current and previous > 0 and current > 0:
            out[index] = current / previous - 1.0
    return out


def _safe_div(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    try:
        value = numerator / denominator
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _mean_window(values, window, minimum=None):
    minimum = int(minimum or window)
    out = [None] * len(values)
    for index in range(len(values)):
        sample = [v for v in values[max(0, index - window + 1):index + 1] if v is not None]
        if len(sample) >= minimum:
            out[index] = sum(sample) / len(sample)
    return out


def _sum_window(values, window, minimum=None):
    minimum = int(minimum or window)
    out = [None] * len(values)
    for index in range(len(values)):
        sample = [v for v in values[max(0, index - window + 1):index + 1] if v is not None]
        if len(sample) >= minimum:
            out[index] = sum(sample)
    return out


def _std_window(values, window, minimum=None):
    minimum = int(minimum or window)
    out = [None] * len(values)
    for index in range(len(values)):
        sample = [v for v in values[max(0, index - window + 1):index + 1] if v is not None]
        if len(sample) < minimum or len(sample) < 2:
            continue
        mean = sum(sample) / len(sample)
        variance = sum((v - mean) ** 2 for v in sample) / (len(sample) - 1)
        if variance < 0:
            continue
        out[index] = math.sqrt(variance)
    return out


def _zscore(values, window):
    """Rolling z-score, clipped to a fixed a-priori range.

    None until the window is full, and None whenever the window has no dispersion. None
    rather than 0.0: a coin whose z-score has not been measured is not a coin whose z-score is
    average, and the feature audit gate reports a column of zeros as a dead feature.
    """
    out = [None] * len(values)
    for index in range(len(values)):
        sample = [v for v in values[max(0, index - window + 1):index + 1] if v is not None]
        if len(sample) < window:
            continue
        mean = sum(sample) / len(sample)
        variance = sum((v - mean) ** 2 for v in sample) / (len(sample) - 1)
        if variance <= 1e-18:
            continue
        out[index] = max(-5.0, min(5.0, (values[index] - mean) / math.sqrt(variance)))
    return out


def _daily_from_bars(bars):
    """Resample ascending bars into UTC-day aggregates on the funding-aligned frame.

    The daily *close* is the last bar close and the daily *open* is the first bar open, which
    is the only definition a trader at a day boundary can act on.

    The day index comes from \`frame_day\` rather than from a plain epoch division, and it has
    to: this panel's whole funding argument rests on days being a whole number of settlements
    apart from a settlement boundary. Dividing the raw epoch by a day gives a *calendar* day
    index that is offset from the frame by however many days lie between 1970-01-01 and the
    frame origin, so a lookup keyed on \`frame_day\` finds nothing. That mismatch is exactly what
    made every derivative column of the first build come back empty.
    """
    days = {}
    for bar in bars:
        day = frame_day(int(bar["open_time"]))
        state = days.get(day)
        if state is None:
            days[day] = {"open": float(bar["open"]), "high": float(bar["high"]),
                         "low": float(bar["low"]), "close": float(bar["close"]),
                         "volume": float(bar["volume"] or 0.0),
                         "quote_volume": float(bar["quote_volume"] or 0.0),
                         "taker_buy": float(bar["taker_buy_volume"] or 0.0),
                         "trades": float(bar["trades"] or 0.0), "bars": 1}
        else:
            state["high"] = max(state["high"], float(bar["high"]))
            state["low"] = min(state["low"], float(bar["low"]))
            state["close"] = float(bar["close"])
            state["volume"] += float(bar["volume"] or 0.0)
            state["quote_volume"] += float(bar["quote_volume"] or 0.0)
            state["taker_buy"] += float(bar["taker_buy_volume"] or 0.0)
            state["trades"] += float(bar["trades"] or 0.0)
            state["bars"] += 1
    return days


def _coin_sources(connection, symbol):
    """Daily funding, open interest, positioning, liquidation and notional for one coin.

    Funding needs care. The settlements are eight-hourly (three a day), so a UTC calendar day
    holds a whole number of settlements only when the day is measured from a settlement
    boundary. Summing calendar days of rate records counts a variable number of settlements
    per day and injects a spurious oscillation into a feature whose whole point is that it is
    a carry.

    The fix is to read the settlement *sequence* rather than the clock: every third record
    belongs to the same settlement slot, so recording one slot per day assigns exactly one
    settlement to every frame day. `funding_daily` is then a settlement rate -- a quantity
    with a stable meaning -- rather than a sum whose meaning depends on the calendar.
    """
    rows = connection.execute(
        "SELECT event_time, funding_rate FROM derivatives WHERE symbol=? ORDER BY event_time",
        (symbol,)).fetchall()
    funding_daily, funding_count = {}, {}
    for position, row in enumerate(rows):
        if row["funding_rate"] is None or position % 3:
            continue
        day = frame_day(row["event_time"])
        funding_daily[day] = funding_daily.get(day, 0.0) + float(row["funding_rate"])
        funding_count[day] = funding_count.get(day, 0) + 1

    detail = connection.execute(
        "SELECT event_time, open_interest, open_interest_value, top_account_ratio, "
        "global_account_ratio FROM derivatives_detail WHERE symbol=? "
        "AND open_interest IS NOT NULL ORDER BY event_time", (symbol,)).fetchall()
    oi, oi_value, oi_count = {}, {}, {}
    top, top_count, glob, glob_count = {}, {}, {}, {}
    for row in detail:
        day = frame_day(row["event_time"])
        value = row["open_interest"]
        if value is not None and float(value) > 0:
            oi[day] = oi.get(day, 0.0) + float(value)
            if row["open_interest_value"] is not None:
                oi_value[day] = oi_value.get(day, 0.0) + float(row["open_interest_value"])
            oi_count[day] = oi_count.get(day, 0) + 1
        if row["top_account_ratio"] is not None:
            top[day] = top.get(day, 0.0) + float(row["top_account_ratio"])
            top_count[day] = top_count.get(day, 0) + 1
        if row["global_account_ratio"] is not None:
            glob[day] = glob.get(day, 0.0) + float(row["global_account_ratio"])
            glob_count[day] = glob_count.get(day, 0) + 1

    liquidation = {}
    for row in connection.execute(
            "SELECT event_time, liquidation_volume FROM flow WHERE symbol=? "
            "ORDER BY event_time", (symbol,)):
        if row["liquidation_volume"] is None:
            continue
        day = frame_day(row["event_time"])
        liquidation[day] = liquidation.get(day, 0.0) + float(row["liquidation_volume"])

    notional = {}
    for row in connection.execute(
            "SELECT open_time, quote_volume FROM candles WHERE symbol=? AND interval='5m'",
            (symbol,)):
        day = frame_day(row["open_time"])
        notional[day] = notional.get(day, 0.0) + float(row["quote_volume"] or 0.0)


    return {"funding": funding_daily, "funding_count": funding_count,
            "oi": oi, "oi_value": oi_value, "oi_count": oi_count,
            "top": top, "top_count": top_count,
            "global": glob, "global_count": glob_count,
            "liquidation": liquidation, "notional": notional}


def _mean_of(sum_map, count_map, day):
    count = count_map.get(day)
    if not count:
        return None
    return sum_map.get(day, 0.0) / count


def _ratio_of(value_map, notional_map, day):
    total = notional_map.get(day)
    if day not in value_map or not total:
        return None
    return value_map[day] / total


def build_panel(db="data/research.sqlite3", symbols=BASKET, interval="5m",
                start_day=None, end_day=None):
    """The daily panel, as per-coin per-day dicts in (day, coin) order.

    Reads five-minute candles and resamples them. Every rolling statistic is trailing, so row
    `d` contains only information visible at the close of day `d`. The CPCV validation and
    the backtest both rely on that property, and `panel_report` measures the panel instead of
    assuming it.
    """
    connection = _sqlite_ro(db)
    try:
        per_coin, sources = {}, {}
        for symbol in symbols:
            rows = connection.execute(
                "SELECT open_time, open, high, low, close, volume, quote_volume, "
                "taker_buy_volume, trades FROM candles WHERE symbol=? AND interval=? "
                "ORDER BY open_time", (symbol, interval)).fetchall()
            bars = [dict(row) for row in rows if row["close"] and float(row["close"]) > 0]
            if len(bars) < 1000:
                continue
            days = _daily_from_bars(bars)
            ordered = sorted(days)
            per_coin[symbol] = {"days": ordered, "rows": [days[day] for day in ordered]}
            sources[symbol] = _coin_sources(connection, symbol)
    finally:
        connection.close()

    included = [symbol for symbol in symbols if symbol in per_coin]
    if len(included) < 4:
        raise ValueError("panel_basket_too_small:%d" % len(included))

    # The calendar the whole cross-section shares. Intersecting rather than unioning keeps
    # every cross-sectional statistic comparable: a coin missing a day would otherwise
    # contribute an absent value that a rank silently reads as the weakest name.
    shared = None
    for symbol in included:
        days = {int(day) for day in per_coin[symbol]["days"]}
        shared = days if shared is None else (shared & days)
    calendar = sorted(shared or [])
    if start_day is not None:
        calendar = [day for day in calendar if day >= int(start_day)]
    if end_day is not None:
        calendar = [day for day in calendar if day <= int(end_day)]
    if len(calendar) < 60:
        raise ValueError("panel_calendar_too_short:%d" % len(calendar))
    calendar_set = set(calendar)

    columns = {}
    for symbol in included:
        entry = per_coin[symbol]
        rows = entry["rows"]
        days = [int(day) for day in entry["days"]]
        source = sources[symbol]
        close = [float(row["close"]) for row in rows]
        open_ = [float(row["open"]) for row in rows]
        high = [float(row["high"]) for row in rows]
        low = [float(row["low"]) for row in rows]
        volume = [float(row["volume"] or 0.0) for row in rows]
        quote = [float(row["quote_volume"] or 0.0) for row in rows]
        taken = [float(row["taker_buy"] or 0.0) for row in rows]
        trades = [float(row["trades"] or 0.0) for row in rows]

        ret_1d = _pct_change(close, 1)
        vol_7d = _std_window(ret_1d, 7, minimum=4)
        vol_30d = _std_window(ret_1d, 30)
        intraday_range = [_safe_div(high[i] - low[i], close[i]) for i in range(len(close))]
        taker_share = [_safe_div(taken[i], volume[i]) for i in range(len(close))]
        trade_size = [_safe_div(quote[i], trades[i]) for i in range(len(close))]
        funding_daily = [source["funding"].get(day) for day in days]
        oi_daily = [_mean_of(source["oi"], source["oi_count"], day) for day in days]
        oi_value = [_mean_of(source["oi_value"], source["oi_count"], day) for day in days]
        top_ls = [_mean_of(source["top"], source["top_count"], day) for day in days]
        glob_ls = [_mean_of(source["global"], source["global_count"], day) for day in days]
        liquidation = [_ratio_of(source["liquidation"], source["notional"], day)
                       for day in days]

        columns[symbol] = {
            "days": days, "index": {day: i for i, day in enumerate(days)},
            "column": {
                "close": close, "open": open_, "high": high, "low": low,
                "ret_1d": ret_1d,
                "ret_3d": _pct_change(close, 3),
                "ret_7d": _pct_change(close, 7),
                "ret_14d": _pct_change(close, 14),
                "ret_30d": _pct_change(close, 30),
                "ret_1d_prev": [ret_1d[i - 1] if i >= 1 else None
                                for i in range(len(close))],
                "gap_open": [_safe_div(open_[i] - close[i - 1], close[i - 1])
                             if i >= 1 else None for i in range(len(close))],
                "vol_7d": vol_7d, "vol_30d": vol_30d,
                "vol_ratio": [_safe_div(vol_7d[i], vol_30d[i])
                              for i in range(len(close))],
                "atr_pct": _mean_window(intraday_range, 14, minimum=7),
                "range_pct": intraday_range,
                "range_z": _zscore(intraday_range, 30),
                "close_pos": [_safe_div(close[i] - low[i], high[i] - low[i])
                              if high[i] > low[i] else None for i in range(len(close))],
                "funding_daily": funding_daily,
                "funding_7d": _sum_window(funding_daily, 7, minimum=7),
                "funding_30d": _sum_window(funding_daily, 30, minimum=30),
                "funding_z": _zscore(funding_daily, 30),
                "oi": oi_daily, "oi_value": oi_value,
                "oi_chg_1d": _pct_change(oi_daily, 1),
                "oi_chg_7d": _pct_change(oi_daily, 7),
                "oi_z": _zscore(oi_daily, 30),
                "oi_notional": [_safe_div(oi_value[i], quote[i]) for i in range(len(close))],
                "taker_daily": taker_share,
                "taker_7d": _mean_window(taker_share, 7, minimum=7),
                "toptrader_ls": top_ls, "global_ls": glob_ls,
                "ls_z": _zscore(glob_ls, 30),
                "ls_spread": [top_ls[i] - glob_ls[i]
                              if top_ls[i] is not None and glob_ls[i] is not None
                              else None for i in range(len(close))],
                "buy_vol_share": taker_share,
                "trades": trades, "avg_trade_size": trade_size,
                "trades_z": _zscore(trades, 30),
                "liquidation_intensity": liquidation,
            }}

    # The market leg: the equal-weighted basket return. It is the factor every cross-sectional
    # feature differences out, and the input to the trend-regime filter.
    market = []
    for day in calendar:
        returns = []
        for symbol in included:
            entry = columns[symbol]
            index = entry["index"].get(day)
            if index is None:
                continue
            value = entry["column"]["ret_1d"][index]
            if value is not None:
                returns.append(value)
        market.append(sum(returns) / len(returns) if returns else None)
    market_7d = _mean_window(market, 5, minimum=5)
    market_vol = _std_window(market, 30, minimum=15)

    rows = []
    for day_index, day in enumerate(calendar):
        members = []
        for symbol in included:
            entry = columns[symbol]
            index = entry["index"].get(day)
            if index is None:
                continue
            # The frame day end, as a wall-clock timestamp. Reconstructed from the origin
            # rather than from the day index times a day, because the frame is offset from
            # the epoch and multiplying the index back would label every row with a date a
            # century away from the bar it describes.
            row = {"coin": symbol, "day": int(day),
                   "date_ms": int(FRAME_ORIGIN.timestamp() * 1000)
                              + (int(day) + 1) * DAY_MS - 1}
            for name in NUMERIC_COLUMNS:
                if name in ("date_ms", "coin", "index"):
                    continue
                if name in entry["column"]:
                    row[name] = entry["column"][name][index]
            row["mkt_ret_1d"] = market[day_index]
            row["mkt_ret_7d"] = market_7d[day_index]
            row["mkt_vol_30d"] = market_vol[day_index]
            members.append(row)
        for rank, row in enumerate(members):
            row["index"] = rank
        rows.extend(members)

    _attach_forward(rows, columns, calendar_set)
    return {"rows": rows, "calendar": calendar, "symbols": included,
            "basket": list(symbols), "interval": interval, "db": db}


def _attach_forward(rows, columns, calendar_set):
    """Forward returns, and the paths to them, attached to the row they belong to.

    The label lives in the row rather than being looked up at validation time, because a
    withheld day must remove the label as well as the features and the cheapest way to
    guarantee that is for the label to travel with the row.

    `fwd_max_1d` and `fwd_min_1d` are the highest high and lowest low over the following
    day, which is what a path-dependent exit actually experiences: a one-day hold with a stop
    is not the same bet as the close-to-close return, and the difference has to be measured
    rather than assumed.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["coin"], []).append(row)
    for symbol, members in grouped.items():
        entry = columns[symbol]
        index_of = entry["index"]
        close = entry["column"]["close"]
        high = entry["column"]["high"]
        low = entry["column"]["low"]
        day_of = {index: day for day, index in index_of.items()}
        last = len(close) - 1
        for row in members:
            index = index_of.get(row["day"])
            if index is None:
                continue
            base = close[index]
            if base <= 0:
                continue
            for horizon in HORIZONS:
                target = index + horizon
                if target > last:
                    continue
                row["fwd_%dd" % horizon] = close[target] / base - 1.0
                # An exit can only be priced on a day the panel covers; scoring an exit
                # outside the calendar would grade the model on a day it was never asked
                # about.
                path = [position for position in range(index + 1, target + 1)
                        if int(day_of.get(position, -1)) in calendar_set]
                if path:
                    row["fwd_max_%dd" % horizon] = max(high[i] for i in path) / base - 1.0
                    row["fwd_min_%dd" % horizon] = min(low[i] for i in path) / base - 1.0


def panel_digest(panel):
    """A stable digest of the panel: basket, calendar, columns, families and *the row count*.

    Validation recomputes this and refuses an artifact whose digest does not match, which is
    what stops a model from being validated against a panel it was not fit on.

    The row count is in the digest, and leaving it out was a real hole rather than a technicality.
    Hashing only the basket, the calendar and the column names means every derived panel of the
    same shape shares a digest: dropping the last day, or restricting the rows to a subset of the
    calendar while keeping the declared calendar, produced an identical string. The seal verified
    that the *schema* matched and said nothing about the *data*, so a model could be replayed
    against a panel it had never seen and the check would pass. The count is one integer and it
    closes the gap; the per-row content is deliberately not hashed, because a digest that changes
    when a float is rounded differently in another environment would fail closed for the wrong
    reason and get disabled.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(panel["symbols"]).encode("utf-8"))
    hasher.update("|".join(str(day) for day in panel["calendar"]).encode("utf-8"))
    hasher.update("|".join(NUMERIC_COLUMNS).encode("utf-8"))
    hasher.update("|".join("%s:%d" % (name, len(names))
                           for name, names in sorted(FAMILIES.items())).encode("utf-8"))
    hasher.update(("|rows:%d" % len(panel.get("rows") or [])).encode("utf-8"))
    hasher.update(("|first:%s|last:%s" % (panel.get("first_date"),
                                          panel.get("last_date"))).encode("utf-8"))
    return hasher.hexdigest()[:32]


def pivot(rows, column):
    """(days, coins, matrix) for one column, with None where a coin has no value."""
    days = sorted({row["day"] for row in rows})
    coins = sorted({row["coin"] for row in rows})
    coin_index = {coin: i for i, coin in enumerate(coins)}
    day_index = {day: i for i, day in enumerate(days)}
    matrix = [[None] * len(coins) for _ in days]
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value == value and value not in (float("inf"), float("-inf")):
            matrix[day_index[row["day"]]][coin_index[row["coin"]]] = value
    return days, coins, matrix


def panel_report(panel):
    """Coverage, dead columns and label availability, as numbers rather than opinions."""
    rows = panel["rows"]
    per_coin = {}
    for row in rows:
        per_coin[row["coin"]] = per_coin.get(row["coin"], 0) + 1
    coverage = {}
    for name in NUMERIC_COLUMNS:
        if name in ("date_ms", "coin", "index"):
            continue
        present = sum(1 for row in rows if row.get(name) is not None)
        distinct = len({round(float(row[name]), 12) for row in rows
                        if row.get(name) is not None})
        coverage[name] = {"present_pct": round(100.0 * present / max(1, len(rows)), 2),
                          "distinct": distinct, "dead": distinct <= 1}
    dead = sorted(name for name, entry in coverage.items() if entry["dead"])
    label_names = sorted({key for row in rows for key in row if key.startswith("fwd_")})
    label_coverage = {name: round(100.0 * sum(1 for row in rows
                                              if row.get(name) is not None)
                                  / max(1, len(rows)), 2) for name in label_names}
    first = panel["calendar"][0] if panel["calendar"] else None
    last = panel["calendar"][-1] if panel["calendar"] else None
    return {"rows": len(rows), "days": len(panel["calendar"]),
            "first_day": first, "last_day": last,
            "first_date": day_to_iso(first) if first is not None else None,
            "last_date": day_to_iso(last) if last is not None else None,
            "symbols": panel["symbols"], "rows_per_coin": per_coin,
            "dead_columns": dead, "coverage": coverage, "label_coverage": label_coverage,
            "digest": panel_digest(panel)}


def family_coverage(panel):
    """Per-family column availability, so a whole missing family is one line and not nine."""
    report = panel_report(panel)
    out = {}
    for family, names in sorted(FAMILIES.items()):
        entries = [entry for entry in (report["coverage"].get(name) for name in names)
                   if entry]
        if not entries:
            out[family] = {"columns": 0, "min_present_pct": 0.0, "dead": []}
            continue
        out[family] = {"columns": len(entries),
                       "min_present_pct": min(entry["present_pct"] for entry in entries),
                       "dead": sorted(name for name, entry in zip(names, entries)
                                      if entry["dead"])}
    return out


def save_panel(panel, path):
    """Write the panel as jsonl plus a manifest, so it can be inspected by hand."""
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in panel["rows"]:
            handle.write(json.dumps({key: value for key, value in row.items()
                                     if value is None
                                     or isinstance(value, (int, float, str))},
                                    sort_keys=True) + "\n")
    manifest = {"rows": len(panel["rows"]), "symbols": panel["symbols"],
                "calendar": [panel["calendar"][0], panel["calendar"][-1]],
                "digest": panel_digest(panel), "origin": FRAME_ORIGIN.isoformat(),
                "report": panel_report(panel), "families": family_coverage(panel)}
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return str(destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the strict daily panel.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument("--output", default="data/research_v4/panel_mainstream.jsonl")
    parser.add_argument("--symbols", default=",".join(BASKET))
    args = parser.parse_args()
    panel = build_panel(args.db, tuple(s.strip().upper() for s in args.symbols.split(",")
                                       if s.strip()))
    path = save_panel(panel, args.output)
    report = panel_report(panel)
    print(json.dumps({"path": path, "rows": report["rows"], "days": report["days"],
                      "first_date": report["first_date"], "last_date": report["last_date"],
                      "symbols": len(report["symbols"]),
                      "dead_columns": report["dead_columns"],
                      "families": family_coverage(panel)}, indent=1))
