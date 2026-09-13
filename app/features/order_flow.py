"""Order-flow and derivative-positioning features.

Two families that were absent for different reasons.

The order-flow family was absent by accident. Binance's kline array has carried quote
volume (index 7), trade count (index 8) and taker buy base volume (index 9) since the
endpoint existed. The ingest path read indices 0-6 and dropped the rest, so a dataset
built from a response that already contained taker imbalance had no way to express it.

The positioning family was absent by omission. Open interest, the long/short ratios and
the taker volume ratio live behind endpoints that keep thirty days and archive nothing,
so they have to be collected on a schedule; the collector in derivatives_collect.py does
that, and these functions turn what it stores into a feature row.

Everything here is computed from rows already in hand. No function fetches anything, and
every one returns a plain dict so it can be merged into the existing feature snapshot.
"""
import math


def _ratio(numerator, denominator, default=0.0):
    if not denominator:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def _z_score(values, value, min_deviation=1e-9):
    """Z-score of the last value against the sample it belongs to."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    deviation = variance ** 0.5
    if deviation <= min_deviation:
        return 0.0
    score = (value - mean) / deviation
    return score if math.isfinite(score) else 0.0


# --------------------------------------------------------------- order flow
def order_flow_features(bars, window=48):
    """Taker imbalance, trade intensity and average trade size from stored candles.

    Returns zeros when the order-flow columns are absent, which is what an older database
    or an unfilled backfill looks like. The trade count is returned alongside so a caller
    can tell "no imbalance" from "not measured".
    """
    empty = {'taker_imbalance': 0.0, 'taker_imbalance_z': 0.0, 'trade_count_z': 0.0,
             'avg_trade_size_z': 0.0, 'quote_volume_ratio': 0.0, 'order_flow_bars': 0}
    usable = [row for row in (bars or []) if float(row.get('trades') or 0) > 0]
    if len(usable) < 5:
        return empty
    recent = usable[-window:]

    def imbalance(row):
        volume = float(row.get('volume') or 0)
        if volume <= 0:
            return None
        taker_buy = float(row.get('taker_buy_volume') or 0)
        # Buy-side taker volume above half the bar means aggressive buying.
        return (2 * taker_buy / volume) - 1.0

    series = [value for value in (imbalance(row) for row in recent) if value is not None]
    if not series:
        return empty
    current = series[-1]
    counts = [float(row.get('trades') or 0) for row in recent]
    sizes = [_ratio(float(row.get('volume') or 0), float(row.get('trades') or 0), 0.0)
             for row in recent]
    quotes = [float(row.get('quote_volume') or 0) for row in recent]
    return {
        'taker_imbalance': current,
        'taker_imbalance_z': _z_score(series, current),
        'trade_count_z': _z_score(counts, counts[-1]) if counts else 0.0,
        'avg_trade_size_z': _z_score(sizes, sizes[-1]) if sizes else 0.0,
        'quote_volume_ratio': _ratio(quotes[-1], sum(quotes[-6:-1]) / 5 if len(quotes) >= 6 else 0.0, 0.0),
        'order_flow_bars': len(recent),
    }


# ------------------------------------------------------- derivative positioning
def positioning_features(rows, window=48, price_change=None):
    """Open interest change, crowding and taker ratio from stored derivative statistics.

    Rows are expected oldest first with the nullable columns from derivatives_detail. A
    missing column stays zero, and the count of usable rows is reported so an empty
    collection is visible rather than silently neutral.
    """
    empty = {'oi_change_5m': 0.0, 'oi_change_1h': 0.0, 'oi_z': 0.0,
             'oi_price_quadrant': 0.0, 'long_short_ratio_z': 0.0,
             'smart_retail_gap': 0.0, 'global_long_short_ratio': 0.0,
             'taker_buy_sell_ratio_z': 0.0, 'positioning_rows': 0}
    usable = [row for row in (rows or []) if row.get('open_interest')]
    if len(usable) < 3:
        return empty
    recent = usable[-window:]
    interests = [float(row['open_interest']) for row in recent]
    current = interests[-1]
    reference = interests[-2] if len(interests) >= 2 else current
    hour_back = interests[-13] if len(interests) >= 13 else interests[0]

    def change(numerator, denominator):
        if not denominator:
            return 0.0
        value = numerator / denominator - 1.0
        return value if math.isfinite(value) else 0.0

    oi_change_5m = change(current, reference)
    oi_change_1h = change(current, hour_back)
    top = recent[-1].get('long_short_ratio')
    global_ratio = recent[-1].get('global_account_ratio')
    taker = [float(row['taker_buy_sell_ratio']) for row in recent
             if row.get('taker_buy_sell_ratio')]
    # Open interest rising with price rising is new money taking the same side; rising
    # against price is short covering. The sign pair is the whole content of the signal.
    #
    # The price leg used to be read from the row's own price_change column, which no writer
    # produced: the positioning endpoints return open interest and account ratios and no
    # price at all, and the schema has no such column. So the quadrant was 0.0 on every row
    # the system ever built, while the comment above it described a signal. The caller that
    # has the candles passes the change instead, measured over the same one-step horizon as
    # the open interest change it is paired with.
    price_up = price_change if price_change is not None else recent[-1].get('price_change')
    quadrant = 0.0
    if price_up is not None:
        quadrant = float((1 if oi_change_5m > 0 else -1) * (1 if float(price_up) > 0 else -1))
    return {
        'oi_change_5m': oi_change_5m,
        'oi_change_1h': oi_change_1h,
        'oi_z': _z_score(interests, current),
        'oi_price_quadrant': quadrant,
        'long_short_ratio_z': _z_score([float(row['long_short_ratio']) for row in recent
                                        if row.get('long_short_ratio')],
                                       float(top)) if top else 0.0,
        # Large accounts positioned differently from the crowd: the gap is the signal, not
        # either ratio on its own.
        'smart_retail_gap': (float(top) - float(global_ratio)) if (top and global_ratio) else 0.0,
        'global_long_short_ratio': float(global_ratio or 0.0),
        'taker_buy_sell_ratio_z': _z_score(taker, taker[-1]) if taker else 0.0,
        'positioning_rows': len(recent),
    }


# --------------------------------------------------- trade prints and liquidations
def flow_features(rows, window=60):
    """Aggressor imbalance and liquidation pressure from the per-minute flow buckets.

    A third source, and the one with the shortest life. The candles carry taker volume per
    bar; these buckets carry it per minute, split by the initiating side, alongside the
    liquidations that happened in the same minute. Liquidations are the part that has no
    substitute: a stop loss and a forced liquidation look identical on a price chart, and
    only one of them means somebody was carried out.

    Rows are oldest first dicts from Store.flow. The bucket count is returned so an empty
    collection reads as unmeasured rather than as balanced flow.
    """
    empty = {'flow_delta_ratio': 0.0, 'flow_delta_z': 0.0, 'flow_trade_intensity_z': 0.0,
             'flow_largest_trade_z': 0.0, 'liquidation_pressure': 0.0,
             'liquidation_share': 0.0, 'liquidation_z': 0.0, 'flow_buckets': 0}
    usable = [row for row in (rows or []) if float(row.get('trades') or 0) > 0
              or float(row.get('liquidations') or 0) > 0]
    if len(usable) < 3:
        return empty
    recent = usable[-window:]
    # Ratio first, then its own history: an imbalance of 0.3 means nothing without knowing
    # whether this symbol's normal imbalance is 0.02 or 0.25.
    deltas = [float(row['delta_ratio']) for row in recent
              if row.get('delta_ratio') is not None]
    counts = [float(row.get('trades') or 0) for row in recent]
    largest = [float(row.get('largest_trade') or 0) for row in recent]
    liquidated = [float(row.get('liquidation_volume') or 0) for row in recent]
    current_delta = deltas[-1] if deltas else 0.0
    last = recent[-1]
    # Signed by the side that was forced out: positive means shorts were liquidated.
    signed = float(last.get('liquidation_delta') or 0)
    volume = float(last.get('liquidation_volume') or 0)
    normal = sum(liquidated[:-1]) / len(liquidated[:-1]) if len(liquidated) > 1 else 0.0
    return {
        'flow_delta_ratio': current_delta,
        'flow_delta_z': _z_score(deltas, current_delta) if deltas else 0.0,
        'flow_trade_intensity_z': _z_score(counts, counts[-1]) if counts else 0.0,
        'flow_largest_trade_z': _z_score(largest, largest[-1]) if largest else 0.0,
        # Direction of the forced flow, scaled by how unusual the size is for this symbol.
        'liquidation_pressure': (1.0 if signed > 0 else -1.0 if signed < 0 else 0.0)
                                * _z_score(liquidated, liquidated[-1]) if liquidated else 0.0,
        # None in storage means the minute had no trades to compare against; zero means it
        # traded and none of it was liquidated. Collapsing both to zero would hide that.
        'liquidation_share': float(last.get('liquidation_share') or 0.0),
        'liquidation_z': _z_score(liquidated, liquidated[-1]) if liquidated else 0.0,
        'flow_buckets': len(recent),
    }


# Feature families exposed to the dataset builder. Kept as names plus a callable so the
# spec, the training job and the live path cannot disagree about what exists.
FAMILIES = {
    'order_flow': order_flow_features,
    'positioning': positioning_features,
    'flow': flow_features,
}
