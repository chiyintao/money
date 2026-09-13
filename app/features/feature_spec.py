"""The feature contract shared by training, validation and serving.

This module deliberately imports nothing. The feature list is needed by dataset
validation, model training and live inference, and keeping it in fit_model created an
import cycle (quality -> fit_model -> dataset_split -> quality) once validation started
deriving its required fields from it.

Every feature is scale-free so a single model can serve every symbol: raw price levels
are not comparable between BTC (78000) and a small alt (0.0006), which is why a model
trained on raw levels only worked for the symbols it had seen.
"""

# --------------------------------------------------------------- v3: price and funding
FEATURES_V3 = ("ema20_gap", "ema50_gap", "rsi", "atr_pct", "return_10", "volume_ratio",
               # Perpetual-specific positioning. A perpetual is tied to spot by funding,
               # and deviations from no-arbitrage values are documented to be larger in
               # crypto than in traditional FX and to comove across coins. The live loop
               # already receives the current rate; these features make its history
               # learnable.
               "funding_rate", "funding_z", "funding_carry_24h", "mark_basis")

# ------------------------------------------- v4: what was collected and never modelled
#
# app/order_flow.py computes three more families from data the system has been collecting
# for its whole life: the taker volume, trade count and quote volume on every candle, the
# open interest and long/short ratios behind the thirty-day endpoints, and the per-minute
# aggressor and liquidation buckets. Every one of those functions had exactly zero
# production callers. The collection was wired and the features were not, so the model
# never saw any of it -- the same shape of defect as the mark price that was fetched,
# stored, and never used to compute the basis.
#
# The diagnostic counters each family returns (order_flow_bars, positioning_rows,
# flow_buckets, trades, avg_trade_size, measured) are deliberately NOT features. They
# describe how much data was available, not what the market did. A model trained on them
# learns the collector, and in a dataset where collection worked they are constant, which
# is exactly the degenerate-bound condition the out-of-distribution gate cannot see.
ORDER_FLOW_FEATURES = ("taker_imbalance", "taker_imbalance_z", "trade_count_z",
                       "avg_trade_size_z", "quote_volume_ratio")
POSITIONING_FEATURES = ("oi_change_5m", "oi_change_1h", "oi_z", "oi_price_quadrant",
                        "long_short_ratio_z", "smart_retail_gap",
                        "global_long_short_ratio", "taker_buy_sell_ratio_z")
FLOW_FEATURES = ("flow_delta_ratio", "flow_delta_z", "flow_trade_intensity_z",
                 "flow_largest_trade_z", "liquidation_pressure", "liquidation_share",
                 "liquidation_z")

FEATURES_V4 = (FEATURES_V3 + ORDER_FLOW_FEATURES + POSITIONING_FEATURES + FLOW_FEATURES)

# Every set the code can build, by the version string an artifact declares. Serving reads
# this to decide whether it can produce what a model asks for, which is a different
# question from whether the model matches the code's current constant. Conflating the two
# is what makes a feature addition an outage: a v3 artifact is perfectly servable by code
# that also knows about v4.
FEATURE_SETS = {"features-v3": FEATURES_V3, "features-v4": FEATURES_V4}
FEATURE_VERSION = "features-v4"
FEATURES = FEATURE_SETS[FEATURE_VERSION]

# Every name any version can produce. An artifact may declare a subset of this; it may not
# declare anything outside it, because serving would have no way to fill the value.
KNOWN_FEATURES = frozenset(name for names in FEATURE_SETS.values() for name in names)

# Which family each modelled feature belongs to, for the checks that report a whole input
# going missing rather than one column at a time.
FAMILY_FEATURES = {"price": FEATURES_V3,
                   "order_flow": ORDER_FLOW_FEATURES,
                   "positioning": POSITIONING_FEATURES,
                   "flow": FLOW_FEATURES}


def features_for(version):
    """The feature list an artifact declaring this version needs, or None."""
    return FEATURE_SETS.get(str(version or ""))


def unproducible(names):
    """Declared feature names this code cannot compute. Empty means servable."""
    return sorted(set(names or ()) - KNOWN_FEATURES)
# The label's definition, versioned separately from the features because the two fail
# differently. A feature change makes a model unable to be served; a label change makes it
# wrong while remaining perfectly servable, which is the harder failure to notice.
#
# label-v1 measured the forward return from the close of the bar the decision was made on:
#   close[i+h] / close[i] - 1
# The live loop cannot enter at that price. It decides on a closed bar and the order fills
# at the next price the market offers, which is the next bar's open. Training therefore
# assumed an entry that was, on average, half a bar of drift better than any fill it would
# ever get -- a bias in the direction of every label, and one that the round-trip cost
# estimate cannot correct because it is not a cost, it is a different price.
LABEL_VERSION = "label-v2"

# Fixed a-priori bounds applied in both training and serving.
#
# A year of 5m bars contains flash crashes and bad prints: DOTUSDT printed a single bar
# ranging 135% (close 3.425 -> 1.498) and ADAUSDT one of 95%. Left alone these produce
# feature values two orders of magnitude beyond the 99th percentile, which widens the
# out-of-distribution bounds until that check is useless and, at serving time, feeds the
# model inputs it never saw. The bounds below are chosen from what the features mean
# rather than measured from the data, so they carry no look-ahead information.
FEATURE_CLIP = {
    "ema20_gap": (-0.5, 0.5),
    "ema50_gap": (-0.5, 0.5),
    "rsi": (0.0, 100.0),
    "atr_pct": (0.0, 0.2),
    "return_10": (-0.5, 0.5),
    "volume_ratio": (0.0, 50.0),
    # Binance caps the funding rate at +/-0.75% per settlement and carry sums three of
    # them. The z-score is bounded because a near-constant funding history makes the
    # denominator tiny and the score explode.
    "funding_rate": (-0.0075, 0.0075),
    "funding_z": (-5.0, 5.0),
    "funding_carry_24h": (-0.0225, 0.0225),
    "mark_basis": (-0.02, 0.02),
    # --- order flow. The imbalance is a share difference by construction, so its bound is
    # not a judgement. The z-scores share funding_z's bound for the same reason: a
    # near-constant history makes the denominator tiny and the score explode.
    "taker_imbalance": (-1.0, 1.0),
    "taker_imbalance_z": (-5.0, 5.0),
    "trade_count_z": (-5.0, 5.0),
    "avg_trade_size_z": (-5.0, 5.0),
    "quote_volume_ratio": (0.0, 50.0),
    # --- positioning. Open interest can gap on a listing or a squeeze, so the change is
    # bounded generously rather than at what a quiet hour looks like.
    "oi_change_5m": (-1.0, 1.0),
    "oi_change_1h": (-1.0, 1.0),
    "oi_z": (-5.0, 5.0),
    "oi_price_quadrant": (-1.0, 1.0),
    "long_short_ratio_z": (-5.0, 5.0),
    "smart_retail_gap": (-5.0, 5.0),
    "global_long_short_ratio": (0.0, 20.0),
    "taker_buy_sell_ratio_z": (-5.0, 5.0),
    # --- per-minute flow. The delta ratio is a share difference; the rest are z-scores.
    "flow_delta_ratio": (-1.0, 1.0),
    "flow_delta_z": (-5.0, 5.0),
    "flow_trade_intensity_z": (-5.0, 5.0),
    "flow_largest_trade_z": (-5.0, 5.0),
    "liquidation_pressure": (-5.0, 5.0),
    "liquidation_share": (0.0, 1.0),
    "liquidation_z": (-5.0, 5.0),
}

# Forward returns are bounded for training as well. A single -56% label contributes a
# squared error roughly 775x a typical one, so a handful of unpredictable flash bars
# would otherwise dominate the fit.
LABEL_CLIP = 0.2


def clip_feature(name, value):
    """Bound one modelled feature; unknown names pass through unchanged."""
    limits = FEATURE_CLIP.get(name)
    if limits is None:
        return value
    low, high = limits
    return low if value < low else high if value > high else value


def clip_features(values):
    """Bound every known feature the caller actually supplied.

    Not every name in FEATURES. The raw snapshot in features.py produces the price and
    funding families and nothing else; the collected families are merged in afterwards by
    FeatureSource. Clipping against the current constant made the two disagree the moment
    the list grew -- adding the collected families turned a working snapshot into a
    KeyError for a column features.py is not responsible for.
    """
    return {name: clip_feature(name, values[name]) for name in KNOWN_FEATURES
            if name in values}
