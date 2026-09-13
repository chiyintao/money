"""Row builders derived from the feature contract.

Tests used to repeat the feature names by hand, so a feature-set change broke a dozen
files and had to be chased through each one. Building rows from FEATURES keeps every
test valid as the contract evolves.
"""
from app.features.feature_spec import FEATURES


def feature_values(index=0, **overrides):
    """Plausible, index-varying values for the current feature set."""
    values = {
        "ema20_gap": 0.001 + 0.0001 * (index % 13),
        "ema50_gap": -0.001 + 0.0001 * (index % 11),
        "rsi": 40 + index % 20,
        "atr_pct": 0.002,
        "return_10": 0.01 if index % 2 else -0.01,
        "volume_ratio": 1.0 + 0.01 * (index % 7),
    }
    for name in FEATURES:
        values.setdefault(name, 0.0)
    values.update(overrides)
    return {name: values[name] for name in FEATURES}


def row(index=0, symbol="X", future_return=None, label_end_time=None, **overrides):
    values = feature_values(index, **overrides)
    values["timestamp"] = index
    values["symbol"] = symbol
    values["label_end_time"] = index if label_end_time is None else label_end_time
    values["future_return"] = index * 0.001 if future_return is None else future_return
    return values


def rows(count, symbols=("X",), **overrides):
    return [row(index, symbol=symbol, **overrides)
            for symbol in symbols for index in range(count)]
