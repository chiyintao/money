"""The feature contract must stay scale-free, or cross-symbol training silently breaks."""

from app.features.feature_spec import FAMILY_FEATURES, FEATURES, FEATURE_CLIP, FEATURE_SETS, FEATURE_VERSION, FEATURES_V3, KNOWN_FEATURES
from app.features.features import snapshot
from app.features.feature_source import FeatureSource


def modelled(rows, source=None):
    """The full modelled row, as the dataset builder and the live loop both build it.

    snapshot() alone produces the price and funding families. The three collected families
    are merged in by the shared FeatureSource, and a test that reads snapshot() directly is
    testing half the contract.
    """
    row, _degraded = (source or FeatureSource(store=None)).snapshot(rows, 'BTCUSDT')
    return row


def series(scale=1.0, count=60, drift=0.0004, amplitude=0.002):
    """A deterministic price path; `scale` multiplies every level."""
    rows = []
    price = 100.0
    for i in range(count):
        price *= 1 + (drift if i % 2 else -drift) + amplitude * ((i % 7) - 3) / 10
        rows.append({"open": price, "high": price * 1.001, "low": price * 0.999,
                     "close": price, "volume": 1000 + (i % 5) * 10})
    return [{"open": r["open"] * scale, "high": r["high"] * scale,
             "low": r["low"] * scale, "close": r["close"] * scale,
             "volume": r["volume"]} for r in rows]


def test_every_modelled_feature_is_produced_by_the_shared_source():
    """Every name in the contract must be in the row, whatever the source could fill.

    The row is what the model reads. A feature that is absent is a KeyError inside the
    design matrix -- or worse, a silently missing column. The source fills what it has no
    data for and reports the name instead of leaving a hole.
    """
    features = modelled(series())
    missing = [name for name in FEATURES if name not in features]
    assert missing == []
    assert all(isinstance(features[name], (int, float)) for name in FEATURES)


def test_a_source_with_no_store_reports_every_collected_family_as_degraded():
    """The three collected families say when they are empty, rather than reading as zero.

    app/order_flow.py computed twenty features from data the system has collected for its
    whole life and nothing called it. Wiring it in is only half the change: a family with
    no rows returns the same faces as a family whose value really is zero, which is exactly
    how four funding features came to be constant at serving time while carrying variance
    in training.
    """
    collected = tuple(FAMILY_FEATURES['order_flow'] + FAMILY_FEATURES['positioning']
                      + FAMILY_FEATURES['flow'])
    assert collected, 'the collected families must exist to be reported'
    source = FeatureSource(store=None)
    row, degraded = source.snapshot(series(), 'BTCUSDT', 1_700_000_000_000, 100.0)
    assert set(collected) <= set(degraded)
    for name in collected:
        assert row[name] == 0.0
    health = source.health()
    assert health['families'] == {name: len(names) for name, names in FAMILY_FEATURES.items()}
    assert health['reported_features'] == len(FEATURES)
    assert health['extra_rows'] == 0, 'a row with nothing collected is not a full row'


def test_modelled_features_are_scale_invariant():
    # BTC trades near 78000 and a small alt near 0.0006. A model pooled across both
    # only works if the modelled features are identical for a scaled price series.
    base = modelled(series(scale=1.0))
    for scale in (0.0006 / 100, 78000.0 / 100, 1e-6):
        scaled = modelled(series(scale=scale))
        for name in FEATURES:
            assert abs(scaled[name] - base[name]) < 1e-9, (name, scale)


def test_raw_levels_are_still_exposed_for_non_modelled_consumers():
    # strategy.py, the stop-distance calculation and the dashboard read raw levels.
    features = snapshot(series())
    for name in ("price", "ema20", "ema50", "atr", "volume", "rsi", "return_10",
                 "atr_pct", "volume_ratio"):
        assert name in features


def test_relative_features_agree_with_the_raw_levels():
    features = snapshot(series())
    assert abs(features["ema20_gap"] - (features["ema20"] / features["price"] - 1)) < 1e-12
    assert abs(features["ema50_gap"] - (features["ema50"] / features["price"] - 1)) < 1e-12
    assert abs(features["atr_pct"] - features["atr"] / features["price"]) < 1e-12


def test_the_contract_declares_every_version_it_can_serve():
    """A version bump must not be an outage.

    Serving selects the columns an artifact declares, out of the names this code can
    produce. Conflating "the artifact matches the current constant" with "the artifact can
    be served" is what turns adding a feature into every loaded model degrading at once.
    """
    assert FEATURE_VERSION in FEATURE_SETS
    assert FEATURES == FEATURE_SETS[FEATURE_VERSION]
    assert len(FEATURES) == len(set(FEATURES))
    assert set(FEATURES_V3) <= set(FEATURES)
    assert set(FEATURES) <= KNOWN_FEATURES
    assert set(KNOWN_FEATURES) <= set(FEATURE_CLIP), "an unbounded feature has no OOD check"
    assert set(KNOWN_FEATURES) == set(sum(FAMILY_FEATURES.values(), ()))
    # v3 stays servable: the artifacts in data/research_v3/candidates declare it.
    assert FEATURE_SETS['features-v3'] == FEATURES_V3


def test_snapshot_rejects_too_little_history():
    import pytest
    with pytest.raises(ValueError, match="warmup"):
        snapshot(series(count=49))
