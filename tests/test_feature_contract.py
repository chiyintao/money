"""Which features a model declares, and what the system does when it declares wrongly.

The promotion gate compared one string: manifest["feature_version"] against whatever
version the code currently produces. That check is blind in the direction that matters.
A model can name a column this code cannot compute at all and be promoted, because the
version string matched. It then serves with that column silently absent -- and worse, the
intersection of "declared" and "the contract" comes out empty, which switches the
degeneracy check off. The gate whose whole job is to refuse a model reading inputs nothing
can fill was disabled by the very input it was meant to catch.
"""
import tempfile

from app.models.calibration import fit_isotonic
from app.features.feature_spec import FEATURE_SETS, FEATURE_VERSION, FEATURES_V3, unproducible
from app.strategy.live_models import ModelDecision
from app.models.model_registry import ModelRegistry

FEATURES_V4 = FEATURE_SETS["features-v4"]

METRICS = {
    "test": {"rows": 100, "directional_accuracy_pct": 55},
    "portfolio_oos": {"costs_included": True, "trades": 40, "net_return": .02,
                      "max_drawdown": .05},
    "calibration": {"fitted": True},
}


def _calibrator():
    return fit_isotonic([.1, .2, .3, .4, .5, .6], [0, 0, 1, 0, 1, 1])


def _attempt(version, features):
    registry = ModelRegistry(tempfile.mkdtemp())
    model = {"coef": [1], "feature_version": version, "features": list(features),
             "split": {"ready": True, "label_intervals_verified": True}}
    registry.register(model, METRICS, {"rows": 300}, "v", calibrator=_calibrator())
    return registry.promote("v")


def test_a_column_nothing_can_compute_is_refused_at_promotion():
    """The gate is at promotion because the alternative is a serve-time fallback.

    TabularPredictor does refuse such an artifact, but it does so while loading, on a path
    that degrades to a fallback -- so the artifact is promoted, shipped, and only then
    reported as unloadable. Refusing it where the evidence is costs nothing.
    """
    import pytest

    with pytest.raises(ValueError, match="unproducible_declared_features:not_a_real_feature"):
        _attempt(FEATURE_VERSION, ["rsi", "not_a_real_feature"])
    assert unproducible(["rsi", "not_a_real_feature"]) == ["not_a_real_feature"]


def test_an_unknown_declared_version_is_refused():
    import pytest

    with pytest.raises(ValueError, match="incompatible_feature_version"):
        _attempt("features-v9", ["rsi"])
    with pytest.raises(ValueError, match="incompatible_feature_version"):
        _attempt(None, ["rsi"])


def test_an_older_version_that_this_code_still_knows_is_servable():
    """String equality made extending the contract impossible.

    A genuine v3 artifact names ten columns this code still fills. Demanding that the
    string equal the newest version refuses it for no reason, and -- the same comparison
    read the other way -- ACCEPTS an artifact that names a column which does not exist
    yet, because only the version string was ever compared.
    """
    assert _attempt("features-v3", FEATURES_V3) == "v"
    # The subset direction is what makes a version boundary meaningful.
    assert _attempt(FEATURE_VERSION, FEATURES_V3) == "v"


def test_a_version_cannot_smuggle_in_a_column_from_a_later_one():
    import pytest

    with pytest.raises(ValueError,
                       match="features_outside_declared_version:taker_imbalance"):
        _attempt("features-v3", ["rsi", "taker_imbalance"])


def test_the_manifest_records_what_the_artifact_declares():
    """The gate can only judge the declared contract if the manifest carries it."""
    registry = ModelRegistry(tempfile.mkdtemp())
    model = {"coef": [1], "feature_version": FEATURE_VERSION, "features": ["rsi", "atr_pct"]}
    manifest = registry.register(model, {}, {"rows": 10}, "v")
    assert manifest["features"] == ["rsi", "atr_pct"]
    assert manifest["feature_version"] == FEATURE_VERSION


def test_an_empty_declaration_still_means_the_whole_contract():
    """The registry refuses on an empty declaration; the runtime treats it as unknown.

    An artifact that declares nothing is not refused at promotion, because a model whose
    input list is unknown is a different case from one that reads a column that cannot be
    produced. Serving asks for everything, which is the conservative reading.
    """
    assert _attempt(FEATURE_VERSION, []) == "v"


class _Member:
    def __init__(self, features):
        self.features = features

    def describe(self):
        return {}


class _Runtime:
    promotion = {"status": "active"}

    def __init__(self, features):
        self.models = {"m": _Member(features)}
        self.load_errors = {}


def _decision(features):
    runtime = _Runtime(features)
    decision = ModelDecision(runtime, feature_source=object())
    # The health check reads the decision object off the runtime, so the fixture has to
    # wire it the way the service does.
    runtime.decisions = decision
    return runtime, decision


def test_required_features_asks_for_the_whole_contract_when_nothing_is_declared():
    _, decision = _decision([])
    assert len(decision.required_features()) == len(FEATURES_V4)


def test_a_partial_declaration_narrows_to_what_is_declared():
    _, decision = _decision(["rsi", "atr_pct", "ema20_gap"])
    assert set(decision.required_features()) == {"rsi", "atr_pct", "ema20_gap"}


def test_a_declaration_that_intersects_nothing_is_reported_not_ignored():
    """An empty required list switches the degeneracy check off. That is the defect.

    The service reported a healthy feature layer, the out-of-range gate had no columns to
    test, and the model was fed nothing it was fitted on.
    """
    from app.ops.health import features_check

    runtime, decision = _decision(["not_a_real_feature"])
    assert decision.required_features() == ()
    assert decision.unusable_features == ["not_a_real_feature"]
    assert "declared_features_unproducible" in runtime.load_errors["features"]
    verdict = features_check(runtime, 0)
    assert verdict["status"] == "degraded"
    assert verdict["detail"] == "declared_features_unproducible"
    assert verdict["unproducible"] == ["not_a_real_feature"]


def test_a_servable_declaration_reports_nothing():
    from app.ops.health import features_check

    class Source:
        def health(self):
            return {"rows": 10, "degraded_rows": 0, "degraded_features": [],
                    "extra_rows": 10, "requested": ["rsi"]}

    runtime, decision = _decision(["rsi"])
    decision.feature_source = Source()
    assert decision.unusable_features == []
    assert runtime.load_errors == {}
    assert features_check(runtime, 0)["status"] == "ok"
