"""What the service loads when the promoted model cannot be used.

The candidate fallback exists for one situation: the registry holds no promoted model at
all, and unvetted weights are better than no decision. It was applied to every situation.
A production model that is present but expired, missing, or unreadable took the same path
and was silently replaced by the very weights the promotion gate refused -- a downgrade in
provenance that looks from the outside exactly like a healthy service.
"""
from app.models.calibration import fit_isotonic
from app.features.feature_spec import FEATURE_VERSION
from app.models.model_registry import ModelRegistry
from app.models.model_runtime import classify_reason, load_calibrated_model
from app.strategy.live_models import RealModelRuntime


def _calibrator():
    return fit_isotonic([.1, .2, .3, .4, .5, .6], [0, 0, 1, 0, 1, 1])


def _promoted(tmp_path, max_age_ms=3600_000):
    reg = ModelRegistry(str(tmp_path / "models"))
    reg.register(
        {"coef": [1], "feature_version": FEATURE_VERSION,
         "split": {"ready": True, "label_intervals_verified": True}},
        {"test": {"rows": 100, "directional_accuracy_pct": 55},
         "portfolio_oos": {"costs_included": True, "trades": 40, "net_return": .02,
                           "max_drawdown": .05},
         "calibration": {"fitted": True}},
        {"rows": 300}, "v1", calibrator=_calibrator())
    reg.promote("v1", max_age_ms=max_age_ms)
    return reg


def test_an_empty_registry_and_an_expired_one_are_different_failures():
    """Both arrive as ValueError and used to leave as the same status.

    The difference decides the operator response -- train a model, or retrain the one that
    went stale -- and it decides whether the loader may substitute rejected weights.
    """
    assert classify_reason("no_production_model") == "absent"
    assert classify_reason("production_model_expired") == "unusable"
    assert classify_reason("production_model_missing") == "unusable"
    assert classify_reason("calibrator_missing") == "unusable"
    # An unrecognised failure is not an empty registry.
    assert classify_reason("something_new_broke") == "unusable"
    assert classify_reason(None) == "unusable"


def test_the_loader_names_the_kind_it_failed_with(tmp_path):
    empty = load_calibrated_model(ModelRegistry(str(tmp_path / "nothing")))
    assert (empty["status"], empty["reason"], empty["kind"]) == (
        "fallback", "no_production_model", "absent")

    reg = _promoted(tmp_path)
    live = load_calibrated_model(reg)
    assert (live["status"], live["kind"]) == ("active", "active")

    # The registry names expiry on its own clock; the loader reports what it was told.
    assert reg.current(now_ms=10 ** 15)["status"] == "expired"


def test_an_expired_production_model_is_not_replaced_by_candidates(tmp_path):
    """The service refuses instead of trading weights that failed review."""
    import json
    import time

    # Promoted with a generous age; the test then moves the deadline itself, so the
    # fixture cannot expire between register() and promote() on a slow machine.
    reg = _promoted(tmp_path)
    # Age the promotion so current() calls it expired.
    production = tmp_path / "models" / "production.json"
    payload = json.loads(production.read_text(encoding="utf-8"))
    payload["expires_at"] = int(time.time() * 1000) - 1000
    production.write_text(json.dumps(payload), encoding="utf-8")
    assert reg.current()["status"] == "expired"

    # A candidate that the runtime would otherwise be happy to load.
    candidate = tmp_path / "candidates" / "lightgbm-abc123"
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text(json.dumps(
        {"backend": "lightgbm", "status": "candidate", "features": ["price"]}),
        encoding="utf-8")

    runtime = RealModelRuntime(str(tmp_path), registry_root=str(tmp_path / "models"),
                               candidate_dir=str(tmp_path / "candidates"),
                               backends=("lightgbm",), chronos_enabled=False)
    runtime.load()
    assert runtime.promotion["kind"] == "unusable"
    assert runtime.promotion["reason"] == "production_model_expired"
    assert runtime.mode == "unavailable"
    assert runtime.models == {}
    assert runtime.load_errors["production"] == (
        "production_unusable:production_model_expired")
    assert runtime.unavailable_reason() == (
        "production_unusable:production_model_expired")


def test_the_refusal_can_be_waived_explicitly(tmp_path):
    import json
    import time

    # Promoted with a generous age; the test then moves the deadline itself, so the
    # fixture cannot expire between register() and promote() on a slow machine.
    reg = _promoted(tmp_path)
    production = tmp_path / "models" / "production.json"
    payload = json.loads(production.read_text(encoding="utf-8"))
    payload["expires_at"] = int(time.time() * 1000) - 1000
    production.write_text(json.dumps(payload), encoding="utf-8")

    runtime = RealModelRuntime(str(tmp_path), registry_root=str(tmp_path / "models"),
                               candidate_dir=str(tmp_path / "candidates"),
                               backends=("lightgbm",), chronos_enabled=False,
                               allow_candidate_fallback=True)
    runtime.load()
    # Nothing to fall back TO here, so it is still unavailable -- but the refusal was not
    # what stopped it, which is the distinction the flag makes.
    assert runtime.load_errors.get("production") is None
    assert runtime.mode == "unavailable"
    assert runtime.promotion["kind"] == "unusable"


def test_an_empty_registry_still_falls_back_to_candidates(tmp_path):
    """The fallback is not removed, only narrowed to the case it was written for."""
    import json

    candidate = tmp_path / "candidates" / "lightgbm-abc123"
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text(json.dumps(
        {"backend": "lightgbm", "status": "candidate", "features": ["price"]}),
        encoding="utf-8")
    runtime = RealModelRuntime(str(tmp_path), registry_root=str(tmp_path / "models"),
                               candidate_dir=str(tmp_path / "candidates"),
                               backends=("lightgbm",), chronos_enabled=False)
    runtime.load()
    assert runtime.promotion["kind"] == "absent"
    assert runtime.load_errors.get("production") is None
    # The candidate directory has a manifest but no artifact, so the member fails to
    # build; what matters here is that the loader tried, which is the old behaviour.
    assert "lightgbm" in runtime.load_errors

def test_health_reads_the_promotion_verdict_from_the_model_runtime():
    """The verdict lives on the model runtime; this check read it off the outer Runtime.

    Runtime has no promotion attribute, so getattr returned None in every deployment and
    the check reported "trading_unpromoted_weights" with mode=None -- for a properly
    promoted model and for a service that had loaded nothing at all. A status read from
    the wrong object, defaulting to a plausible value.
    """
    from app.ops.health import model_check

    class Members:
        promotion = {"status": "active", "kind": "active", "version": "v1"}
        mode = "production"
        models = {"production": object()}

    class Expired:
        promotion = {"status": "fallback", "kind": "unusable",
                     "reason": "production_model_expired"}
        mode = "unavailable"
        models = {}

    class Outer:
        decisions = None

    outer = Outer()
    outer.models = Members()
    promoted = model_check(outer, 0)
    assert (promoted["status"], promoted["detail"], promoted["mode"]) == (
        "ok", "tradable", "active")

    outer.models = Expired()
    unusable = model_check(outer, 0)
    assert unusable["status"] == "degraded"
    assert unusable["detail"] == "production_unusable:production_model_expired"
    assert unusable["mode"] == "unavailable"

    # No runtime at all is still a skip: not configured is not the same as configured
    # and unable.
    assert model_check(Outer(), 0)["detail"] == "no_model_runtime"


def test_health_and_the_loader_agree_about_the_same_registry(tmp_path):
    """End to end: an expired production model reaches /health as unusable."""
    import json
    import time
    from app.ops.health import model_check

    # Promoted with a generous age; the test then moves the deadline itself, so the
    # fixture cannot expire between register() and promote() on a slow machine.
    reg = _promoted(tmp_path)
    production = tmp_path / "models" / "production.json"
    payload = json.loads(production.read_text(encoding="utf-8"))
    payload["expires_at"] = int(time.time() * 1000) - 1000
    production.write_text(json.dumps(payload), encoding="utf-8")
    assert reg.current()["status"] == "expired"

    models = RealModelRuntime(str(tmp_path), registry_root=str(tmp_path / "models"),
                              candidate_dir=str(tmp_path / "candidates"),
                              backends=("lightgbm",), chronos_enabled=False)
    models.load()

    class Outer:
        decisions = None

    outer = Outer()
    outer.models = models
    verdict = model_check(outer, 0)
    assert verdict["status"] == "degraded"
    assert verdict["detail"] == "production_unusable:production_model_expired"
    assert models.mode == "unavailable" and models.models == {}
