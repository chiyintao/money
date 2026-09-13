import asyncio
import json
import random
import time

import pytest

from app.models.drift import drift_report, drift_from_rows, feature_drift, psi
from app.models.drift_monitor import DriftMonitor, sample_dataset
from app.models.retrain import RetrainScheduler
from app.models.training_policy import retrain_required


def rows(count, shift=0.0, symbols=("AUSDT", "BUSDT")):
    return [{"x": index * 0.01 + shift + random.random() * 0.001,
             "y": (index % 7) + shift,
             "symbol": symbols[index % len(symbols)]}
            for index in range(count)]


def test_quantile_bins_ignore_a_reference_outlier():
    # Equal-width bins over min/max let one outlier compress every other observation into
    # a single bin, so the index measures the outlier instead of the distribution.
    reference = [index * 0.01 for index in range(1000)] + [1e6]
    same_shape = [index * 0.01 for index in range(1000)]
    assert psi(reference, same_shape) < 0.05


def test_drift_is_detected_and_reported_per_feature():
    reference = {"x": [index * 0.01 for index in range(1000)],
                 "y": [index * 0.01 for index in range(1000)]}
    current = {"x": [index * 0.01 + 5 for index in range(1000)],
               "y": [index * 0.01 for index in range(1000)]}
    report = drift_report(reference, current)
    assert report["status"] == "drift"
    assert report["drifted"] == ["x"]
    assert report["features"]["y"]["drifted"] is False
    assert report["compared"] == 2


def test_names_restrict_the_comparison():
    reference = {"x": list(range(1000)), "y": list(range(1000))}
    current = {"x": [value + 900 for value in range(1000)],
               "y": [value + 900 for value in range(1000)]}
    report = drift_report(reference, current, names=["y"])
    assert report["compared"] == 1
    assert report["drifted"] == ["y"]


def test_small_samples_get_a_wider_threshold():
    # PSI is a sample statistic; at 200 reference rows 0.15 is noise, at 200000 it is a
    # regime change. A fixed constant cannot tell those apart.
    small = feature_drift(list(range(200)), [value + 5 for value in range(200)])
    large = feature_drift(list(range(5000)), [value + 5 for value in range(5000)])
    assert small["threshold"] > large["threshold"]


def test_drift_from_rows_tolerates_missing_and_non_numeric_values():
    report = drift_from_rows(rows(400), rows(400, shift=9.0), ["x", "y", "absent"])
    assert report["drifted"]
    assert report["features"]["absent"]["reference_rows"] == 0
    assert "absent" not in report["drifted"]  # unmeasurable is not drifted


def test_sample_dataset_draws_from_the_whole_file(tmp_path):
    # A dataset is written chronologically, so the first N rows are one symbol on one day.
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(5000):
            handle.write(json.dumps({"x": float(index), "y": 1.0}) + "\n")
    sample = sample_dataset(str(path), features=("x", "y"), limit=500, seed=3)
    assert sample["rows"] == 5000
    assert sample["sampled"] == 500
    # Reservoir sampling, not the head of the file.
    assert max(sample["columns"]["x"]) > 4000


def test_sample_dataset_returns_none_for_a_missing_file(tmp_path):
    assert sample_dataset(str(tmp_path / "absent.jsonl")) is None


def test_monitor_reports_unavailable_without_a_reference():
    monitor = DriftMonitor(reference={}, interval_seconds=0)
    report = monitor.check()
    assert report["status"] == "unavailable"
    assert monitor.blocks_entries() is False


def test_monitor_waits_for_enough_observations():
    reference = {"x": [index * 0.01 for index in range(1000)]}
    monitor = DriftMonitor(reference=reference, features=("x",), interval_seconds=0)
    for index in range(5):
        monitor.observe({"x": index * 0.01})
    assert monitor.check()["status"] == "insufficient"


def test_monitor_detects_drift_and_can_block():
    reference = {"x": [index * 0.01 for index in range(1000)]}
    monitor = DriftMonitor(reference=reference, features=("x",), window=200,
                           interval_seconds=0, policy="block")
    for index in range(200):
        monitor.observe({"x": index * 0.01 + 50.0})
    report = monitor.check()
    assert report["status"] == "drift"
    assert report["drifted"] == ["x"]
    assert monitor.blocks_entries() is True
    assert monitor.health()["drifted"] == ["x"]


def test_monitor_is_stable_on_matching_inputs():
    rng = random.Random(5)
    reference = {"x": [rng.gauss(0, 1) for _ in range(4000)]}
    monitor = DriftMonitor(reference=reference, features=("x",), window=500,
                           interval_seconds=0, policy="block")
    for _ in range(500):
        monitor.observe({"x": rng.gauss(0, 1)})
    assert monitor.check()["status"] == "stable"
    assert monitor.blocks_entries() is False


def test_monitor_window_is_bounded():
    monitor = DriftMonitor(reference={}, window=50, features=("x",), interval_seconds=0)
    for index in range(500):
        monitor.observe({"x": float(index)})
    assert len(monitor.observations) == 50
    assert monitor.observations[-1]["x"] == 499.0


def test_monitor_ignores_non_numeric_and_unknown_features():
    reference = {"x": [index * 0.01 for index in range(1000)]}
    monitor = DriftMonitor(reference=reference, features=("x",), interval_seconds=0)
    monitor.observe({"x": None, "unknown": 5.0})
    assert monitor.observations == []


def test_monitor_due_respects_its_interval():
    # last_run_ms starts at zero, so against real epoch milliseconds the first check is
    # always due; the interval only governs the ones after it.
    monitor = DriftMonitor(reference={}, interval_seconds=60)
    now = int(time.time() * 1000)
    assert monitor.due(now_ms=now) is True
    monitor.check(now_ms=now)
    assert monitor.due(now_ms=now) is False
    assert monitor.due(now_ms=now + 61_000) is True


def test_retrain_required_reasons():
    assert retrain_required(None) == (True, "no_model")
    now = int(time.time() * 1000)
    fresh = {"created_at": now, "dataset": {"sha256": "abc"}, "feature_version": "features-v3"}
    assert retrain_required(fresh, now_ms=now)[0] is False
    assert retrain_required(fresh, now_ms=now, current_dataset_hash="def")[1] == "dataset_changed"
    assert retrain_required(fresh, now_ms=now, current_feature_version="features-v4")[1] == "features_changed"
    assert retrain_required(fresh, now_ms=now + 10_000_000, max_age_ms=1000)[1] == "model_expired"


def test_scheduler_calls_the_trainer_only_when_stale():
    calls = []

    async def train():
        calls.append(1)
        return {"status": "done"}

    async def scenario():
        now = int(time.time() * 1000)
        fresh = {"created_at": now, "dataset": {}, "feature_version": "features-v3"}
        scheduler = RetrainScheduler(train, lambda: fresh, interval_ms=1000,
                                     feature_version="features-v3")
        first = await scheduler.run_once(now_ms=now)
        stale = {"created_at": now - 10_000, "dataset": {}, "feature_version": "features-v3"}
        second = RetrainScheduler(train, lambda: stale, interval_ms=1000,
                                  feature_version="features-v3")
        result = await second.run_once(now_ms=now)
        return first, result, second.snapshot()

    first, result, snapshot = asyncio.run(scenario())
    assert first["status"] == "skipped"
    assert calls == [1]  # the fresh manifest must not have triggered a training run
    assert result["status"] == "trained"
    assert result["reason"] == "model_expired"
    assert snapshot["last_status"] == "done"


def test_provenance_requires_a_promoted_model():
    from app.strategy.live_models import ModelDecision

    class Runtime:
        models = {"production": object()}
        promotion = {"status": "candidate", "reason": "no_portfolio_oos"}
        loaded_at = 0

    decision = ModelDecision.__new__(ModelDecision)
    decision.runtime = Runtime()
    decision.require_promoted = True
    decision.max_model_age_ms = 0
    assert decision.provenance_problem() == "model_not_promoted:no_portfolio_oos"
    decision.require_promoted = False
    assert decision.provenance_problem() is None


def test_provenance_rejects_expired_weights():
    from app.strategy.live_models import ModelDecision

    class Runtime:
        models = {"production": object()}
        promotion = {"status": "active", "created_at": int(time.time() * 1000) - 10_000_000}
        loaded_at = int(time.time() * 1000)

    decision = ModelDecision.__new__(ModelDecision)
    decision.runtime = Runtime()
    decision.require_promoted = False
    decision.max_model_age_ms = 3600 * 1000
    assert decision.provenance_problem() == "model_expired"
    decision.max_model_age_ms = 0
    assert decision.provenance_problem() is None


def test_provenance_is_silent_when_no_model_is_loaded():
    from app.strategy.live_models import ModelDecision

    class Runtime:
        models = {}
        promotion = {}
        loaded_at = 0

    decision = ModelDecision.__new__(ModelDecision)
    decision.runtime = Runtime()
    decision.require_promoted = True
    decision.max_model_age_ms = 3600 * 1000
    # No model at all is a different failure, reported by the mode string.
    assert decision.provenance_problem() is None

def _drifted_monitor(policy="block"):
    reference = {"x": [index * 0.01 for index in range(1000)]}
    monitor = DriftMonitor(reference=reference, features=("x",), window=200,
                           interval_seconds=0, policy=policy)
    for index in range(200):
        monitor.observe({"x": index * 0.01 + 50.0})
    monitor.check()
    return monitor


def test_a_blocked_drift_verdict_actually_refuses_entries():
    """The policy the operator set has to be the policy the decision layer applies.

    DRIFT_POLICY=block was read by the health check, which reported
    "drift_blocking_entries", and by nothing else. The monitor recorded the event, the
    dashboard said entries were blocked, and entries were not blocked.
    """
    from app.strategy.decision_loop import _drift_entry_check

    class Errors:
        def __init__(self): self.notes = []
        def note(self, key, exc): self.notes.append((key, repr(exc)))

    class Runtime:
        pass

    runtime = Runtime()
    runtime.errors = Errors()
    runtime.drift = _drifted_monitor(policy="block")
    allowed, reason = _drift_entry_check(runtime)
    assert allowed is False and reason == "feature_drift:x"

    # Under the default policy the finding is reported and trading continues.
    runtime.drift = _drifted_monitor(policy="warn")
    assert _drift_entry_check(runtime) == (True, None)

    # No monitor, and a monitor with no verdict yet, both permit trading rather than
    # stopping the session on a check that never ran.
    runtime.drift = None
    assert _drift_entry_check(runtime) == (True, None)
    runtime.drift = DriftMonitor(reference={}, interval_seconds=0, policy="block")
    assert _drift_entry_check(runtime) == (True, None)


def test_a_broken_drift_monitor_does_not_stop_the_session():
    from app.strategy.decision_loop import _drift_entry_check

    class Errors:
        def __init__(self): self.notes = []
        def note(self, key, exc): self.notes.append((key, repr(exc)))

    class Monitor:
        def blocks_entries(self): raise RuntimeError("monitor exploded")

    class Runtime:
        pass

    runtime = Runtime()
    runtime.errors = Errors()
    runtime.drift = Monitor()
    allowed, reason = _drift_entry_check(runtime)
    assert allowed is True and reason is None
    assert runtime.errors.notes, "the failure is reported rather than swallowed"


def test_drift_makes_the_model_require_retraining():
    """Age, dataset and contract all say the artifact is current. The market does not."""
    manifest = {"created_at": int(time.time() * 1000), "feature_version": "features-v4",
                "dataset": {"sha256": "abc"}}
    clean = retrain_required(manifest, current_dataset_hash="abc",
                             current_feature_version="features-v4", drifted=[])
    assert clean == (False, "fresh")
    drifted = retrain_required(manifest, current_dataset_hash="abc",
                               current_feature_version="features-v4",
                               drifted=["x", "y"])
    assert drifted == (True, "feature_drift:x,y")
    # None means the monitor never produced a verdict, which is not the same as clean:
    # the artifact checks still decide.
    assert retrain_required(manifest, current_dataset_hash="abc",
                            current_feature_version="features-v4", drifted=None)[0] is False


def test_the_scheduler_reads_the_monitor_rather_than_a_snapshot():
    """The verdict is taken at evaluation time -- the monitor is the live object."""
    monitor = _drifted_monitor(policy="warn")
    scheduler = RetrainScheduler(lambda: {}, lambda: None,
                                 drift_fn=lambda: monitor.last_report)
    assert scheduler.drifted_features() == ["x"]
    required, reason = scheduler.evaluate()
    assert required is True and reason == "feature_drift:x"
    assert scheduler.snapshot()["drift_wired"] is True

    # No monitor configured, and a monitor that has not run, both read as "not checked"
    # rather than as "clean" or as a retrain demand every interval.
    assert RetrainScheduler(lambda: {}, lambda: None).drifted_features() is None
    quiet = RetrainScheduler(lambda: {}, lambda: None, drift_fn=lambda: {})
    assert quiet.drifted_features() is None
    assert quiet.evaluate()[1] == "no_model"

