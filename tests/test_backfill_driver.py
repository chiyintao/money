"""The backfill driver's bookkeeping, which decides what gets retried.

Progress state is the part of a long backfill that is easiest to get wrong and hardest
to notice: a state file that loses entries does not fail, it silently re-downloads
everything, and one that keeps wrong entries silently skips data forever.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_archives.py"


@pytest.fixture()
def driver(tmp_path, monkeypatch):
    """Load the driver with its state directory redirected to a temp path."""
    spec = importlib.util.spec_from_file_location("backfill_driver", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_driver"] = module
    monkeypatch.setattr(sys, "path", [str(SCRIPT.parents[1])] + sys.path)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "state")
    return module


def test_state_is_per_stage_not_shared(driver, tmp_path):
    """A shared file let the klines process erase the metrics progress.

    Measured: a metrics run completed 366 files while the klines process held the same
    file, leaving `metrics_done` empty so every restart redid all 366.
    """
    assert driver.state_path("klines") != driver.state_path("metrics")


def test_progress_survives_a_round_trip(driver):
    state = driver.load_state("klines")
    state["done"] = ["BTCUSDT:2025-09"]
    driver.save_state("klines", state)
    assert driver.load_state("klines")["done"] == ["BTCUSDT:2025-09"]


def test_stages_do_not_see_each_others_progress(driver):
    klines = driver.load_state("klines")
    klines["done"] = ["BTCUSDT:2025-09"]
    driver.save_state("klines", klines)
    assert driver.load_state("metrics")["done"] == []


def test_a_corrupt_state_file_reads_as_empty_rather_than_raising(driver):
    driver.STATE_DIR.mkdir(parents=True, exist_ok=True)
    driver.state_path("klines").write_text("{not json", encoding="utf-8")
    assert driver.load_state("klines")["done"] == []


def test_saving_leaves_no_partial_file_behind(driver):
    """A half-written file reads back as empty, discarding every completed download."""
    state = driver.load_state("klines")
    state["done"] = ["A", "B"]
    driver.save_state("klines", state)
    leftovers = [p.name for p in driver.STATE_DIR.iterdir() if p.suffix == ".part"]
    assert leftovers == []


def test_metric_days_covers_the_candle_range(driver, monkeypatch):
    class FakeDb:
        def execute(self, _sql):
            class Cursor:
                def fetchone(self):
                    return (1757487600000, 1757487600000 + 86400000 * 2)
            return Cursor()

    class FakeStore:
        db = FakeDb()

    monkeypatch.setattr(driver, "Store", lambda *a, **k: FakeStore())
    assert driver.metric_days() == ["2025-09-10", "2025-09-11", "2025-09-12"]