"""The knobs that decide how much of the machine a training run may use.

Both were constants before: the dataset builder walked one symbol at a time and each
booster was pinned at two threads. On a twenty-four-core host that made the longest stage
of every run a single-core job, and the thread count was not reachable from any caller.
These tests pin the parsing and the bounds, not the speed: a benchmark belongs in the
audit, not in a test that would fail on a busy machine.
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.tabular_model import _gpu_requested, _threads  # noqa: E402
from app.models.train import _dataset_workers  # noqa: E402


@contextmanager
def env(key, value):
    """Set an environment variable for the duration of the block."""
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_dataset_workers_default_to_half_the_machine():
    with env("MODEL_DATASET_WORKERS", None):
        workers = _dataset_workers(15)
    assert 1 <= workers <= 15
    # The service streams quotes from the same machine while a run is in flight, so the
    # builder is not allowed to take every core.
    assert workers <= max(1, (os.cpu_count() or 2) // 2)


def test_dataset_workers_never_exceed_the_symbol_count():
    with env("MODEL_DATASET_WORKERS", None):
        assert _dataset_workers(1) == 1
        assert _dataset_workers(2) <= 2


def test_dataset_workers_one_is_the_serial_path():
    with env("MODEL_DATASET_WORKERS", "1"):
        assert _dataset_workers(15) == 1


def test_an_explicit_dataset_worker_count_is_honoured():
    with env("MODEL_DATASET_WORKERS", "4"):
        assert _dataset_workers(15) == 4
        # ... but not past the number of symbols there are to build.
        assert _dataset_workers(2) == 2


def test_a_nonsense_dataset_worker_count_falls_back():
    for value in ("", "many", "-3", "0"):
        with env("MODEL_DATASET_WORKERS", value):
            assert _dataset_workers(15) >= 1


def test_threads_default_to_most_of_the_machine():
    with env("MODEL_THREADS", None):
        threads = _threads()
    assert 1 <= threads <= 16


def test_an_explicit_thread_count_is_honoured():
    with env("MODEL_THREADS", "3"):
        assert _threads() == 3


def test_the_gpu_is_off_unless_asked_for():
    # The published LightGBM wheel has no GPU tree learner, so this must never be on by
    # default: asking for a device the build cannot provide raises instead of falling back.
    with env("MODEL_GBDT_DEVICE", None):
        assert _gpu_requested() is False
    with env("MODEL_GBDT_DEVICE", "cpu"):
        assert _gpu_requested() is False


def test_the_gpu_can_be_requested():
    for value in ("gpu", "GPU", "cuda"):
        with env("MODEL_GBDT_DEVICE", value):
            assert _gpu_requested() is True
