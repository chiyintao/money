"""Walk-forward fold construction, and the leakage it has to prevent.

The split between train and test was already purged. The boundary between the fit set
and the early-stopping set was not, and that one is easy to miss because both sit inside
"training": the iteration count is chosen on rows whose labels overlap the rows the model
just fitted. Measured on a real dataset the gap there was 0 bars while labels reached 12
bars forward.
"""
import pytest

from app.models.walkforward import _label_span_bars, _purged_split, time_blocks

STEP = 300_000


def _row(bar, symbol="BTCUSDT", horizon=12):
    return {"timestamp": bar * STEP, "symbol": symbol, "future_return": 0.001,
            "label_end_time": (bar + horizon) * STEP}


def _rows(bars, symbols=("BTCUSDT", "ETHUSDT")):
    return [_row(bar, symbol) for bar in range(bars) for symbol in symbols]


def test_label_span_uses_the_maximum_not_the_median():
    """One long label is enough to leak, so the gap is built from the longest."""
    rows = _rows(50, ("BTCUSDT",))
    rows[7]["label_end_time"] = (7 + 40) * STEP
    times = sorted({row["timestamp"] for row in rows})
    assert _label_span_bars(rows, times) == 40


def test_label_span_rounds_up_a_partial_bar():
    rows = _rows(50, ("BTCUSDT",))
    rows[3]["label_end_time"] = 3 * STEP + int(12.4 * STEP)
    times = sorted({row["timestamp"] for row in rows})
    assert _label_span_bars(rows, times) == 13


def test_label_span_is_none_without_intervals():
    """None rather than a guess: inventing a span would silently under-purge."""
    rows = [{"timestamp": b * STEP, "symbol": "BTCUSDT"} for b in range(20)]
    times = sorted({row["timestamp"] for row in rows})
    assert _label_span_bars(rows, times) is None


def _split(bars=400, horizon=12, validation_fraction=0.1):
    rows = _rows(bars)
    times = sorted({row["timestamp"] for row in rows})
    by_time = {}
    for index, row in enumerate(rows):
        by_time.setdefault(row["timestamp"], []).append(index)
    train_idx = [i for stamp in times[:360] for i in by_time[stamp]]
    fit_idx, valid_idx = _purged_split(train_idx, rows, by_time, times, horizon,
                                       validation_fraction)
    return rows, fit_idx, valid_idx


def test_fit_labels_resolve_before_validation_opens():
    """The property the fix exists for."""
    rows, fit_idx, valid_idx = _split()
    last_fit_label_end = max(rows[i]["label_end_time"] for i in fit_idx)
    valid_start = min(rows[i]["timestamp"] for i in valid_idx)
    assert last_fit_label_end < valid_start


def test_the_gap_is_the_requested_number_of_bars():
    rows, fit_idx, valid_idx = _split(horizon=12)
    last_fit = max(rows[i]["timestamp"] for i in fit_idx)
    valid_start = min(rows[i]["timestamp"] for i in valid_idx)
    assert (valid_start - last_fit) / STEP == 13


def test_a_longer_horizon_widens_the_gap():
    _r1, fit1, valid1 = _split(horizon=6)
    _r2, fit2, valid2 = _split(horizon=24)
    assert len(fit2) < len(fit1)


def test_fit_and_validation_never_share_a_timestamp():
    """Cutting by row index split one bar's symbols across the two sets."""
    rows, fit_idx, valid_idx = _split()
    fit_times = {rows[i]["timestamp"] for i in fit_idx}
    valid_times = {rows[i]["timestamp"] for i in valid_idx}
    assert fit_times & valid_times == set()


def test_validation_keeps_its_intended_size():
    _rows_, fit_idx, valid_idx = _split(bars=400, validation_fraction=0.1)
    assert len(valid_idx) == pytest.approx(0.1 * (len(fit_idx) + len(valid_idx)), rel=0.25)


def test_too_little_history_falls_back_instead_of_failing():
    """A fold with no room to purge still reports something rather than raising."""
    rows = _rows(30, ("BTCUSDT",))
    times = sorted({row["timestamp"] for row in rows})
    by_time = {}
    for index, row in enumerate(rows):
        by_time.setdefault(row["timestamp"], []).append(index)
    train_idx = list(range(len(rows)))
    fit_idx, valid_idx = _purged_split(train_idx, rows, by_time, times, 12)
    assert fit_idx and valid_idx


def test_time_blocks_partition_the_timeline_in_order():
    rows = _rows(120, ("BTCUSDT",))
    blocks = time_blocks(rows, 4)
    assert len(blocks) == 4
    flat = [stamp for block in blocks for stamp in block]
    assert flat == sorted(flat)