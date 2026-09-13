"""History ingestion: range planning and kline parsing."""
import time

from app.market.ingest import coverage, interval_ms, missing_ranges, parse_klines, stored_span
from app.storage.storage import Store

STEP = interval_ms("5m")
START = 1_700_000_000_000


def candle(open_time):
    return {"open_time": open_time, "close_time": open_time + STEP - 1,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 10.0, "is_closed": 1}


def store_with(tmp_path, count, first_index=0):
    store = Store(str(tmp_path))
    rows = [candle(START + (first_index + i) * STEP) for i in range(count)]
    if rows:
        store.upsert_candles("BTCUSDT", "5m", rows)
    return store


def test_interval_ms_rejects_unknown_intervals():
    import pytest
    assert interval_ms("5m") == 300_000
    with pytest.raises(ValueError, match="unsupported_interval"):
        interval_ms("7m")


def test_empty_symbol_needs_the_whole_window(tmp_path):
    store = store_with(tmp_path, 0)
    end = START + 100 * STEP
    assert missing_ranges(store, "BTCUSDT", "5m", START, end) == [(START, end)]
    store.close()


def test_recent_bars_alone_still_require_the_earlier_window(tmp_path):
    # Regression: resuming forward from the newest bar made a symbol that had only
    # recent data look "covered", so the whole earlier window was silently skipped.
    store = store_with(tmp_path, 10, first_index=90)
    end = START + 100 * STEP
    oldest, newest = stored_span(store, "BTCUSDT", "5m")
    assert oldest == START + 90 * STEP
    assert missing_ranges(store, "BTCUSDT", "5m", START, end) == [(START, oldest)]
    store.close()


def test_full_coverage_needs_nothing(tmp_path):
    store = store_with(tmp_path, 100)
    end = START + 99 * STEP
    assert missing_ranges(store, "BTCUSDT", "5m", START, end) == []
    store.close()


def test_a_hole_in_the_middle_is_reported(tmp_path):
    # An interrupted download can leave a gap between stored bars. Scanning only the
    # oldest and newest bar would call this covered and train across the hole.
    store = store_with(tmp_path, 20)
    store.upsert_candles("BTCUSDT", "5m", [candle(START + i * STEP) for i in range(60, 80)])
    end = START + 79 * STEP
    ranges = missing_ranges(store, "BTCUSDT", "5m", START, end)
    assert ranges == [(START + 20 * STEP, START + 60 * STEP)]
    store.close()


def test_head_gap_hole_and_tail_gap_are_all_returned(tmp_path):
    store = store_with(tmp_path, 10, first_index=10)
    store.upsert_candles("BTCUSDT", "5m", [candle(START + i * STEP) for i in range(30, 40)])
    end = START + 59 * STEP
    ranges = missing_ranges(store, "BTCUSDT", "5m", START, end)
    assert ranges == [(START, START + 10 * STEP),
                      (START + 20 * STEP, START + 30 * STEP),
                      (START + 40 * STEP, START + 59 * STEP)]
    store.close()


def test_parse_klines_marks_unclosed_bars(tmp_path):
    now = 1_700_000_000_000
    payload = [
        [now - 600_000, "1.0", "2.0", "0.5", "1.5", "10.0", now - 300_001],
        [now - 300_000, "1.5", "2.5", "1.0", "2.0", "20.0", now + 300_000],
    ]
    rows = parse_klines(payload, now_ms=now)
    assert [row["is_closed"] for row in rows] == [1, 0]
    assert rows[0]["open_time"] == now - 600_000
    assert rows[1]["close"] == 2.0


def test_parse_klines_skips_malformed_entries():
    rows = parse_klines([[1, "x"], None, ["bad"]])
    assert rows == []


def test_coverage_reports_gaps(tmp_path):
    store = store_with(tmp_path, 5)
    store.upsert_candles("BTCUSDT", "5m", [candle(START + (10 + i) * STEP) for i in range(5)])
    report = coverage(store, ["BTCUSDT"], "5m")
    assert len(report) == 1
    assert report[0]["bars"] == 10
    assert report[0]["missing"] == 5
    assert report[0]["largest_gap_bars"] == 5
    assert report[0]["coverage_pct"] == 66.667
    store.close()
