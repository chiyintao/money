"""Ingestion estimates and progress reporting."""
from app.market.ingest import MAX_LIMIT, estimate_seconds, interval_ms


def test_estimate_scales_with_interval_and_symbol_count():
    # Measured: a year of 5m bars for fifteen symbols is about 71 pages each.
    five_minute = estimate_seconds(["S%d" % i for i in range(15)], "5m", 365)
    four_hour = estimate_seconds(["S%d" % i for i in range(15)], "4h", 365)
    assert five_minute > four_hour
    assert 900 < five_minute < 1400       # ~17 minutes
    assert four_hour < 120                # well under a minute


def test_estimate_grows_with_more_symbols():
    few = estimate_seconds(["A"] * 3, "5m", 365)
    many = estimate_seconds(["A"] * 15, "5m", 365)
    assert many > few


def test_estimate_is_never_zero():
    assert estimate_seconds(["A"], "4h", 1) >= 1


def test_interval_ms_matches_the_estimate_table():
    assert interval_ms("5m") == 300_000
    assert interval_ms("4h") == 14_400_000
    assert MAX_LIMIT == 1500
