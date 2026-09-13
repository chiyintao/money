"""The archive reader, against the things that actually went wrong.

Every test here comes from a measured failure rather than a guessed edge case. The CDN
for these archives is unreliable in two specific ways, and both fail in the direction
that loses data silently, so both are pinned:

* it answers 404 for files that exist, in windows lasting minutes;
* it resolves to four addresses and one of them refuses connections, while urllib only
  ever tries the first.
"""
import io
import urllib.error
import zipfile

import pytest

from app.market import archive


def _zip_csv(text, name="BTCUSDT-5m-2025-09.csv"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr(name, text)
    return buffer.getvalue()


HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
          "taker_buy_volume,taker_buy_quote_volume,ignore")
ROW = ("1757487600000,111742.7,111789.1,111710.0,111764.4,215.273,1757487899999,"
       "24059857.68,1234,96.87,10830000.0,0")


def test_no_monthly_archive_is_named_for_metrics():
    """Metrics are published daily only; a monthly URL would 404 forever."""
    assert archive.metrics_filename("BTCUSDT", "2024-01-05") == \
        "BTCUSDT-metrics-2024-01-05.zip"


def test_url_layout():
    """Klines live under an interval directory; metrics do not.

    The previous assertion here omitted the `5m` segment, so it agreed with the code
    instead of with the archive. Every klines download then took a well-formed 404, which
    the backfill recorded as "archive absent" -- a permanent data gap reported as normal.
    A test that restates the bug is worse than no test, because it certifies it.
    """
    url = archive.archive_url("futures/um", "monthly", "klines", "BTCUSDT",
                              archive.klines_filename("BTCUSDT", "5m", "2025-09"),
                              interval="5m")
    assert url == ("https://data.binance.vision/data/futures/um/monthly/klines/"
                   "BTCUSDT/5m/BTCUSDT-5m-2025-09.zip")


def test_klines_url_refuses_to_be_built_without_an_interval():
    """The failure mode this guards is silent, so it has to be an exception."""
    try:
        archive.archive_url("futures/um", "monthly", "klines", "BTCUSDT",
                            archive.klines_filename("BTCUSDT", "5m", "2025-09"))
    except ValueError as exc:
        assert "interval_required_for_klines" in str(exc)
    else:
        raise AssertionError("klines URL was built without an interval")


def test_metrics_url_has_no_interval_directory():
    """Metrics are daily-only and are not filed under an interval."""
    url = archive.archive_url("futures/um", "daily", "metrics", "BTCUSDT",
                              archive.metrics_filename("BTCUSDT", "2025-09-10"))
    assert url == ("https://data.binance.vision/data/futures/um/daily/metrics/"
                   "BTCUSDT/BTCUSDT-metrics-2025-09-10.zip")


def test_columns_are_read_by_header_not_position():
    """`count` and `trades` name the same field in different vintages.

    Reading by a fixed index silently stores the wrong number on whichever layout
    differs, which is worse than failing because nothing downstream can tell.
    """
    header, rows = archive.read_csv(_zip_csv(HEADER + "\n" + ROW))
    bars = archive.parse_kline_rows(header, rows)
    assert len(bars) == 1
    assert bars[0]["taker_buy_volume"] == pytest.approx(96.87)
    assert bars[0]["quote_volume"] == pytest.approx(24059857.68)
    assert bars[0]["trades"] == 1234


def test_renamed_count_column_still_maps_to_trades():
    header = HEADER.replace("count", "trades")
    parsed_header, rows = archive.read_csv(_zip_csv(header + "\n" + ROW))
    assert archive.parse_kline_rows(parsed_header, rows)[0]["trades"] == 1234


def test_blank_trailing_line_is_skipped():
    """Several archives end with an empty row, which would parse as a bar of zeros."""
    _header, rows = archive.read_csv(_zip_csv(HEADER + "\n" + ROW + "\n\n"))
    assert len(rows) == 1


def test_flow_columns_are_absent_not_zero_when_the_file_lacks_them():
    """An older kline file has no taker_buy_volume; absence must not become 0.

    A zero would claim "no aggressive buying happened", a different and stronger
    statement than "this file does not say".
    """
    narrow = "open_time,open,high,low,close,volume,close_time,quote_volume,count"
    row = ("1757487600000,111742.7,111789.1,111710.0,111764.4,215.273,"
           "1757487899999,24059857.68,1234")
    header, rows = archive.read_csv(_zip_csv(narrow + "\n" + row))
    bars = archive.parse_kline_rows(header, rows)
    assert "taker_buy_volume" not in bars[0]
    assert bars[0]["quote_volume"] == pytest.approx(24059857.68)


def test_metrics_parse_into_the_stored_schema():
    text = ("create_time,symbol,sum_open_interest,sum_open_interest_value,"
            "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
            "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
            "2026-08-30 00:10:00,BTCUSDT,107635.96,8411054168.84,1.22556583,"
            "2.07953500,1.18550372,0.39217600")
    header, rows = archive.read_csv(_zip_csv(text, "BTCUSDT-metrics-2026-08-30.csv"))
    records = archive.parse_metrics_rows(header, rows)
    assert len(records) == 1
    assert records[0]["open_interest"] == pytest.approx(107635.96)
    assert records[0]["long_short_ratio"] == pytest.approx(2.079535)
    assert records[0]["top_account_ratio"] == pytest.approx(1.22556583)
    assert records[0]["global_account_ratio"] == pytest.approx(1.18550372)
    assert records[0]["taker_buy_sell_ratio"] == pytest.approx(0.392176)
    # Stated rather than assumed: the archive has no volume split, so these stay unset
    # instead of being filled with a zero the archive never published.
    assert "taker_buy_volume" not in records[0]


def test_metrics_timestamp_is_utc_milliseconds():
    header, rows = archive.read_csv(_zip_csv(
        "create_time,sum_open_interest\n2026-08-30 00:10:00,107635.96", "m.csv"))
    assert archive.parse_metrics_rows(header, rows)[0]["event_time"] == 1788048600000


def test_epoch_timestamps_in_microseconds_are_normalised():
    """Archives have used us for this column; a stray 1000 breaks every join."""
    header, rows = archive.read_csv(_zip_csv(
        "create_time,sum_open_interest\n1788048600000000,107635.96", "m.csv"))
    assert archive.parse_metrics_rows(header, rows)[0]["event_time"] == 1788048600000


def test_unparseable_rows_are_dropped_not_defaulted():
    text = HEADER + "\n" + ROW + "\nnot,a,bar,at,all,x,y,z,w,v,u,0"
    header, rows = archive.read_csv(_zip_csv(text))
    assert len(archive.parse_kline_rows(header, rows)) == 1


def test_a_404_is_retried_before_it_is_believed(monkeypatch):
    """A single 404 is not evidence of absence on this CDN.

    The window is real and was measured: a file published in 2024 and immutable since
    answered 404 on every retry for minutes, while curl fetched it with 200 seconds
    earlier. Believing the first 404 drops a month from the backfill and reports
    success.
    """
    calls = {"n": 0}

    def flaky(request, url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return b"payload"

    monkeypatch.setattr(archive, "_download_any_address", flaky)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    assert archive.fetch("https://example.invalid/x.zip", retries=5) == b"payload"
    assert calls["n"] == 3


def test_absence_is_reported_only_after_the_retry_budget(monkeypatch):
    def always_missing(request, url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(archive, "_download_any_address", always_missing)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    assert archive.fetch("https://example.invalid/x.zip", retries=2) is None


def test_a_connection_failure_raises_instead_of_reporting_absence(monkeypatch):
    """The most damaging confusion available: an outage is not a missing month."""
    def unreachable(request, url, timeout):
        raise archive.ArchiveError("every resolved address failed")

    monkeypatch.setattr(archive, "_download_any_address", unreachable)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    with pytest.raises(archive.ArchiveError):
        archive.fetch("https://example.invalid/x.zip", retries=1)


def test_every_address_is_tried_before_giving_up(monkeypatch):
    """One edge node refuses connections; urllib alone would only try the first.

    Measured: of four addresses for this host, three served the same immutable file in
    0.3s while the fourth timed out. Without the walk, about a quarter of all fetches
    fail for a reason unrelated to the request.
    """
    tried = []

    def only_third_works(address, parts, port, timeout):
        tried.append(address)
        if len(tried) < 3:
            raise OSError("connection refused")
        return b"body"

    monkeypatch.setattr(archive, "_resolve",
                        lambda host, port=443: ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    monkeypatch.setattr(archive, "_fetch_from", only_third_works)
    assert archive.fetch("https://example.invalid/x.zip", retries=0) == b"body"
    assert tried == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


def test_cached_payload_is_served_without_touching_the_network(monkeypatch, tmp_path):
    monkeypatch.setattr(archive, "_download_any_address",
                        lambda *a, **k: pytest.fail("network hit for a cached file"))
    (tmp_path / "x.zip").write_bytes(b"cached")
    assert archive.fetch("https://example.invalid/x.zip",
                         cache_dir=str(tmp_path)) == b"cached"


def test_a_download_lands_in_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(archive, "_download_any_address", lambda *a, **k: b"fresh")
    archive.fetch("https://example.invalid/x.zip", cache_dir=str(tmp_path))
    assert (tmp_path / "x.zip").read_bytes() == b"fresh"


def test_a_partial_download_is_never_left_behind(monkeypatch, tmp_path):
    """A truncated file in the cache would be served as complete forever."""
    def explode(request, url, timeout):
        raise archive.ArchiveError("failed mid-flight")

    monkeypatch.setattr(archive, "_download_any_address", explode)
    monkeypatch.setattr(archive.time, "sleep", lambda _s: None)
    with pytest.raises(archive.ArchiveError):
        archive.fetch("https://example.invalid/x.zip", cache_dir=str(tmp_path), retries=0)
    assert list(tmp_path.iterdir()) == []


def test_a_csv_less_zip_is_treated_as_empty():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("readme.txt", "nothing here")
    assert archive.read_csv(buffer.getvalue()) == ([], [])


def test_missing_payload_reads_as_empty():
    assert archive.read_csv(None) == ([], [])