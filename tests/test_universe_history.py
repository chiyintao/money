import pytest

"""Point-in-time universe tests.

Each test names the bias it exists to catch. Both of them make a backtest look better
than it was, and neither is visible in the output of a backtest built on today's universe.
"""
import time

from app.storage.storage import Store
from app.models.training_job import TrainingRunner
from app.features.universe import TIER_EXCLUDED, TIER_MAINSTREAM, TIER_SPECULATIVE, classify_one
from app.features.universe_history import data_horizon, exclusion_reason, point_in_time, series_from_rows, series_from_windows, symbol_facts, symbols_at, universe_changes

DAY = 86_400_000


# 288 five-minute bars in a day, so the tier thresholds in universe.RULES apply to the
# daily total and the fixture takes its argument in the same unit.
BARS_PER_DAY = 288


def candles(symbol, start_ms, end_ms, daily_quote_volume, step_ms=300_000, price=100.0):
    """Candles whose 24h quote volume is daily_quote_volume."""
    per_bar = daily_quote_volume / BARS_PER_DAY
    rows = []
    stamp = start_ms
    while stamp < end_ms:
        rows.append({"symbol": symbol, "interval": "5m", "open_time": stamp,
                     "close_time": stamp + step_ms - 1, "open": price, "high": price * 1.01,
                     "low": price * 0.99, "close": price, "volume": per_bar / price,
                     "quote_volume": per_bar, "trades": 900,
                     "taker_buy_volume": per_bar / price / 2, "is_closed": 1})
        stamp += step_ms
    return rows


def test_symbol_facts_are_measured_to_the_moment_asked_about():
    rows = candles("AUSDT", 0, 100 * DAY, 50_000_000)
    early = symbol_facts(rows, 50 * DAY)
    late = symbol_facts(rows, 90 * DAY)
    # Same contract, same data: only the moment differs, and the age moves with it.
    assert late["age_days"] > early["age_days"]
    assert abs(late["age_days"] - early["age_days"] - 40) < 1
    # 24h of 5m bars, not the whole history.
    assert early["bars"] == 288


def test_a_symbol_that_had_not_listed_yet_is_absent():
    # Look-ahead: a contract listed last month must not appear a year earlier.
    rows = candles("NEWUSDT", 400 * DAY, 500 * DAY, 50_000_000)
    assert symbol_facts(rows, 300 * DAY) is None
    assert symbol_facts(rows, 450 * DAY) is not None


def test_a_symbol_that_stopped_trading_is_absent():
    # Survivorship: a contract delisted mid-window must not carry its later absence, nor
    # be quietly dropped from the earlier period where it did trade.
    rows = candles("DEADUSDT", 0, 200 * DAY, 50_000_000)
    assert symbol_facts(rows, 100 * DAY) is not None
    assert symbol_facts(rows, 400 * DAY) is None


def test_quote_volume_falls_back_to_base_volume_times_price():
    # A database written before the quote-volume column existed would otherwise report
    # zero volume and exclude every symbol.
    rows = candles("AUSDT", 0, 10 * DAY, 50_000_000)
    for row in rows:
        row["quote_volume"] = None
    facts = symbol_facts(rows, 5 * DAY)
    assert facts["quote_volume"] > 0


def test_point_in_time_ranks_by_the_volume_of_that_moment():
    # The mainstream gate wants 730 days of listing age, so the moment asked about has to
    # be past that for the deep symbol to reach the top tier.
    series = {"BIGUSDT": candles("BIGUSDT", 0, 1400 * DAY, 200_000_000),
              "SMALLUSDT": candles("SMALLUSDT", 0, 1400 * DAY, 20_000_000)}
    report = point_in_time(series, 1000 * DAY)
    assert [item["symbol"] for item in report["graded"]] == ["BIGUSDT", "SMALLUSDT"]
    assert report["counts"][TIER_MAINSTREAM] == 1
    assert report["counts"][TIER_SPECULATIVE] == 1


def test_volume_is_read_from_the_window_not_the_history():
    # A symbol that was busy early and quiet late must be judged on the latter.
    rows = candles("FADEDUSDT", 0, 900 * DAY, 200_000_000)
    rows += candles("FADEDUSDT", 900 * DAY + 300_000, 1400 * DAY, 1_000_000)
    early = point_in_time({"FADEDUSDT": rows}, 850 * DAY)["graded"][0]
    late = point_in_time({"FADEDUSDT": rows}, 1200 * DAY)["graded"][0]
    assert early["tier"] == TIER_MAINSTREAM
    assert late["tier"] == TIER_EXCLUDED
    assert late["quote_volume"] < early["quote_volume"] / 100


def test_listing_age_is_flagged_as_a_lower_bound_when_truncated():
    rows = candles("AUSDT", 0, 700 * DAY, 50_000_000)
    # The rows begin at day 500 because that is where the fetch started, so the true
    # listing is not in hand and the age can only be a lower bound.
    truncated = symbol_facts(rows, 600 * DAY, fetch_start_ms=500 * DAY)
    assert truncated is not None
    assert truncated["listed_age_is_lower_bound"] is True
    # Rows that reach back before the moment asked about measure the age properly.
    complete = symbol_facts(rows, 600 * DAY, fetch_start_ms=-DAY)
    assert complete["listed_age_is_lower_bound"] is False


def test_universe_changes_report_both_directions():
    # OLDUSDT trades through day 400 and then stops; at day 300 it is in, at day 750 it is
    # gone. That departure is the survivorship case.
    series = {"OLDUSDT": candles("OLDUSDT", 0, 400 * DAY, 90_000_000),
              "LIVEUSDT": candles("LIVEUSDT", 0, 800 * DAY, 200_000_000),
              "NEWUSDT": candles("NEWUSDT", 700 * DAY, 800 * DAY, 90_000_000)}
    before = point_in_time(series, 300 * DAY)
    after = point_in_time(series, 750 * DAY)
    changes = universe_changes(before, after)
    assert changes["added"] == ["NEWUSDT"]
    assert changes["removed"] == ["OLDUSDT"]
    assert "OLDUSDT" in changes["removed_after_exclusion"]


def test_symbols_at_filters_and_limits():
    series = {"BIGUSDT": candles("BIGUSDT", 0, 1400 * DAY, 200_000_000),
              "SMALLUSDT": candles("SMALLUSDT", 0, 1400 * DAY, 20_000_000)}
    assert symbols_at(series, 1000 * DAY, tier=TIER_MAINSTREAM) == ["BIGUSDT"]
    assert symbols_at(series, 1000 * DAY, limit=1) == ["BIGUSDT"]


def test_series_from_rows_groups_and_filters():
    rows = candles("AUSDT", 0, DAY, 1_000_000) + candles("BUSDT", 0, DAY, 1_000_000)
    grouped = series_from_rows(rows)
    assert sorted(grouped) == ["AUSDT", "BUSDT"]
    assert list(series_from_rows(rows, symbols=["AUSDT"])) == ["AUSDT"]
    assert series_from_rows([]) == {}


def test_candles_range_is_bounded_by_time_and_symbol(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    store.upsert_candles("AUSDT", "5m", candles("AUSDT", 0, 10 * DAY, 1_000_000))
    store.upsert_candles("BUSDT", "5m", candles("BUSDT", 0, 10 * DAY, 1_000_000))
    assert len(store.candles_range(["AUSDT"], 0, 10 * DAY)) == 2880
    assert len(store.candles_range(["AUSDT"], 0, 2 * DAY)) == 576
    assert len(store.candles_range(["AUSDT", "BUSDT"], 0, 10 * DAY)) == 5760
    assert store.candles_range([], 0, 10 * DAY) == []
    # A caller that cannot afford the rows is told so. The statement used to carry a
    # default LIMIT and no page loop, so an over-budget range returned the *oldest* rows
    # and dropped the rest; the assertion here used to be "== [] or True", which passed
    # whichever way it went.
    with pytest.raises(ValueError):
        store.candles_range(["AUSDT"], 0, 10 * DAY, limit=10)
    # Paged to completion: one page must not truncate the window.
    assert len(store.candles_range(["AUSDT", "BUSDT"], 0, 10 * DAY, page=100)) == 5760
    store.close()


def test_the_training_job_fetches_the_trailing_window_it_needs(tmp_path):
    # The reconstruction at the window start needs the 24h *before* it. Fetching from the
    # start itself left that window empty, which marked every symbol as not tradeable and
    # reported an empty universe as a finding.
    store = Store(str(tmp_path / "test.db"))
    now = int(time.time() * 1000)
    origin = now - 500 * DAY
    store.upsert_candles("LIVEUSDT", "5m", candles("LIVEUSDT", origin, now, 200_000_000),
                         now_ms=now)
    # DEADUSDT was trading when the window opened and stopped inside it.
    store.upsert_candles("DEADUSDT", "5m",
                         candles("DEADUSDT", origin, now - 300 * DAY, 90_000_000), now_ms=now)
    # NEWUSDT listed inside the window, so it did not exist at the start.
    store.upsert_candles("NEWUSDT", "5m",
                         candles("NEWUSDT", now - 60 * DAY, now, 90_000_000), now_ms=now)
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    report = runner._point_in_time(["LIVEUSDT", "DEADUSDT", "NEWUSDT"], job.options)
    assert report["status"] == "ok"
    assert report["graded_at_start"], "the trailing window must be in hand"
    assert report["at_start"] != {}
    # Listed after the window opened: the look-ahead case.
    assert report["not_tradeable_at_start"] == ["NEWUSDT"]
    # Present at the start and gone by the end: the survivorship case.
    assert "DEADUSDT" in report["changes"]["removed"]
    assert "DEADUSDT" not in report["not_tradeable_at_start"]
    assert any("lower bound" in note for note in report["notes"])
    store.close()


def test_the_training_job_says_so_when_the_universe_is_empty(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    now = int(time.time() * 1000)
    # Candles exist, but only in the middle of the window: nothing traded in the 24h
    # ending at the window start.
    store.upsert_candles("GAPUSDT", "5m",
                         candles("GAPUSDT", now - 200 * DAY, now - 100 * DAY, 90_000_000),
                         now_ms=now)
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    report = runner._point_in_time(["GAPUSDT"], job.options)
    assert report["status"] == "ok"
    assert report["at_start"] == {}
    assert any("no symbol qualified at the window start" in note for note in report["notes"])
    store.close()


def _settings():
    from app.core.config import Settings

    return Settings()

def test_a_lagging_feed_is_not_reported_as_a_delisting():
    """An ingest that stopped and a market that emptied are the same missing key.

    These candles are what the store looks like when the training tier is refreshed only by
    a training run: its symbols most recently traded long before the wall clock, while the
    data itself is intact. Reconstructed at the wall clock, every one of them reads as
    having stopped trading, and "stopped printing" and "was delisted" are the same absence.
    """
    now = int(time.time() * 1000)
    origin = now - 500 * DAY
    live = candles("LIVEUSDT", origin, now - 234 * DAY, 500_000_000)
    # ... and this one genuinely stopped a week before the data ends.
    gone = candles("GONEUSDT", origin, now - 241 * DAY, 500_000_000)
    series = series_from_rows(live + gone)

    extent = data_horizon(series, 300_000)
    assert extent["horizon"] == max(r["close_time"] for r in live)
    assert extent["per_symbol"]["GONEUSDT"]["last"] < extent["horizon"] - 6 * DAY

    # At the wall clock nothing qualifies: a fact about the ingest, not about the market.
    at_clock = point_in_time(series, now)
    assert at_clock["measurable"] is False and at_clock["graded_count"] == 0
    assert set(at_clock["untraded_reasons"].values()) == {"no_bars_in_window"}

    # At the data horizon it resolves, and separates the two.
    at_horizon = point_in_time(series, extent["horizon"])
    assert at_horizon["graded_count"] == 1 and at_horizon["tradeable"] is True
    assert at_horizon["untraded"] == ["GONEUSDT"]
    assert at_horizon["untraded_reasons"]["GONEUSDT"] == "no_bars_in_window"

def test_every_exclusion_says_which_kind_it_was():
    """A missing symbol is three different findings: unknown, too early, or stopped."""
    now = int(time.time() * 1000)
    by_symbol = {
        # Never collected at all.
        "ABSENTUSDT": [],
        # First bar well after the moment being reconstructed.
        "NEWUSDT": candles("NEWUSDT", now + DAY, now + 10 * DAY, 500_000_000),
        # Trading once, silent across the whole trailing window.
        "DEADUSDT": candles("DEADUSDT", now - 400 * DAY, now - 200 * DAY, 500_000_000),
    }
    report = point_in_time(by_symbol, now)
    assert report["untraded_reasons"] == {"ABSENTUSDT": "no_data",
                                          "NEWUSDT": "not_listed_yet",
                                          "DEADUSDT": "no_bars_in_window"}
    assert report["untraded_count"] == 3 and report["graded_count"] == 0


def test_an_excluded_candidate_set_is_not_an_empty_universe():
    """Every symbol in the age gate is a request too long for the store, not a finding."""
    now = int(time.time() * 1000)
    # The store holds exactly the window that was asked for, so at the window start every
    # symbol is one day old and the age gate excludes the lot.
    start = now - 365 * DAY
    series = series_from_rows(candles("YOUNGUSDT", start - DAY, now, 500_000_000))
    report = point_in_time(series, start, fetch_start_ms=start - DAY)
    # Graded, and put in the age gate: a verdict, but not one that supports building a
    # dataset from it.
    assert report["counts"] == {TIER_EXCLUDED: 1}
    assert report["graded"][0]["listed_age_is_lower_bound"] is True
    # Two different answers, and a caller needs both: the reconstruction ran (measurable),
    # and it found nothing it could trade (not tradeable).
    assert report["measurable"] is True
    assert report["tradeable"] is False

def test_candle_windows_reads_only_what_the_reconstruction_asks_for(tmp_path):
    """A day per moment and the listing time, not the 1.39M rows in between."""
    store = Store(str(tmp_path / "test.db"))
    rows = candles("AUSDT", 0, 10 * DAY, 1_000_000)
    store.upsert_candles("AUSDT", "5m", rows)
    moment = 5 * DAY
    windows = store.candle_windows(["AUSDT"], (moment,), interval="5m")
    entry = windows["AUSDT"]
    assert entry["listed_at"] == rows[0]["close_time"]
    assert entry["last_at"] == rows[-1]["close_time"]
    # The trailing day, not the ten days stored.
    assert len(entry["windows"][moment]) == BARS_PER_DAY
    assert len(store.candle_windows([], (moment,))) == 0
    assert store.candle_windows(["MISSINGUSDT"], (moment,)) == {}
    store.close()


def test_assembling_windows_counts_a_bar_once_and_keeps_the_listing_time():
    """Overlapping windows would otherwise sum the same volume twice."""
    rows = candles("AUSDT", 0, 3 * DAY, 1_000_000)
    # Two windows that overlap: the same bar is offered twice.
    series = series_from_windows(
        [{"AUSDT": {"listed_at": rows[0]["close_time"], "last_at": rows[-1]["close_time"],
                    "windows": {2 * DAY: rows, 2 * DAY + 1000: rows}}}], (2 * DAY,))
    assert len(series["AUSDT"]) == len(rows)
    facts = symbol_facts(series["AUSDT"], 2 * DAY)
    # One day of volume and one day of bars. Counting the overlapping bar twice puts
    # either figure at double, which is exactly what the volume gate reads.
    assert facts["bars"] == BARS_PER_DAY
    assert facts["quote_volume"] == pytest.approx(1_000_000)

    # When the windows do not reach back to the listing, the listing is still known:
    # without the stub the symbol reads as one day old and the age gate drops it.
    tail = rows[BARS_PER_DAY:2 * BARS_PER_DAY]
    narrow = series_from_windows(
        [{"AUSDT": {"listed_at": rows[0]["close_time"], "last_at": rows[-1]["close_time"],
                    "windows": {2 * DAY: tail}}}], (2 * DAY,))
    assert narrow["AUSDT"][0]["close_time"] == rows[0]["close_time"]
    assert symbol_facts(narrow["AUSDT"], 2 * DAY)["listed_at"] == rows[0]["close_time"]


def test_the_reconstruction_verdict_reaches_the_dataset(tmp_path):
    """Computed, stored and logged is not used: the rows have to be cut."""
    store = Store(str(tmp_path / "test.db"))
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    start = int(time.time() * 1000) - 365 * DAY
    job.point_in_time = {"status": "ok", "as_of": start, "measurable": True,
                         "tradeable": True, "at_start": {"speculative": 1},
                         "graded_at_start": [{"symbol": "LIVEUSDT"}]}
    rows = [{"symbol": "LIVEUSDT", "timestamp": start + 1000},
            {"symbol": "LIVEUSDT", "timestamp": start - 1000},
            {"symbol": "NEWUSDT", "timestamp": start + 1000}]
    kept, report = runner._restrict(job, rows)
    assert [row["symbol"] for row in kept] == ["LIVEUSDT"]
    assert report["rows_before"] == 3 and report["rows_after"] == 1
    # The two causes are counted apart: one symbol listed inside the window, one row
    # predates the window that was asked for.
    assert report["rows_dropped_symbol"] == 1
    assert report["rows_dropped_before_start"] == 1
    assert report["dropped_symbols"] == ["NEWUSDT"]
    store.close()


def test_the_job_refuses_to_empty_a_dataset_on_a_measurement_that_failed(tmp_path):
    """An unmeasured universe is not an empty one, and the two need different answers."""
    store = Store(str(tmp_path / "test.db"))
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    start = int(time.time() * 1000) - 365 * DAY
    rows = [{"symbol": "AUSDT", "timestamp": start + 1000}]

    job.point_in_time = None
    assert runner._restrict(job, rows)[1]["reason"] == "no_start_universe"

    # The reconstruction ran and classified nothing at all: the fetch window was empty.
    job.point_in_time = {"status": "ok", "as_of": start, "measurable": False,
                         "graded_at_start": [], "at_start": {}}
    kept, report = runner._restrict(job, rows)
    assert kept == rows and report["reason"] == "start_universe_unmeasured"

    # It classified every candidate and excluded the lot: a verdict, and not one that
    # supports trading on.
    job.point_in_time = {"status": "ok", "as_of": start, "measurable": True,
                         "tradeable": False, "graded_at_start": [],
                         "at_start": {"excluded": 14}}
    kept, report = runner._restrict(job, rows)
    assert kept == rows and report["reason"] == "nothing_tradeable_at_start"
    assert report["at_start"] == {"excluded": 14}
    store.close()


def test_a_window_longer_than_the_store_says_so(tmp_path):
    """Every symbol one day old is a request longer than the store, not an empty market."""
    store = Store(str(tmp_path / "test.db"))
    now = int(time.time() * 1000)
    store.upsert_candles("AUSDT", "5m", candles("AUSDT", now - 366 * DAY, now, 900_000_000),
                         now_ms=now)
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    report = runner._point_in_time(["AUSDT"], job.options)
    job.point_in_time = report          # what _run does before the dataset stage
    assert report["status"] == "ok"
    assert report["diagnosis"] == "request_reaches_before_the_store"
    # A verdict was reached -- the symbol is in the age gate -- and that is not the same
    # as the universe being empty, so the dataset is not cut on the strength of it.
    assert report["measurable"] is True and report["tradeable"] is False
    assert report["at_start"] == {TIER_EXCLUDED: 1}
    assert any("not established" in note for note in report["notes"])
    rows = [{"symbol": "AUSDT", "timestamp": report["as_of"] + 1}]
    kept, restriction = runner._restrict(job, rows)
    assert kept == rows and restriction["reason"] == "nothing_tradeable_at_start"
    store.close()


def test_a_universe_that_resolved_is_reported_as_such(tmp_path):
    """The other direction, so the field is not a constant."""
    store = Store(str(tmp_path / "test.db"))
    now = int(time.time() * 1000)
    # Two years of store against a one-year window: at the window start the symbol is a
    # year old, so the age gate has something real to judge.
    store.upsert_candles("AUSDT", "5m", candles("AUSDT", now - 730 * DAY, now, 900_000_000),
                         now_ms=now)
    runner = TrainingRunner(_settings(), store)
    job = runner.start(TIER_SPECULATIVE, days=365)
    report = runner._point_in_time(["AUSDT"], job.options)
    job.point_in_time = report
    assert report["diagnosis"] == "universe_measured"
    assert report["tradeable"] is True and report["at_start"] != {TIER_EXCLUDED: 1}
    # The verdict now reaches the rows: this symbol is in, and a row from before the
    # window opened is not.
    rows = [{"symbol": "AUSDT", "timestamp": report["as_of"] + 1},
            {"symbol": "AUSDT", "timestamp": report["as_of"] - 1}]
    kept, restriction = runner._restrict(job, rows)
    assert restriction["status"] == "ok" and len(kept) == 1
    assert restriction["rows_dropped_before_start"] == 1
    store.close()

def test_a_missing_trade_count_is_not_a_thin_contract():
    """Every stored candle has NULL here, so this is the case that actually happens."""
    now = int(time.time() * 1000)
    rows = candles("BUSYUSDT", now - 400 * DAY, now, 500_000_000)
    for row in rows:
        row["trades"] = None
    facts = symbol_facts(rows, now)
    assert facts["trades"] is None and facts["trades_measured"] is False
    # The volume is measured and enormous; only the count is missing.
    tier, reasons = classify_one({"symbol": "BUSYUSDT", "quote_volume": facts["quote_volume"],
                                  "trades": facts["trades"], "price": facts["price"],
                                  "age_days": facts["age_days"], "daily_range_pct": None})
    assert tier == TIER_EXCLUDED
    assert reasons == ["unmeasured: trade count"]
    # With the count present it clears the speculative bar, so the verdict really did turn
    # on whether the column was measured rather than on the market.
    facts["trades"] = 900 * BARS_PER_DAY
    tier, _ = classify_one({"symbol": "BUSYUSDT", "quote_volume": facts["quote_volume"],
                            "trades": facts["trades"], "price": facts["price"],
                            "age_days": facts["age_days"], "daily_range_pct": None})
    assert tier == TIER_SPECULATIVE


def test_a_recorded_zero_trade_count_still_excludes():
    """The other direction: an unmeasured count must not become a free pass."""
    now = int(time.time() * 1000)
    rows = candles("QUIETUSDT", now - 400 * DAY, now, 500_000_000)
    for row in rows:
        row["trades"] = 0
    facts = symbol_facts(rows, now)
    assert facts["trades"] == 0 and facts["trades_measured"] is True
    tier, reasons = classify_one({"symbol": "QUIETUSDT", "quote_volume": facts["quote_volume"],
                                  "trades": 0, "price": facts["price"],
                                  "age_days": facts["age_days"], "daily_range_pct": None})
    assert tier == TIER_EXCLUDED and reasons == ["too few trades to fill"]


def test_quote_volume_says_whether_it_was_read_or_derived():
    """A derived volume is a good approximation, but it is not the exchange's number."""
    now = int(time.time() * 1000)
    rows = candles("AUSDT", now - 10 * DAY, now, 50_000_000)
    assert symbol_facts(rows, now)["quote_volume_source"] == "column"
    for row in rows:
        row["quote_volume"] = None
    assert symbol_facts(rows, now)["quote_volume_source"] == "derived"


def test_an_unread_column_is_reported_as_such_not_as_a_thin_market(tmp_path):
    """The verdict that says re-ingest, not the one that says trade a smaller set."""
    store = Store(str(tmp_path / "test.db"))
    runner = TrainingRunner(_settings(), store)
    # No trade count column: measured volume, unmeasured activity. The history is long
    # enough that the age gate passes, so the count is the only thing standing in the way.
    now = int(time.time() * 1000)
    rows = candles("BUSYUSDT", now - 400 * DAY, now, 900_000_000)
    for row in rows:
        row["trades"] = None
    store.upsert_candles("BUSYUSDT", "5m", rows, now_ms=now)
    job = runner.start(TIER_SPECULATIVE, days=100)
    report = runner._point_in_time(["BUSYUSDT"], job.options)
    assert report["diagnosis"] == "criteria_unmeasured"
    assert report["unmeasured_criteria_at_start"] == {"trade count": 1}
    assert any("empty column" in note for note in report["notes"])
    job.point_in_time = report
    kept, restriction = runner._restrict(job, [{"symbol": "BUSYUSDT", "timestamp": 1}])
    assert restriction["reason"] == report["diagnosis"]
    assert restriction["unmeasured_criteria"] == {"trade count": 1}
    store.close()

