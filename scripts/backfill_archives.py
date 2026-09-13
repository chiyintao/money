"""Run the P1 #12/#13 backfill in the background.

The archive CDN returns 404 for minutes at a time for files that exist, so a backfill is
a long-running job that must survive those windows rather than fail with them. This drives
the per-file work, retries across windows, and records what it could not get so the next
run knows where to resume.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.market import archive, backfill  # noqa: E402
from app.storage.storage import Store  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]
MONTHS = ["2025-%02d" % m for m in range(9, 13)] + ["2026-%02d" % m for m in range(1, 10)]
CACHE = "data/archive_cache"
# One state file per stage, not per job. The two stages have different natural rates (155
# monthly files vs ~4,400 daily ones) and are meant to run concurrently, but a shared file
# means each process writes the other's progress away: measured, a metrics run completed 366
# files while the klines process held the file and left `metrics_done` empty, so every
# restart redid all of them.
STATE_DIR = Path("data/backfill_state")


def state_path(stage):
    return STATE_DIR / ("%s.json" % stage)


def load_state(stage):
    path = state_path(stage)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("done", [])
            data.setdefault("unresolved", {})
            return data
        except ValueError:
            pass
    return {"done": [], "unresolved": {}}


def save_state(stage, state):
    """Write atomically, so a killed run cannot leave a truncated state file.

    A half-written JSON file reads back as empty, which would silently discard every
    completed file and re-download the whole range on the next run.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(stage)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run_klines(store, state, passes=6):
    """Fill the kline order-flow columns, retrying across CDN failure windows.

    Several passes rather than one: the CDN answers 404 for minutes at a time, so a single
    sweep leaves holes that are purely a matter of timing. Each pass skips whatever the
    previous one finished, and stops early once a pass resolves nothing new.
    """
    done = set(state["done"])
    filled_total = 0
    pending = [(s, m) for s in SYMBOLS for m in MONTHS
               if "%s:%s" % (s, m) not in done]
    for attempt in range(passes):
        if not pending:
            break
        before = len(pending)
        print("--- pass %d: %d archives pending ---" % (attempt + 1, before), flush=True)
        filled_total += _sweep_klines(store, state, pending)
        done = set(state["done"])
        pending = [(s, m) for s, m in pending if "%s:%s" % (s, m) not in done]
        if len(pending) == before:
            # Nothing moved: the remaining files are absent or the CDN is closed to us.
            # Another immediate pass would just repeat the same failures.
            print("--- no progress; %d left unresolved ---" % len(pending), flush=True)
            break
    return filled_total


def _sweep_klines(store, state, pending):
    """One pass over the pending archives, saving progress after each file.

    Progress is written per file rather than at the end, because a run interrupted by a
    failure window must resume without redoing work it already completed.
    """
    done = set(state["done"])
    filled_total = 0
    for symbol, month in pending:
        key = "%s:%s" % (symbol, month)
        if key in done:
            continue
        url = archive.archive_url("futures/um", "monthly", "klines", symbol,
                                  archive.klines_filename(symbol, "5m", month),
                                  interval="5m")
        try:
            payload = archive.fetch(url, cache_dir=CACHE)
        except archive.ArchiveError as exc:
            print("NET  %s %s" % (key, str(exc)[:60]), flush=True)
            state["unresolved"][key] = "network"
            save_state("klines", state)
            continue
        if payload is None:
            # NOT marked done. This CDN was measured answering 404 for a file that curl
            # fetched with 200 seconds earlier, so one absence answer is not evidence -- and
            # recording it as finished would make the gap permanent. It stays unresolved and
            # is retried. A file that is genuinely absent stays unresolved forever, which
            # costs one cheap request per pass and cannot lose data.
            print("MISS %s (unresolved)" % key, flush=True)
            state["unresolved"][key] = "absent"
            save_state("klines", state)
            continue
        header, rows = archive.read_csv(payload)
        bars = archive.parse_kline_rows(header, rows)
        filled = _fill(store, symbol, bars)
        filled_total += filled
        print("OK   %s bars=%d filled=%d" % (key, len(bars), filled), flush=True)
        done.add(key)
        state["done"] = sorted(done)
        state["unresolved"].pop(key, None)
        save_state("klines", state)
    return filled_total


def _fill(store, symbol, bars):
    update = ("UPDATE candles SET %s=? WHERE symbol=? AND interval=? AND open_time=? "
              "AND %s IS NULL")
    filled = 0
    for bar in bars:
        for column in ("taker_buy_volume", "quote_volume", "trades"):
            if column not in bar:
                continue
            filled += max(0, store.db.execute(update % (column, column),
                          (bar[column], symbol, "5m", bar["open_time"])).rowcount)
    store.db.commit()
    return filled


def run_metrics(store, state, passes=3):
    """Fill open-interest and positioning history from the daily metrics archives.

    Daily files only: Binance publishes no monthly metrics archive, so a year across twelve
    symbols is about 4,400 small downloads. Each file holds ~289 five-minute samples, which
    is what the positioning features (oi_*, long_short_ratio_z, smart_retail_gap,
    global_long_short_ratio) are built from -- eight more names that were previously
    unbuildable across the history.
    """
    done = set(state["done"])
    pending = [(s, d) for s in SYMBOLS for d in metric_days()
               if "%s:%s" % (s, d) not in done]
    written_total = 0
    for attempt in range(passes):
        if not pending:
            break
        before = len(pending)
        print("--- metrics pass %d: %d files pending ---" % (attempt + 1, before),
              flush=True)
        written_total += _sweep_metrics(store, state, pending)
        done = set(state["done"])
        pending = [(s, d) for s, d in pending if "%s:%s" % (s, d) not in done]
        if len(pending) == before:
            print("--- no progress; %d metrics files unresolved ---" % len(pending),
                  flush=True)
            break
    return written_total


def metric_days():
    """Every day the database has candles for, which is the range worth backfilling."""
    import datetime as _dt
    oldest, newest = Store("data").db.execute(
        "SELECT MIN(open_time), MAX(open_time) FROM candles").fetchone()
    if not oldest:
        return []
    start = _dt.datetime.utcfromtimestamp(oldest / 1000).date()
    end = _dt.datetime.utcfromtimestamp(newest / 1000).date()
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor += _dt.timedelta(days=1)
    return days


def _sweep_metrics(store, state, pending):
    done = set(state["done"])
    written_total = 0
    for symbol, day in pending:
        key = "%s:%s" % (symbol, day)
        if key in done:
            continue
        url = archive.archive_url("futures/um", "daily", "metrics", symbol,
                                  archive.metrics_filename(symbol, day))
        try:
            payload = archive.fetch(url, cache_dir=CACHE)
        except archive.ArchiveError as exc:
            print("NET  %s %s" % (key, str(exc)[:60]), flush=True)
            continue
        if payload is None:
            print("MISS %s (unresolved)" % key, flush=True)
            continue
        header, rows = archive.read_csv(payload)
        records = archive.parse_metrics_rows(header, rows)
        written = backfill._write_derivatives(store, symbol, records)
        written_total += written
        print("OK   %s samples=%d written=%d" % (key, len(records), written), flush=True)
        done.add(key)
        state["done"] = sorted(done)
        save_state("metrics", state)
    return written_total


def main():
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "klines"
    store = Store("data")
    state = load_state(stage)
    t0 = time.time()
    if stage in ("klines", "all"):
        print("coverage before: %s" % json.dumps(backfill.column_coverage(store),
                                                 sort_keys=True), flush=True)
        filled = run_klines(store, state)
        print("klines: %d values filled in %.0fs" % (filled, time.time() - t0),
              flush=True)
        print("coverage after : %s" % json.dumps(backfill.column_coverage(store),
                                                 sort_keys=True), flush=True)
    if stage in ("metrics", "all"):
        before = store.db.execute(
            "SELECT COUNT(*) FROM derivatives_detail").fetchone()[0]
        written = run_metrics(store, state)
        after = store.db.execute(
            "SELECT COUNT(*) FROM derivatives_detail").fetchone()[0]
        print("metrics: %d rows added (%d -> %d) in %.0fs" % (
            written, before, after, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()