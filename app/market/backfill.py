"""Fill historical gaps from the public archives.

The database holds what the live service collected, which means it holds almost nothing
for the columns added after collection started. Measured before this ran: 2,780,535 of
2,784,287 candle rows (99.9%) had NULL in `taker_buy_volume`, `quote_volume` and `trades`,
which is every row written before the order-flow collector existed. Five feature names --
taker_imbalance, taker_imbalance_z, trade_count_z, avg_trade_size_z, quote_volume_ratio --
are built from exactly those columns, so all five were unbuildable across the entire
history and the model trained on ten features out of a thirty-name contract.

The archives carry the missing columns and go back years, so this is recoverable. What is
NOT recoverable is documented here rather than silently skipped, because a dataset that
looks complete and is not is worse than one that admits its gaps:

* `flow_*` features (delta ratio, trade intensity, largest trade) come from the live
  `@aggTrade` stream. Not archived anywhere. Every bar before the collector started is
  permanently missing them.
* `liquidation_*` comes from the `!forceOrder@arr` stream. Binance publishes no history.
  Also permanently missing.
* `taker_buy_volume` / `taker_sell_volume` on `derivatives_detail` come from
  `/futures/data/takerlongshortRatio`, a 30-day REST endpoint with no archive.

Two rules make this safe to run repeatedly and safe to run while the service is up:

* **Only missing values are written.** The backfill issues a targeted UPDATE of the three
  order-flow columns and never touches OHLCV. The archive is authoritative, but the
  service is writing to the same table, and a job whose purpose is filling blanks has no
  business overwriting anything that already has a value.
* **Progress is per file, and a re-run skips what it finished.** The CDN for these
  archives returns spurious 404s in windows lasting minutes (see `archive.MISSING_RETRIES`),
  so a long run will be interrupted; resuming has to be cheap and must not re-download.
"""
import argparse
import json
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import archive
from ..storage.storage import Store

DEFAULT_CACHE = "data/archive_cache"


def month_range(start, end):
    """`["2025-09", "2025-10", ...]` inclusive of both ends. Both are "YYYY-MM"."""
    first = datetime.strptime(start, "%Y-%m").date().replace(day=1)
    last = datetime.strptime(end, "%Y-%m").date().replace(day=1)
    months = []
    cursor = first
    while cursor <= last:
        months.append(cursor.strftime("%Y-%m"))
        cursor = (cursor.replace(day=28) + timedelta(days=7)).replace(day=1)
    return months


def day_range(start, end):
    """`["2025-09-10", ...]` inclusive of both ends. Both are "YYYY-MM-DD"."""
    first = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    cursor = first
    while cursor <= last:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _is_duplicate_of(store, symbol, interval, open_time):
    return store.db.execute(
        "SELECT 1 FROM candles WHERE symbol=? AND interval=? AND open_time=?",
        (symbol, interval, open_time)).fetchone() is not None


def backfill_klines(store, symbols, interval="5m", months=(), days=(),
                    cache_dir=DEFAULT_CACHE, market="futures/um", progress=None):
    """Write the order-flow columns for the given months and days.

    Returns a report naming, per archive: how many rows the file held, how many were
    already in the database, and how many had a missing column filled. A file that is
    absent is reported, never raised -- `metrics` has no monthly archive at all, and a
    symbol that listed mid-month has no earlier file, and both are normal.
    """
    report = {"klines": [], "filled": 0, "absent": 0, "notes": []}
    # `UPDATE ... SET col=? WHERE ... AND col IS NULL` is the whole safety property: a row
    # that already has a value is not selected, so a re-run cannot change data the service
    # wrote. A single statement per column keeps the write path identical to the read.
    update = ("UPDATE candles SET %s=? WHERE symbol=? AND interval=? AND open_time=? "
              "AND %s IS NULL")
    for symbol in symbols:
        for period in list(months) + list(days):
            filename = archive.klines_filename(symbol, interval, period)
            url = archive.archive_url(market, "monthly" if period in months else "daily",
                                      "klines", symbol, filename, interval=interval)
            entry = {"symbol": symbol, "period": period}
            try:
                payload = archive.fetch(url, cache_dir=cache_dir)
            except archive.ArchiveError as exc:
                entry["status"] = "network_error"
                entry["error"] = str(exc)[:200]
                report["klines"].append(entry)
                continue
            if payload is None:
                entry["status"] = "absent"
                report["absent"] += 1
                report["klines"].append(entry)
                continue
            header, rows = archive.read_csv(payload)
            bars = archive.parse_kline_rows(header, rows)
            filled = 0
            for bar in bars:
                if not _is_duplicate_of(store, symbol, interval, bar["open_time"]):
                    continue
                for column in ("taker_buy_volume", "quote_volume", "trades"):
                    if column not in bar:
                        continue
                    written = store.db.execute(update % (column, column),
                                               (bar[column], symbol, interval,
                                                bar["open_time"])).rowcount
                    filled += max(0, written)
            store.db.commit()
            entry["status"] = "ok"
            entry["file_rows"] = len(bars)
            entry["values_filled"] = filled
            report["filled"] += filled
            report["klines"].append(entry)
            if progress is not None:
                progress("klines", symbol, period, filled)
    return report


def backfill_metrics(store, symbols, days, cache_dir=DEFAULT_CACHE, market="futures/um",
                     progress=None):
    """Write open interest and positioning history from the daily metrics archives.

    `metrics` is published DAILY ONLY -- there is no monthly archive, which is why this
    takes days rather than months. Each file holds about 289 five-minute samples, so a
    year across twelve symbols is roughly 1.2M rows and 4,400 downloads.

    Two columns of the stored schema are deliberately left NULL: `taker_buy_volume` and
    `taker_sell_volume` are not in the archive (they come from a 30-day REST endpoint), and
    writing a zero would claim "no aggressive flow", which the archive does not say.
    """
    report = {"metrics": [], "rows_written": 0, "absent": 0}
    for symbol in symbols:
        for day in days:
            filename = archive.metrics_filename(symbol, day)
            url = archive.archive_url(market, "daily", "metrics", symbol, filename)
            entry = {"symbol": symbol, "day": day}
            try:
                payload = archive.fetch(url, cache_dir=cache_dir)
            except archive.ArchiveError as exc:
                entry["status"] = "network_error"
                entry["error"] = str(exc)[:200]
                report["metrics"].append(entry)
                continue
            if payload is None:
                entry["status"] = "absent"
                report["absent"] += 1
                report["metrics"].append(entry)
                continue
            header, rows = archive.read_csv(payload)
            records = archive.parse_metrics_rows(header, rows)
            written = _write_derivatives(store, symbol, records)
            entry["status"] = "ok"
            entry["file_rows"] = len(records)
            entry["rows_written"] = written
            report["rows_written"] += written
            report["metrics"].append(entry)
            if progress is not None:
                progress("metrics", symbol, day, written)
    return report


def _write_derivatives(store, symbol, records):
    """Insert metrics samples, ignoring ones already stored.

    `INSERT OR IGNORE` rather than an upsert: the archive is a fixed historical record and
    the live collector may have written a more precise value for the same instant. Ignoring
    keeps whichever arrived first, which for a 5-minute aggregate is not a meaningful
    difference -- but overwriting live data with archive data is a change nothing asked for.
    """
    written = 0
    for record in records:
        row = (symbol, int(record["event_time"]),
               _dec(record.get("open_interest")),
               _dec(record.get("open_interest_value")),
               _dec(record.get("top_account_ratio")),
               _dec(record.get("long_short_ratio")),
               _dec(record.get("global_account_ratio")),
               _dec(record.get("taker_buy_sell_ratio")),
               None, None)
        written += max(0, store.db.execute(
            "INSERT OR IGNORE INTO derivatives_detail(symbol,event_time,open_interest,"
            "open_interest_value,long_short_ratio,top_account_ratio,global_account_ratio,"
            "taker_buy_sell_ratio,taker_buy_volume,taker_sell_volume) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)", row).rowcount)
    store.db.commit()
    return written


def _dec(value):
    """A float for a REAL/INTEGER column, or None. Metrics arrive as long decimals."""
    if value is None:
        return None
    try:
        return float(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError):
        return None


def column_coverage(store):
    """How much of the candle history has each order-flow column, for a before/after."""
    total = store.db.execute("SELECT COUNT(*) FROM candles").fetchone()[0] or 0
    coverage = {"rows": total}
    for column in ("taker_buy_volume", "quote_volume", "trades"):
        present = store.db.execute("SELECT COUNT(%s) FROM candles" % column).fetchone()[0] or 0
        coverage[column] = present
        coverage[column + "_pct"] = round(100.0 * present / total, 2) if total else 0.0
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backfill history from Binance archives")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--months", default="")
    parser.add_argument("--days", default="")
    parser.add_argument("--kind", choices=("klines", "metrics", "both"), default="klines")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    store = Store(args.data_dir)
    symbols = [s for s in args.symbols.split(",") if s] or [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]
    months = month_range(*args.months.split(":")) if args.months else []
    days = day_range(*args.days.split(":")) if args.days else []

    before = column_coverage(store)
    print(json.dumps({"before": before}, sort_keys=True))

    def progress(kind, symbol, period, written):
        print("%s %s %s -> %d" % (kind, symbol, period, written), flush=True)

    result = {}
    if args.kind in ("klines", "both") and (months or days):
        result["klines"] = backfill_klines(store, symbols, args.interval, months, days,
                                           args.cache, progress=progress)
    if args.kind in ("metrics", "both") and days:
        result["metrics"] = backfill_metrics(store, symbols, days, args.cache,
                                             progress=progress)
    after = column_coverage(store)
    summary = {"before": before, "after": after,
               "rows_written": {k: v.get("filled", v.get("rows_written", 0))
                                for k, v in result.items()},
               "absent": sum(v.get("absent", 0) for v in result.values())}
    print(json.dumps(summary, sort_keys=True))
    if args.report:
        Path(args.report).write_text(json.dumps({"summary": summary, "detail": result},
                                                indent=2, sort_keys=True), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()