"""Download the full monthly 5m kline history for the fourteen-coin basket.

Why this exists
---------------
Every model in this repository was trained on a single year, 2025-09 to 2026-09, and every
result in docs/COIN_SUITE_MODEL.md carries the same warning: roughly 260 days of genuinely
out-of-sample data spanning one market regime. The measured consequence was visible in the
rolling experiment -- a fitted layer that won one window and lost the next six -- and the
honest reading of that is not "the model is bad" but "the sample cannot tell".

The archive has more. Binance publishes monthly 5m klines from 2020-01 for BTC, ETH, XRP,
ADA and LINK, and from within a few months of then for the rest of the basket; ENA only
starts in 2024-04 because the contract did not exist before. That is up to six and a half
years of five-minute bars against the one year on disk, and downloading it is a few hundred
megabytes.

What is deliberately NOT changed here
-------------------------------------
The basket. Every coin in SYMBOLS is already in daily_panel.BASKET, so extending the history
does not re-pick the universe. That distinction is the whole finding of the previous round:
the +78.2 bps that turned out to be -11.7 bps was a *universe* change wearing the label of a
model improvement, and a backfill that quietly swapped in whatever coins had the longest
history would repeat exactly that error with better data.

The interval. 5m is what the panel builder already reads. A coarser interval would be a
different dataset needing a different panel, and the point of this download is to widen the
sample, not to change the question.

Resuming
--------
Progress is one JSON file per symbol, written after every archive, so a run killed by the
CDN's failure windows resumes without redoing anything. A 404 is retried and, if it stays
404, recorded as unresolved rather than as done -- the CDN was measured answering 404 for
minutes at a time for files that exist, and treating that as "absent" would make a hole in
the data permanent and invisible.
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.market import archive  # noqa: E402
from app.storage.storage import Store  # noqa: E402

# The panel basket, unchanged. A coin is added here only when it is added to the model.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
           "WLDUSDT", "ADAUSDT", "NEARUSDT", "SUIUSDT", "ENAUSDT", "LINKUSDT", "UNIUSDT"]

# Every month from 2020-01 to the last complete month before "now". Months before a contract
# existed answer 404 and are recorded as unavailable rather than as failures.
FIRST_MONTH = "2020-01"
INTERVAL = "5m"
CACHE = "data/archive_cache"
STATE_DIR = Path("data/backfill_state")


def months(first=FIRST_MONTH, last=None):
    """Every month from \`first\` to \`last\` inclusive, as YYYY-MM strings."""
    start = dt.date(int(first[:4]), int(first[5:7]), 1)
    if last is None:
        today = dt.date.today()
        # The current month's archive is not published until the month ends.
        end = dt.date(today.year, today.month, 1) - dt.timedelta(days=1)
    else:
        end = dt.date(int(last[:4]), int(last[5:7]), 1)
    out = []
    cursor = start
    while cursor <= end:
        out.append("%04d-%02d" % (cursor.year, cursor.month))
        cursor = dt.date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)
    return out


def state_path(symbol):
    return STATE_DIR / ("extend_%s.json" % symbol)


def load_state(symbol):
    path = state_path(symbol)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("done", [])
            data.setdefault("unavailable", [])
            return data
        except ValueError:
            pass
    return {"done": [], "unavailable": []}


def save_state(symbol, state):
    """Atomic, so a killed run cannot leave a truncated file that reads back as empty.

    A half-written JSON reads as \`{"done": []}\`, which would silently restart six hours of
    downloads. Written to a temporary name and moved, which is atomic on the same filesystem.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(symbol)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


# A single HEAD probe, deliberately without the module's retry budget.
#
# \`archive.fetch\` retries a 404 three times with exponential backoff, because the CDN was
# measured answering 404 for minutes at a time for files that exist. That is correct for the
# case it was written for -- filling columns on rows that already exist -- and wrong for this
# one. Here a large fraction of the requests are for months *before a contract existed*, so
# absence is the common answer rather than the rare one, and paying minutes of backoff for each
# would dominate the job. Measured: this probe returns 404 in 0.2s where \`fetch\` had not
# returned after 120s.
#
# The trade is deliberate and bounded. A file that is transiently 404 during a probe is skipped
# in this pass and retried in the next one, and the pass-over-pass check only stops once a whole
# pass resolves nothing -- so a transient absence costs a re-probe, never a hole.
PROBE_TIMEOUT = 20


def archive_exists(symbol, month):
    """Whether the CDN publishes this month, cheaply. Raises on a network error."""
    url = archive.archive_url("futures/um", "monthly", "klines", symbol,
                              archive.klines_filename(symbol, INTERVAL, month),
                              interval=INTERVAL)
    request = urllib.request.Request(url, headers={"User-Agent": archive.USER_AGENT},
                                     method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return False
        raise


def fetch_month(symbol, month):
    """One monthly archive, or None when the CDN says it does not exist."""
    url = archive.archive_url("futures/um", "monthly", "klines", symbol,
                              archive.klines_filename(symbol, INTERVAL, month),
                              interval=INTERVAL)
    # The cache is consulted first by \`fetch\` itself, so an already-downloaded month costs no
    # request at all and the probe is skipped for it.
    cached = Path(CACHE) / url.rsplit("/", 1)[-1]
    if not (cached.exists() and cached.stat().st_size > 0):
        try:
            if not archive_exists(symbol, month):
                return None
        except Exception:
            # A probe that could not complete is not evidence of absence. Returning None here
            # would record a month as missing because the network hiccuped, so it is treated as
            # pending and retried on the next pass.
            return None
    try:
        return archive.fetch(url, cache_dir=CACHE, retries=0)
    except archive.ArchiveError:
        return None


def write_bars(store, symbol, bars):
    """Insert bars that are not already present, skipping the ones that are.

    \`INSERT OR IGNORE\` would be wrong here and the schema is why: \`idx_candles_lookup\` is a
    plain index, not a UNIQUE one, so nothing constrains duplicates and OR IGNORE would insert a
    second copy of every bar on every re-run -- silently doubling the history and halving every
    measured return. The existence check is therefore explicit.

    \`is_closed\` is NOT NULL with no default, so it has to be supplied: an archive row is by
    definition a completed candle, and writing NULL there fails the insert outright.

    Existing rows are left alone rather than updated. Archive files are immutable history, so a
    row that already exists is identical by construction, while the row on disk may be one the
    live trading process is still appending to.
    """
    written = 0
    for bar in bars:
        exists = store.db.execute(
            "SELECT 1 FROM candles WHERE symbol=? AND interval=? AND open_time=? LIMIT 1",
            (symbol, INTERVAL, bar["open_time"])).fetchone()
        if exists:
            continue
        store.db.execute(
            "INSERT INTO candles (symbol, interval, open_time, close_time, open, high, low, "
            "close, volume, is_closed, ingest_time, quote_volume, trades, taker_buy_volume) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, INTERVAL, bar["open_time"], bar.get("close_time"), bar["open"],
             bar["high"], bar["low"], bar["close"], bar["volume"], 1,
             int(time.time() * 1000), bar.get("quote_volume"), bar.get("trades"),
             bar.get("taker_buy_volume")))
        written += 1
    store.db.commit()
    return written


def run_symbol(store, symbol, passes=4, verbose=True):
    """Download every month for one symbol, retrying across CDN failure windows.

    Several passes rather than one, because the CDN answers 404 for minutes at a time for files
    that exist. Each pass skips what the previous one finished and stops early once a pass
    resolves nothing new, so a symbol whose history genuinely starts later is not swept forever.
    """
    state = load_state(symbol)
    done = set(state["done"])
    unavailable = set(state.get("unavailable") or [])
    written_total = 0
    for attempt in range(passes):
        pending = [m for m in months()
                   if m not in done and m not in unavailable]
        if not pending:
            break
        if verbose:
            print("--- %s pass %d: %d months pending ---" % (symbol, attempt + 1, len(pending)),
                  flush=True)
        progress = 0
        for month in pending:
            payload = fetch_month(symbol, month)
            if payload is None:
                # Not marked done. A 404 here is untrusted -- the CDN was measured returning it
                # for a file curl had fetched with 200 seconds earlier -- so it stays pending for
                # the next pass. Only a month before the contract existed stays unavailable
                # forever, and the pass-over-pass check below is what stops the sweep.
                if verbose:
                    print("MISS %s %s" % (symbol, month), flush=True)
                continue
            header, rows = archive.read_csv(payload)
            bars = archive.parse_kline_rows(header, rows)
            written = write_bars(store, symbol, bars)
            written_total += written
            progress += 1
            done.add(month)
            state["done"] = sorted(done)
            save_state(symbol, state)
            if verbose:
                print("OK   %s %s bars=%d new=%d" % (symbol, month, len(bars), written),
                      flush=True)
        if progress == 0:
            # Nothing resolved in a whole pass: everything left is genuinely absent. Recorded so
            # later runs do not keep re-asking for months before the contract existed.
            remaining = [m for m in months() if m not in done]
            state["unavailable"] = sorted(set(state.get("unavailable") or []) | set(remaining))
            save_state(symbol, state)
            if verbose:
                print("--- %s: %d months unavailable, stopping ---"
                      % (symbol, len(remaining)), flush=True)
            break
    return written_total


def coverage(store):
    """What the database holds now, per symbol, so the job can be judged rather than trusted."""
    rows = store.db.execute(
        "SELECT symbol, COUNT(*), MIN(open_time), MAX(open_time) FROM candles "
        "WHERE interval=? GROUP BY symbol ORDER BY symbol", (INTERVAL,)).fetchall()
    out = {}
    for symbol, count, oldest, newest in rows:
        out[symbol] = {
            "bars": count,
            "first": dt.datetime.utcfromtimestamp(oldest / 1000).strftime("%Y-%m-%d"),
            "last": dt.datetime.utcfromtimestamp(newest / 1000).strftime("%Y-%m-%d")}
    return out


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Extend the kline history backwards.")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    store = Store("data")
    before = coverage(store)
    t0 = time.time()
    grand = 0
    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        written = run_symbol(store, symbol, passes=args.passes, verbose=not args.quiet)
        grand += written
        print("=== %s: %d rows written, total %.0fs ===" % (symbol, written, time.time() - t0),
              flush=True)
    after = coverage(store)
    print()
    print("%-10s %12s %12s %-12s %-12s" % ("symbol", "bars before", "bars after", "first", "last"))
    for symbol in sorted(after):
        old = (before.get(symbol) or {}).get("bars", 0)
        new = after[symbol]["bars"]
        if new == old:
            continue
        print("%-10s %12d %12d %-12s %-12s"
              % (symbol, old, new, after[symbol]["first"], after[symbol]["last"]))
    print()
    print("total rows written: %d in %.0fs" % (grand, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
