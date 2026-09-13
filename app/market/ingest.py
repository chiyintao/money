"""Historical kline ingestion for multi-symbol research datasets.

The live loop only ever sees a few hundred candles, which is far too little to train
on. This module pages backwards through the exchange kline endpoint, writes bars into
the candles table, and reports coverage so a dataset is never built from a gap.

Ingestion is resumable: a re-run continues from the newest stored bar instead of
refetching everything, so an interrupted download costs only the last page.
"""
import argparse
import asyncio
import json
import time

import aiohttp

from ..core.config import Settings
from ..storage.storage import Store

# Re-exported from config so there is exactly one interval table in the codebase.
# Two copies would eventually disagree, and the failure is a silent unit error.
from ..core.config import INTERVAL_MS  # noqa: E402  (kept next to its consumers below)

MAX_LIMIT = 1500


def interval_ms(interval):
    if interval not in INTERVAL_MS:
        raise ValueError("unsupported_interval:" + str(interval))
    return INTERVAL_MS[interval]


def parse_klines(payload, now_ms=None):
    """Convert a raw kline array into storable candle dicts.

    Bars the exchange has not closed yet are marked closed=False so the dataset
    builder can exclude them without guessing.
    """
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows = []
    for item in payload or []:
        try:
            open_time, close_time = int(item[0]), int(item[6])
            # Fields 7, 8 and 9 are quote volume, trade count and taker buy base volume.
            # They have always been in the response and were always discarded, which is why
            # the cheapest order-flow features available (taker imbalance, trade intensity)
            # were missing from a dataset built from an endpoint that already returned them.
            rows.append({"open_time": open_time, "close_time": close_time,
                         "open": float(item[1]), "high": float(item[2]),
                         "low": float(item[3]), "close": float(item[4]),
                         "volume": float(item[5]),
                         "quote_volume": _number(item, 7),
                         "trades": _count(item, 8),
                         "taker_buy_volume": _number(item, 9),
                         "is_closed": int(close_time <= now_ms)})
        except (IndexError, TypeError, ValueError):
            continue
    return rows


def _number(item, index):
    """The field, or None when the response did not carry it.

    Not 0.0. A zero is a measurement: a bar can genuinely have zero taker buying. A
    response that stopped short of field 9 did not measure anything, and the two are
    indistinguishable once both are written as 0.0 -- which is how every one of the 2.78M
    stored candles came to report a trade count of zero, and how the tier gate came to
    exclude the entire universe for "too few trades to fill".
    """
    try:
        value = float(item[index])
    except (IndexError, TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _count(item, index):
    try:
        return int(float(item[index]))
    except (IndexError, TypeError, ValueError):
        return None


async def fetch_page(session, base_url, symbol, interval, start_time, limit=MAX_LIMIT,
                     attempts=5):
    """Fetch one kline page, backing off on throttling and transient failures."""
    url = base_url.rstrip("/") + "/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = int(start_time)
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            async with session.get(url, params=params, timeout=60) as response:
                if response.status in (418, 429):
                    retry_after = float(response.headers.get("Retry-After", delay))
                    await asyncio.sleep(max(delay, retry_after))
                    delay *= 2
                    continue
                if response.status >= 500:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if response.status != 200:
                    body = await response.text()
                    return None, "http_%d:%s" % (response.status, body[:120])
                return await response.json(), None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == attempts:
                return None, type(exc).__name__ + ":" + str(exc)[:80]
            await asyncio.sleep(delay)
            delay *= 2
    return None, "exhausted_retries"


def stored_span(store, symbol, interval):
    """Oldest and newest stored open_time, or (None, None) when the symbol is empty."""
    row = store.db.execute(
        "SELECT MIN(open_time) AS oldest, MAX(open_time) AS newest "
        "FROM candles WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
    if not row or row["newest"] is None:
        return None, None
    return int(row["oldest"]), int(row["newest"])


def interior_holes(store, symbol, interval, oldest, newest, step):
    """Gaps strictly between stored bars.

    Looking only at the oldest and newest bar misses an interrupted download that left
    a hole in the middle, and training on a series with holes silently corrupts the
    forward labels that span them.
    """
    stamps = [int(row["open_time"]) for row in store.db.execute(
        "SELECT open_time FROM candles WHERE symbol=? AND interval=? AND open_time BETWEEN ? AND ? "
        "ORDER BY open_time", (symbol, interval, oldest, newest))]
    holes = []
    for previous, current in zip(stamps, stamps[1:]):
        if current - previous > step:
            holes.append((previous + step, current))
    return holes


def missing_ranges(store, symbol, interval, start_ms, end_ms, max_ranges=64):
    """Ranges still needed to cover [start_ms, end_ms].

    Resuming forward from the newest bar alone is not enough: a symbol that already
    holds recent bars would look 'covered' while the whole earlier window is missing.
    The prefix before the oldest bar, every interior hole, and the suffix after the
    newest bar are all returned.
    """
    step = interval_ms(interval)
    oldest, newest = stored_span(store, symbol, interval)
    if oldest is None:
        return [(int(start_ms), int(end_ms))]
    ranges = []
    if oldest > start_ms:
        ranges.append((int(start_ms), oldest))
    ranges.extend(interior_holes(store, symbol, interval, oldest, newest, step))
    if newest + step < end_ms:
        ranges.append((newest + step, int(end_ms)))
    return ranges[:max_ranges]


async def ingest_symbol(session, base_url, store, symbol, interval, start_ms, end_ms,
                        limit=MAX_LIMIT, verbose=True, on_progress=None,
                        progress_every=10):
    """Page forward from start_ms to end_ms, writing each page as it arrives."""
    step = interval_ms(interval)
    cursor = int(start_ms)
    written = 0
    pages = 0
    errors = []
    while cursor < end_ms:
        payload, error = await fetch_page(session, base_url, symbol, interval, cursor, limit)
        if error:
            errors.append(error)
            break
        rows = parse_klines(payload)
        if not rows:
            break
        written += store.upsert_candles(symbol, interval, rows)
        pages += 1
        last = rows[-1]["open_time"]
        if last < cursor:  # exchange returned overlapping data; avoid a stall
            break
        cursor = last + step
        if len(rows) < limit:
            break
        if progress_every and pages % progress_every == 0:
            # A symbol takes minutes to download, so a per-symbol line alone leaves the
            # operator staring at nothing and assuming the fetch has hung.
            message = "    %s page %d rows=%d" % (symbol, pages, written)
            if verbose:
                print(message, flush=True)
            if on_progress:
                on_progress(message)
    return {"symbol": symbol, "pages": pages, "written": written, "errors": errors}


def estimate_seconds(symbols, interval, days, concurrency=6, seconds_per_page=4.8):
    """Rough wall-clock estimate, so a long download is announced rather than silent.

    A year of 5m bars is about 71 pages per symbol; at the latency seen through a proxy
    a full 15-symbol run takes a quarter of an hour, and reporting nothing for that long
    looks like a hang.
    """
    step = interval_ms(interval)
    bars = int(days * 86_400_000 / step)
    pages = max(1, -(-bars // MAX_LIMIT))
    waves = max(1, -(-len(symbols) // max(1, concurrency)))
    return int(waves * pages * seconds_per_page)


async def ingest(symbols, interval, days, base_url, data_dir, concurrency=6,
                 end_ms=None, resume=True, on_progress=None):
    end_ms = int(time.time() * 1000) if end_ms is None else int(end_ms)
    start_ms = end_ms - days * 86_400_000
    store = Store(data_dir)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    results = []

    if on_progress:
        seconds = estimate_seconds(symbols, interval, days, concurrency)
        span = ("about %d minutes" % (seconds // 60) if seconds >= 120
                else "about %d seconds" % seconds)
        on_progress("downloading %d symbols at %s over %d days: %s"
                    % (len(symbols), interval, days, span))

    async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
        async def one(symbol):
            async with semaphore:
                ranges = (missing_ranges(store, symbol, interval, start_ms, end_ms)
                          if resume else [(start_ms, end_ms)])
                if not ranges:
                    print("  %s already covers the window" % symbol, flush=True)
                    return {"symbol": symbol, "pages": 0, "written": 0, "errors": []}
                began = time.time()
                outcome = {"symbol": symbol, "pages": 0, "written": 0, "errors": []}
                for begin, stop in ranges:
                    part = await ingest_symbol(session, base_url, store, symbol, interval,
                                               begin, stop, on_progress=on_progress)
                    outcome["pages"] += part["pages"]
                    outcome["written"] += part["written"]
                    outcome["errors"] += part["errors"]
                outcome["seconds"] = round(time.time() - began, 1)
                message = "  %-12s pages=%-4d rows=%-7d %.0fs %s" % (
                    symbol, outcome["pages"], outcome["written"], outcome["seconds"],
                    outcome["errors"] or "")
                print(message, flush=True)
                if on_progress:
                    on_progress(message)
                return outcome

        results = await asyncio.gather(*(one(s) for s in symbols))
    return results


def coverage(store, symbols, interval):
    """Per-symbol coverage with gap detection, so dirty data is visible up front."""
    step = interval_ms(interval)
    report = []
    for symbol in symbols:
        rows = store.db.execute(
            "SELECT COUNT(*) AS n, MIN(open_time) AS first, MAX(open_time) AS last "
            "FROM candles WHERE symbol=? AND interval=? AND is_closed=1",
            (symbol, interval)).fetchone()
        count = int(rows["n"] or 0)
        first, last = rows["first"], rows["last"]
        gaps = 0
        biggest_gap = 0
        if count:
            stamps = [int(r["open_time"]) for r in store.db.execute(
                "SELECT open_time FROM candles WHERE symbol=? AND interval=? AND is_closed=1 "
                "ORDER BY open_time", (symbol, interval))]
            for previous, current in zip(stamps, stamps[1:]):
                missing = (current - previous) // step - 1
                if missing > 0:
                    gaps += int(missing)
                    biggest_gap = max(biggest_gap, int(missing))
        expected = int((last - first) // step + 1) if count else 0
        report.append({"symbol": symbol, "bars": count, "first": first, "last": last,
                       "expected": expected, "missing": gaps, "largest_gap_bars": biggest_gap,
                       "coverage_pct": round(100.0 * count / expected, 3) if expected else 0.0})
    return report


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
                   "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT")


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="Download historical klines.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--base-url", default=settings.base_url)
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report", action="store_true", help="only print coverage")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    store = Store(args.data_dir)
    if args.report:
        print(json.dumps(coverage(store, symbols, args.interval), indent=2))
        return 0

    print("ingesting %d symbols interval=%s days=%d concurrency=%d"
          % (len(symbols), args.interval, args.days, args.concurrency), flush=True)
    began = time.time()
    results = asyncio.run(ingest(symbols, args.interval, args.days, args.base_url,
                                 args.data_dir, args.concurrency,
                                 resume=not args.no_resume))
    failed = [r for r in results if r["errors"]]
    print("done in %.1fs; symbols=%d pages=%d rows=%d failures=%d"
          % (time.time() - began, len(results), sum(r["pages"] for r in results),
             sum(r["written"] for r in results), len(failed)), flush=True)
    for row in coverage(store, symbols, args.interval):
        print("  %-12s bars=%-7d coverage=%6.2f%% missing=%-6d largest_gap=%d"
              % (row["symbol"], row["bars"], row["coverage_pct"], row["missing"],
                 row["largest_gap_bars"]), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
