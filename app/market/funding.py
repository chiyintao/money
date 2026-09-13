"""Historical funding rates for perpetual futures.

Funding is what ties a perpetual to spot, and deviations of perpetuals from no-arbitrage
values are documented to be larger in crypto than in traditional FX and to comove across
coins. The live loop already receives the current rate on every mark-price update; what
was missing was its history, so it could never be a training feature -- the derivatives
table had the columns but held no rows.

Funding is published every eight hours, so a year costs about 1,100 records per symbol and
two API calls. That makes this the cheapest independent information source available.
"""
import argparse
import asyncio
import bisect
import json
import time

import aiohttp

from ..core.config import Settings
from ..storage.storage import Store

RECORDS_PER_DAY = 3  # funding settles every eight hours

# A funding history that never moves has a deviation of zero, but in floating point the
# sample deviation comes out around 1e-20 rather than 0, and dividing by it turns pure
# rounding noise into a z-score near 1. Anything below this floor (0.0001% per settlement)
# is economically indistinguishable from no variation, so the score is pinned to zero.
MIN_FUNDING_DEVIATION = 1e-6

# How old a mark price may be before it cannot support a basis.
#
# This bound is the whole point of this constant. The mark used to be read from the same
# row as the funding rate, and funding settles every eight hours, so "basis" was computed
# as the price change since the previous settlement -- not a basis at all. Measured on the
# shipped training set: mark_basis sat on its +/-2% clip for 122,201 of 1,258,344 rows,
# 9.711%, while every other feature saturated at 0.046% or less. A tenth of the training
# set fed the model a constant.
#
# Fifteen minutes is comfortably above the five-minute cadence the derivatives collector
# runs at and far below the eight-hour funding cadence, so a contemporaneous mark is
# accepted and a settlement mark never is.
MAX_BASIS_AGE_MS = 15 * 60 * 1000


def parse_funding(payload):
    """Convert the funding endpoint payload into storable rows."""
    rows = []
    for item in payload or []:
        try:
            rows.append({"symbol": str(item["symbol"]),
                         "event_time": int(item["fundingTime"]),
                         "funding_rate": float(item["fundingRate"]),
                         "mark_price": float(item.get("markPrice", 0) or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return rows


async def fetch_funding(session, base_url, symbol, start_ms, end_ms, limit=1000, attempts=4):
    url = base_url.rstrip("/") + "/fapi/v1/fundingRate"
    params = {"symbol": symbol, "startTime": int(start_ms), "endTime": int(end_ms),
              "limit": limit}
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            async with session.get(url, params=params, timeout=60) as response:
                if response.status in (418, 429):
                    await asyncio.sleep(max(delay, float(response.headers.get("Retry-After", delay))))
                    delay *= 2
                    continue
                if response.status != 200:
                    return None, "http_%d" % response.status
                return await response.json(), None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == attempts:
                return None, type(exc).__name__
            await asyncio.sleep(delay)
            delay *= 2
    return None, "exhausted_retries"


async def ingest_symbol(session, base_url, store, symbol, start_ms, end_ms, limit=1000):
    """Page forward until the window is covered; funding is sparse so two pages is typical."""
    cursor = int(start_ms)
    written = 0
    pages = 0
    while cursor < end_ms:
        payload, error = await fetch_funding(session, base_url, symbol, cursor, end_ms, limit)
        if error:
            return {"symbol": symbol, "pages": pages, "written": written, "error": error}
        rows = parse_funding(payload)
        if not rows:
            break
        store.record_derivatives(rows)
        written += len(rows)
        pages += 1
        last = max(row["event_time"] for row in rows)
        if last <= cursor:
            break
        cursor = last + 1
        if len(rows) < limit:
            break
    return {"symbol": symbol, "pages": pages, "written": written, "error": None}


async def ingest(symbols, days, base_url, data_dir, concurrency=6):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    store = Store(data_dir)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
        async def one(symbol):
            async with semaphore:
                outcome = await ingest_symbol(session, base_url, store, symbol,
                                              start_ms, end_ms)
                print("  %-12s records=%-6d pages=%d %s"
                      % (symbol, outcome["written"], outcome["pages"],
                         outcome["error"] or ""), flush=True)
                return outcome
        return await asyncio.gather(*(one(s) for s in symbols))


def funding_series(store, symbol):
    """Published funding records for one symbol, oldest first.

    The third return value is the mark price carried on each funding row. It is kept for
    callers written before mark_series existed, but it is an eight-hourly mark and must not
    be used to compute a basis; see mark_series.
    """
    rows = store.db.execute(
        "SELECT event_time, funding_rate, mark_price FROM derivatives WHERE symbol=? "
        "ORDER BY event_time", (symbol,)).fetchall()
    times = [int(row["event_time"]) for row in rows]
    rates = [float(row["funding_rate"]) for row in rows]
    marks = [float(row["mark_price"]) for row in rows]
    return times, rates, marks


def mark_series(store, symbol):
    """Mark prices observed close to the bars, oldest first.

    Read from derivatives_detail, where the collector stores open interest and open
    interest value every five minutes: the value divided by the size is the average price
    of the bucket, which is a mark price observed at bar resolution. The collector has
    always fetched both numbers and stored them; nothing had ever used them for this.

    Falls back to the funding rows when the detail table has nothing for the symbol, so an
    older database behaves exactly as it did before rather than losing the feature. The
    caller's staleness bound is what decides whether the fallback is usable.
    """
    rows = []
    try:
        rows = store.db.execute(
            "SELECT event_time, open_interest_value / open_interest AS mark "
            "FROM derivatives_detail WHERE symbol=? AND open_interest > 0 "
            "AND open_interest_value > 0 ORDER BY event_time", (symbol,)).fetchall()
    except Exception:
        rows = []
    times = []
    marks = []
    for row in rows:
        try:
            mark = float(row["mark"])
        except (TypeError, ValueError, KeyError):
            continue
        if mark > 0 and mark == mark and mark not in (float("inf"), float("-inf")):
            times.append(int(row["event_time"]))
            marks.append(mark)
    if times:
        return times, marks
    _times, _rates, funding_marks = funding_series(store, symbol)
    pairs = [(time, mark) for time, mark in zip(_times, funding_marks) if mark > 0]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def funding_features(times, rates, marks, bar_time, price=None, window=30, mark_series=None,
                     max_basis_age_ms=MAX_BASIS_AGE_MS):
    """Funding state known at `bar_time`, never using a later publication.

    The record published at or before the bar is the one a trader could have seen, so the
    lookup is a right bisect rather than a nearest match.

    Funding and mark prices have different cadences -- eight hours against five minutes --
    so they get their own series and their own bisect. Sharing one index, which is what this
    did, silently tied the basis to the funding settlement and turned it into the price
    change since that settlement. mark_series is passed as a (times, values) pair because a
    time axis on its own is not enough: the two series' values are different, and pairing a
    mark axis with the funding values looks correct and is not. It defaults to the funding
    series for callers that genuinely have one, and the age bound applies either way.

    mark_basis comes back as None when no mark is visible at the bar or the newest one is
    older than max_basis_age_ms. None rather than 0.0 on purpose: zero sits inside the
    training bounds, so a filled-in default is indistinguishable from a real reading and
    the caller cannot report it.
    """
    position = bisect.bisect_right(times, bar_time) - 1
    if position < 0:
        # Nothing published yet, so none of the four is known. Zeros here would be
        # placeholders that sit inside the training bounds, which is how a caller ends up
        # treating "no funding data" as "funding is exactly neutral" -- the mistake this
        # whole family was dead from at serving time. The caller fills the defaults and
        # reports them as degraded; see FeatureSource.
        return {"funding_rate": None, "funding_z": None, "funding_carry_24h": None,
                "mark_basis": None}
    current = rates[position]
    history = rates[max(0, position - window + 1):position + 1]
    if len(history) > 1:
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / (len(history) - 1)
        deviation = variance ** 0.5
        z_score = (current - mean) / deviation if deviation > MIN_FUNDING_DEVIATION else 0.0
    else:
        z_score = 0.0
    carry = sum(rates[max(0, position - RECORDS_PER_DAY + 1):position + 1])
    mark_times, mark_values = mark_series if mark_series else (times, marks)
    return {"funding_rate": current, "funding_z": z_score,
            "funding_carry_24h": carry,
            "mark_basis": mark_basis(mark_times, mark_values, bar_time, price,
                                     max_basis_age_ms)}


def mark_basis(mark_times, marks, bar_time, price, max_basis_age_ms=MAX_BASIS_AGE_MS):
    """The bar's basis, or None when no mark price close enough to the bar exists.

    Separated from the funding lookup because the two series have different cadences, and
    because this is the part that was wrong: an eight-hour-old mark compared against the
    current close is a price change, not a basis, and on the shipped dataset it saturated
    at the clip for a tenth of all rows.
    """
    if not price or price <= 0 or marks is None or not mark_times or not marks:
        return None
    position = bisect.bisect_right(mark_times, bar_time) - 1
    if position < 0:
        return None
    if int(bar_time) - int(mark_times[position]) > int(max_basis_age_ms):
        return None
    mark = marks[position] if position < len(marks) else 0.0
    if not mark or mark <= 0:
        return None
    return mark / price - 1


def coverage(store, symbols):
    report = []
    for symbol in symbols:
        # A funding rate of exactly zero is a legitimate neutral settlement, not a
        # missing record, so it must not be filtered out of the coverage count.
        row = store.db.execute(
            "SELECT COUNT(*) AS n, MIN(event_time) AS first, MAX(event_time) AS last "
            "FROM derivatives WHERE symbol=?", (symbol,)).fetchone()
        report.append({"symbol": symbol, "records": int(row["n"] or 0),
                       "first": row["first"], "last": row["last"]})
    return report


def main(argv=None):
    settings = Settings()
    from .ingest import DEFAULT_SYMBOLS
    parser = argparse.ArgumentParser(description="Download funding-rate history.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--base-url", default=settings.base_url)
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    store = Store(args.data_dir)
    if args.report:
        print(json.dumps(coverage(store, symbols), indent=2))
        return 0
    began = time.time()
    results = asyncio.run(ingest(symbols, args.days, args.base_url, args.data_dir,
                                 args.concurrency))
    failed = [r for r in results if r["error"]]
    print("funding done in %.0fs; records=%d failures=%d"
          % (time.time() - began, sum(r["written"] for r in results), len(failed)))
    for row in coverage(store, symbols):
        print("  %-12s records=%-6d %s -> %s"
              % (row["symbol"], row["records"], row["first"], row["last"]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
