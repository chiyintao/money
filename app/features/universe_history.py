"""Point-in-time universe reconstruction.

The tier rules in universe.py are a judgement about a symbol made from exchange metadata
and 24h statistics -- and every call passed the statistics as of *now*. That is fine for
choosing what to trade today and wrong for everything else, because a backtest over two
years selected the symbols that are liquid at the moment the backtest was launched.

Two failures follow, and they push in the same direction. A symbol that was deep in 2023
and delisted in 2024 never appears at all, so its losses are absent from the result --
survivorship. A symbol that listed last month appears for the whole window including the
year before it existed -- look-ahead. Both make a strategy look better than it was, and
neither is visible in the output.

This module rebuilds the same classification from the candles that were actually trading
at the moment in question. Quote volume, trade count, price and 24h range all come from
the trailing 24 hours ending at that moment; listing age is measured to that moment; a
symbol with no candles in that window was not tradeable and is not in the universe.
Nothing is read from the present.

The live path also records what it classified each day, so the record accumulates rather
than having to be reconstructed. Storage is append-only and keyed by day.
"""
from .universe import TIER_EXCLUDED, classify_one, describe

# The lookback the tier rules call "24h". Reconstruction needs to know how many bars that
# is for the interval in question rather than assuming one.
DAY_MS = 86_400_000


def _bar_time(row):
    """When a candle happened, preferring its close.

    A label made from a bar resolves after the bar closes, so the universe at time T should
    include bars that had closed by T -- not bars that were still forming.
    """
    for key in ("close_time", "open_time", "timestamp"):
        value = row.get(key)
        if value:
            return int(value)
    return 0


def _quantity(row, key):
    try:
        value = float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if value == value and abs(value) != float("inf") else 0.0


def symbol_facts(rows, as_of_ms, window_ms=DAY_MS, fetch_start_ms=None):
    """One symbol as it looked at a moment, or None if it was not trading then.

    The rows must reach back at least window_ms before as_of_ms. A caller that hands in
    rows beginning exactly at as_of_ms gets an empty trailing window and every symbol
    reads as delisted -- so fetch_start_ms is reported back, and the caller asserts on it
    rather than discovering the gap in a universe that is quietly empty.
    """
    if not rows:
        return None
    as_of = int(as_of_ms)
    ordered = sorted((row for row in rows if _bar_time(row)), key=_bar_time)
    if not ordered:
        return None
    listed = _bar_time(ordered[0])
    # Not yet listed: there is no bar at or before the moment being asked about.
    if listed > as_of:
        return None
    window = [row for row in ordered if as_of - window_ms < _bar_time(row) <= as_of]
    if not window:
        # Listed before the moment, but nothing traded in the trailing window: the contract
        # was delisted, halted, or has stopped printing. Either way it was not a candidate.
        return None
    last = window[-1]
    quote_volume = sum(_quantity(row, "quote_volume") for row in window)
    source = "column"
    if quote_volume <= 0:
        # Older databases predate the quote-volume column; reconstruct it from base volume
        # and price rather than reporting a zero that would exclude every symbol. The
        # approximation is exact for a USDT-quoted contract, which is all this trades.
        quote_volume = sum(_quantity(row, "volume") * _quantity(row, "close") for row in window)
        source = "derived" if quote_volume > 0 else "none"
    # A trade count of zero is a measurement. A trade count that was never recorded is not,
    # and the difference decides whether the tier gate is judging the exchange or judging
    # our own ingest. Every stored candle currently holds NULL here, so this is the case
    # that actually happens.
    measured = [row.get("trades") for row in window if row.get("trades") is not None]
    trades = int(sum(_quantity(row, "trades") for row in window)) if measured else None
    high = max(_quantity(row, "high") for row in window)
    low = min(value for value in (_quantity(row, "low") for row in window) if value > 0) \
        if any(_quantity(row, "low") > 0 for row in window) else 0.0
    # When the rows begin at the fetch boundary, the true listing predates the data in
    # hand and age_days is a lower bound. Understating age makes the age gate stricter,
    # which is the safe direction; flagging it is what keeps it from being a silent one.
    truncated = fetch_start_ms is not None and listed <= int(fetch_start_ms) + window_ms
    return {"symbol": last.get("symbol"),
            "quote_volume": quote_volume,
            "trades": trades,
            "trades_measured": bool(measured),
            "trades_rows_measured": len(measured),
            "quote_volume_source": source,
            "price": _quantity(last, "close"),
            "high": high, "low": low,
            # Measured to the moment asked about, not to now.
            "age_days": (as_of - listed) / DAY_MS if listed > 0 else 0.0,
            "bars": len(window),
            "listed_at": listed,
            # When this symbol's own series stops, against the moment being reconstructed.
            # A symbol that is not in the universe at a moment has two very different
            # explanations -- it stopped trading, or our ingest stopped -- and the verdict
            # alone cannot tell them apart. The gap is what distinguishes them.
            "last_bar_ms": _bar_time(last),
            "feed_lag_ms": max(0, as_of - _bar_time(last)),
            "listed_age_is_lower_bound": bool(truncated),
            "window_ms": window_ms}


def exclusion_reason(rows, as_of_ms, window_ms=DAY_MS):
    """Why symbol_facts returned None, as one of three distinct facts.

    "Untraded" is an answer, not a reason. A symbol with no stored rows at all, one whose
    first bar is after the moment, and one that was trading earlier but printed nothing in
    the trailing window are three different findings: missing data, listed later, and
    stopped. Collapsing them into a single absent symbol is how a reconstruction with an
    empty fetch window reads exactly like a genuine market with nothing tradeable in it.
    """
    as_of = int(as_of_ms)
    ordered = sorted((row for row in rows or () if _bar_time(row)), key=_bar_time)
    if not ordered:
        return "no_data"
    if _bar_time(ordered[0]) > as_of:
        return "not_listed_yet"
    window = [row for row in ordered if as_of - window_ms < _bar_time(row) <= as_of]
    if not window:
        return "no_bars_in_window"
    return None


def data_horizon(series, interval_ms=None):
    """Where the stored data actually ends, per symbol and overall.

    The reconstruction has to be evaluated at the moment the data reaches, not at the wall
    clock. On a deployment whose candles for the training tier are refreshed only when a
    training run happens, "now" is more than a day past the last stored bar -- and every
    symbol then reads as having stopped trading. That is a fact about the ingest being
    reported as a fact about the market, and the two have opposite meanings.
    """
    first = None
    horizon = None
    per_symbol = {}
    for symbol, rows in (series or {}).items():
        times = [_bar_time(row) for row in rows or () if _bar_time(row)]
        if not times:
            per_symbol[symbol] = {"first": None, "last": None, "bars": 0}
            continue
        low, high = min(times), max(times)
        per_symbol[symbol] = {"first": low, "last": high, "bars": len(times)}
        first = low if first is None else min(first, low)
        horizon = high if horizon is None else max(horizon, high)
    span = None
    if first is not None and horizon is not None and interval_ms:
        span = int(horizon - first) // int(interval_ms) + 1
    return {"first": first, "horizon": horizon, "symbols": len(per_symbol),
            "per_symbol": per_symbol, "expected_bars": span, "interval_ms": interval_ms}

def extent_from_windows(windows):
    """Where each symbol's stored data starts and ends, from Store.candle_windows.

    The same shape data_horizon returns, without needing every bar in between to say it.
    """
    per_symbol = {symbol: {"first": int(entry["listed_at"]), "last": int(entry["last_at"])}
                  for symbol, entry in (windows or {}).items()}
    if not per_symbol:
        return {"first": None, "horizon": None, "symbols": 0, "per_symbol": {}}
    return {"first": min(item["first"] for item in per_symbol.values()),
            "horizon": max(item["last"] for item in per_symbol.values()),
            "symbols": len(per_symbol), "per_symbol": per_symbol}


def series_from_windows(sources, moments):
    """One candle series per symbol, assembled from several bounded window reads.

    ``sources`` is one Store.candle_windows result per group of moments. Bars are keyed by
    close time because two overlapping windows would otherwise contribute the same bar
    twice, and symbol_facts would sum its volume twice. Each series gets a one-field stub
    row at the listing time so that age is still measured from the true first bar rather
    than from the earliest bar the windows happened to cover.
    """
    merged = {}
    for source in sources:
        for symbol, entry in (source or {}).items():
            target = merged.setdefault(str(symbol), {"listed_at": int(entry["listed_at"]),
                                                     "rows": {}})
            target["listed_at"] = min(target["listed_at"], int(entry["listed_at"]))
            for rows in (entry.get("windows") or {}).values():
                for row in rows or ():
                    target["rows"][_bar_time(row)] = row
    series = {}
    for symbol, entry in merged.items():
        ordered = [entry["rows"][key] for key in sorted(entry["rows"]) if key]
        if entry["listed_at"] and (not ordered or entry["listed_at"] < _bar_time(ordered[0])):
            ordered = [{"symbol": symbol, "close_time": entry["listed_at"]}] + ordered
        series[symbol] = ordered
    return series


def point_in_time(series, as_of_ms, rules=None, window_ms=DAY_MS, fetch_start_ms=None):
    """Classify every symbol as it stood at as_of_ms.

    series maps symbol to its candle rows, which must reach back at least window_ms before
    as_of_ms. Symbols that were not trading at that moment are absent from the result rather
    than present with zeroed statistics, so a caller cannot mistake "delisted" for
    "illiquid".
    """
    graded = []
    excluded_unlisted = []
    # Not named "reasons": classify_one already returns a reasons list that is unpacked
    # below, and shadowing it turned every excluded symbol into an index error.
    untraded_reasons = {}
    for symbol, rows in (series or {}).items():
        facts = symbol_facts(rows, as_of_ms, window_ms, fetch_start_ms=fetch_start_ms)
        if facts is None:
            excluded_unlisted.append(symbol)
            untraded_reasons[symbol] = (exclusion_reason(rows, as_of_ms, window_ms)
                                        or "unclassified")
            continue
        described = describe({"symbol": symbol, "volume": facts["quote_volume"],
                              "count": facts["trades"], "price": facts["price"],
                              "high": facts["high"], "low": facts["low"],
                              # describe() measures age from the listing time to a "now";
                              # here that now is the moment being reconstructed.
                              "open_time": facts["listed_at"]}, int(as_of_ms))
        tier, reasons = classify_one(described, rules)
        graded.append({**described, "tier": tier, "reasons": reasons,
                       "bars": facts["bars"], "listed_at": facts["listed_at"],
                       "listed_age_is_lower_bound": facts["listed_age_is_lower_bound"]})
    graded.sort(key=lambda item: -item["quote_volume"])
    counts = _counts(graded)
    return {"as_of": int(as_of_ms), "graded": graded, "untraded": sorted(excluded_unlisted),
            "untraded_reasons": untraded_reasons, "unmeasured_criteria": _unmeasured(graded),
            "counts": counts, "window_ms": int(window_ms),
            "graded_count": len(graded), "untraded_count": len(excluded_unlisted),
            "tradeable": any(tier != TIER_EXCLUDED for tier in counts),
            "measurable": _measurable(counts),
            "fetch_start": None if fetch_start_ms is None else int(fetch_start_ms)}


def _measurable(counts):
    """Whether the reconstruction produced any verdict at all.

    Distinct from "something was tradeable", and the difference is the whole point. An
    empty counts dict means no symbol was classified: the fetch window was empty, or the
    store holds nothing for these candidates. Every symbol in the excluded tier is a
    verdict -- a real one, that the request reaches further back than the data does -- and
    a caller that treats the two the same either reports a failed measurement as an empty
    market or builds a dataset from a measurement that found nothing.
    """
    return bool(counts)


def _unmeasured(graded):
    """Which tier criteria were decided by a column that holds nothing, and for how many.

    An excluded symbol is not one finding. "Listed only 96 days" and "the trade count was
    never recorded" land in the same tier and call for opposite responses -- one is the
    market, the other is our ingest. This is what tells the caller which it is holding.
    """
    counts = {}
    for item in graded or ():
        for reason in item.get("reasons") or ():
            if isinstance(reason, str) and reason.startswith("unmeasured:"):
                name = reason.split(":", 1)[1].strip()
                counts[name] = counts.get(name, 0) + 1
    return counts


def _counts(graded):
    counts = {}
    for item in graded:
        counts[item["tier"]] = counts.get(item["tier"], 0) + 1
    return counts


def symbols_at(series, as_of_ms, tier=None, limit=None, rules=None, window_ms=DAY_MS,
               fetch_start_ms=None):
    """The symbols that qualified at a moment, most liquid first."""
    report = point_in_time(series, as_of_ms, rules=rules, window_ms=window_ms,
                           fetch_start_ms=fetch_start_ms)
    graded = [item for item in report["graded"]
              if tier is None or item["tier"] == tier]
    symbols = [item["symbol"] for item in graded]
    return symbols[:limit] if limit else symbols


def universe_changes(before, after):
    """How the universe moved between two reconstructions.

    Reported because it is the measurement the old approach could not produce: a symbol
    that leaves the universe is a symbol whose later history would otherwise have been
    silently excluded from the result.
    """
    old = {item["symbol"]: item for item in (before or {}).get("graded", [])}
    new = {item["symbol"]: item for item in (after or {}).get("graded", [])}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    retiered = sorted(symbol for symbol in set(old) & set(new)
                      if old[symbol]["tier"] != new[symbol]["tier"])
    return {"added": added, "removed": removed, "retiered": retiered,
            "before_counts": dict((before or {}).get("counts") or {}),
            "after_counts": dict((after or {}).get("counts") or {}),
            "added_after_exclusion": sorted(symbol for symbol in added
                                            if new[symbol]["tier"] != TIER_EXCLUDED),
            "removed_after_exclusion": sorted(symbol for symbol in removed
                                              if old[symbol]["tier"] != TIER_EXCLUDED)}


def series_from_rows(rows, symbols=None):
    """Group candle rows by symbol, the shape point_in_time expects."""
    wanted = set(symbols) if symbols else None
    grouped = {}
    for row in rows or ():
        symbol = row.get("symbol")
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        grouped.setdefault(symbol, []).append(row)
    return grouped
