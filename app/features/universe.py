"""Split the perpetual universe into a mainstream tier and a speculative one.

The two tiers are not a judgement about the projects; they describe what a model is
being asked to learn. A mainstream perpetual is deep and comparatively calm, so a fixed
round-trip cost is a large fraction of its move. A speculative one is thin and violent,
so the same cost is small relative to the move but the data is noisier and the book is
shallower. Pooling them teaches one model two different regimes, which is why the
training runs are separated.

Classification uses only exchange metadata and 24h statistics, so it is reproducible and
every symbol can be told why it landed where it did.
"""

TIER_MAINSTREAM = "mainstream"
TIER_SPECULATIVE = "speculative"
TIER_EXCLUDED = "excluded"

RULES = {
    # Established, deep, and calm enough that a 12bp round trip is not the whole move.
    TIER_MAINSTREAM: {"min_age_days": 730, "min_quote_volume": 100_000_000,
                      "min_trades": 100_000, "min_price": 0.01,
                      "max_daily_range_pct": 15.0},
    # Tradeable and old enough to have history; "fast money" is mainly defined by the
    # volatility it fails the mainstream bar on.
    TIER_SPECULATIVE: {"min_age_days": 90, "min_quote_volume": 10_000_000,
                       "min_trades": 20_000},
}


def _daily_range_pct(row):
    high, low, price = float(row.get("high", 0) or 0), float(row.get("low", 0) or 0), \
        float(row.get("price", 0) or 0)
    if price <= 0 or high <= 0 or low <= 0 or high < low:
        return None
    return (high - low) / price * 100.0


def describe(row, now_ms):
    """The measured quantities a tier decision is made from.

    ``count`` may be None, meaning the trade count was never recorded rather than that the
    symbol had no trades. It is carried through as None so the gate below can say which of
    the two it is looking at.
    """
    onboard = int(row.get("open_time", 0) or 0)
    age_days = (now_ms - onboard) / 86_400_000 if onboard > 0 else 0.0
    count = row.get("count")
    return {"symbol": row.get("symbol"),
            "quote_volume": float(row.get("volume", 0) or 0),
            "trades": None if count is None else int(count or 0),
            "price": float(row.get("price", 0) or 0),
            "age_days": round(age_days, 1),
            "daily_range_pct": _daily_range_pct(row)}


def classify_one(facts, rules=None):
    """Return (tier, reasons) for one described symbol."""
    rules = rules or RULES
    main, spec = rules[TIER_MAINSTREAM], rules[TIER_SPECULATIVE]
    failures = []
    if facts["age_days"] < main["min_age_days"]:
        failures.append("listed only %.0f days" % facts["age_days"])
    if facts["quote_volume"] < main["min_quote_volume"]:
        failures.append("24h volume below %.0fM" % (main["min_quote_volume"] / 1e6))
    if _below_count(facts, main["min_trades"]):
        failures.append("few trades")
    if facts["price"] < main["min_price"]:
        failures.append("price below %.4g" % main["min_price"])
    span = facts["daily_range_pct"]
    if span is None:
        failures.append("no 24h range")
    elif span > main["max_daily_range_pct"]:
        failures.append("daily range %.1f%% above %.0f%%"
                        % (span, main["max_daily_range_pct"]))
    if not failures:
        return TIER_MAINSTREAM, ["deep, established and calm"]

    blockers = []
    if facts["age_days"] < spec["min_age_days"]:
        blockers.append("listed only %.0f days" % facts["age_days"])
    if facts["quote_volume"] < spec["min_quote_volume"]:
        blockers.append("24h volume below %.0fM" % (spec["min_quote_volume"] / 1e6))
    unmeasured = [] if facts["trades"] is not None else ["trade count"]
    if _below_count(facts, spec["min_trades"]):
        blockers.append("too few trades to fill")
    if unmeasured:
        # Excluded, but for a reason about our own data rather than about the market. The
        # symbol cannot be placed in a tier by a criterion nothing measured, and saying so
        # is what separates "this contract is too thin" from "this column is empty".
        return TIER_EXCLUDED, ["unmeasured: " + ", ".join(unmeasured)]
    if blockers:
        return TIER_EXCLUDED, blockers
    return TIER_SPECULATIVE, failures


def _below_count(facts, threshold):
    """Whether a measured trade count is under the bar. An unmeasured one is not under it."""
    return facts["trades"] is not None and facts["trades"] < threshold


def classify(market_rows, rules=None, now_ms=None):
    """Tier every symbol, most liquid first, each with the reason it is where it is."""
    import time
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    graded = []
    for row in market_rows or []:
        facts = describe(row, now_ms)
        if not facts["symbol"]:
            continue
        tier, reasons = classify_one(facts, rules)
        graded.append({**facts, "tier": tier, "reasons": reasons})
    graded.sort(key=lambda item: -item["quote_volume"])
    return graded


def symbols_for(tier, market_rows, rules=None, limit=None, now_ms=None):
    graded = [item for item in classify(market_rows, rules, now_ms) if item["tier"] == tier]
    symbols = [item["symbol"] for item in graded]
    return symbols[:limit] if limit else symbols


def summarise(graded):
    counts = {}
    for item in graded:
        counts[item["tier"]] = counts.get(item["tier"], 0) + 1
    return counts
