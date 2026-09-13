"""Trade-flow and liquidation aggregation.

The market socket already subscribed to aggTrade -- every trade prints -- and the parsed
event was used for one thing: to trigger a decision tick. The trade itself, which carries
the aggressor side, was discarded. So the highest-frequency and most direct measure of
order flow the venue publishes left the process a few milliseconds after it arrived.

forceOrder, the liquidation stream, was not subscribed at all. Liquidations are the
clearest evidence of forced positioning: a cascade is mechanically different from a
slow repricing, and it is exactly the situation in which a stop is filled far from where
it was placed.

Storing every trade is not the answer -- a busy perpetual prints thousands per minute and
none of them is individually informative. What is informative is the per-minute summary:
how much volume was buyer-initiated against seller-initiated, how many trades, how big the
largest one was, and how much was liquidated on each side. That is what this aggregates.

Buckets are keyed to the wall-clock minute rather than to a bar, so the same summary can
be joined to any interval and does not need to know which one the session runs.
"""
import time

BUCKET_MS = 60000


def bucket_of(event_time_ms, bucket_ms=BUCKET_MS):
    """Start of the aggregate bucket an event belongs to."""
    return (int(event_time_ms) // bucket_ms) * bucket_ms


def liquidation_side(raw_side):
    """Which side of the book a liquidation order represents.

    A forceOrder carries the side of the *order*, not of the position. A SELL liquidation
    order is a long being closed out, so it is long-side liquidation volume. Getting this
    backwards inverts the reading of every cascade, so it is named here rather than
    inlined.
    """
    side = str(raw_side or "").upper()
    if side == "SELL":
        return "long"
    if side == "BUY":
        return "short"
    return None


def parse_force_order(message):
    """Normalise a forceOrder payload into a liquidation event."""
    data = message.get("data", message) if isinstance(message, dict) else {}
    order = data.get("o") or {}
    return {"type": "liquidation", "symbol": order.get("s") or data.get("s"),
            "event_time": int(order.get("T") or data.get("E") or 0),
            "side": liquidation_side(order.get("S")),
            "quantity": float(order.get("z") or order.get("q") or 0 or 0),
            "price": float(order.get("ap") or order.get("p") or 0 or 0),
            "status": order.get("X")}


class _Bucket:
    __slots__ = ("buy_volume", "sell_volume", "trades", "buy_trades", "notional",
                 "largest_trade", "liquidations", "liquidation_long_volume",
                 "liquidation_short_volume", "liquidation_long_notional",
                 "liquidation_short_notional", "largest_liquidation")

    def __init__(self):
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trades = 0
        self.buy_trades = 0
        self.notional = 0.0
        self.largest_trade = 0.0
        self.liquidations = 0
        self.liquidation_long_volume = 0.0
        self.liquidation_short_volume = 0.0
        self.liquidation_long_notional = 0.0
        self.liquidation_short_notional = 0.0
        self.largest_liquidation = 0.0

    def row(self, symbol, event_time):
        total = self.buy_volume + self.sell_volume
        liquidated = self.liquidation_long_volume + self.liquidation_short_volume
        return {"symbol": symbol, "event_time": int(event_time),
                "buy_volume": self.buy_volume, "sell_volume": self.sell_volume,
                "trades": self.trades, "buy_trades": self.buy_trades,
                "notional": self.notional, "largest_trade": self.largest_trade,
                "delta_volume": self.buy_volume - self.sell_volume,
                "delta_ratio": ((self.buy_volume - self.sell_volume) / total) if total else None,
                "liquidations": self.liquidations,
                "liquidation_long_volume": self.liquidation_long_volume,
                "liquidation_short_volume": self.liquidation_short_volume,
                "liquidation_long_notional": self.liquidation_long_notional,
                "liquidation_short_notional": self.liquidation_short_notional,
                # Which side was forced out, as a signed number: positive means shorts
                # were liquidated (a short squeeze), negative means longs were.
                "liquidation_delta": self.liquidation_short_volume - self.liquidation_long_volume,
                "liquidation_volume": liquidated,
                "largest_liquidation": self.largest_liquidation,
                # Liquidated volume as a share of traded volume. A cascade is this ratio
                # spiking, not the absolute size, which scales with the symbol.
                "liquidation_share": (liquidated / total) if total else None}


class FlowCollector:
    """Accumulates trades and liquidations, and hands back finished minute buckets.

    Deliberately not a writer: it is called from the socket handler, which runs on every
    trade print and must not touch the database. The decision loop drains it.
    """

    def __init__(self, bucket_ms=BUCKET_MS, max_buckets=4096, drain_interval_ms=30000):
        self.bucket_ms = int(bucket_ms)
        self.max_buckets = int(max_buckets)
        self.drain_interval_ms = int(drain_interval_ms)
        self.last_drain_ms = 0
        self.buckets = {}
        self.stats = {"trades": 0, "liquidations": 0, "buckets": 0, "written": 0,
                      "dropped": 0}

    def add_trade(self, symbol, event_time, quantity, price, side):
        if not symbol or not quantity:
            return
        bucket = self.buckets.get((symbol, bucket_of(event_time, self.bucket_ms)))
        if bucket is None:
            bucket = self._open(symbol, event_time)
            if bucket is None:
                return
        volume = abs(float(quantity))
        notional = volume * abs(float(price or 0))
        bucket.trades += 1
        bucket.notional += notional
        bucket.largest_trade = max(bucket.largest_trade, notional)
        if str(side).upper() == "BUY":
            bucket.buy_volume += volume
            bucket.buy_trades += 1
        else:
            bucket.sell_volume += volume
        self.stats["trades"] += 1

    def add_liquidation(self, symbol, event_time, quantity, price, side):
        if not symbol or side not in ("long", "short"):
            return
        bucket = self.buckets.get((symbol, bucket_of(event_time, self.bucket_ms)))
        if bucket is None:
            bucket = self._open(symbol, event_time)
            if bucket is None:
                return
        volume = abs(float(quantity or 0))
        notional = volume * abs(float(price or 0))
        bucket.liquidations += 1
        bucket.largest_liquidation = max(bucket.largest_liquidation, volume)
        if side == "long":
            bucket.liquidation_long_volume += volume
            bucket.liquidation_long_notional += notional
        else:
            bucket.liquidation_short_volume += volume
            bucket.liquidation_short_notional += notional
        self.stats["liquidations"] += 1

    def observe(self, event):
        """Route one normalised socket event; unknown types are ignored."""
        kind = (event or {}).get("type")
        if kind == "trade":
            self.add_trade(event.get("symbol"), event.get("event_time", 0),
                           event.get("quantity", 0), event.get("price", 0),
                           event.get("side"))
        elif kind == "liquidation":
            self.add_liquidation(event.get("symbol"), event.get("event_time", 0),
                                 event.get("quantity", 0), event.get("price", 0),
                                 event.get("side"))

    def _open(self, symbol, event_time):
        # A busy session with a wide symbol list could otherwise grow without bound if the
        # drain stops running; the oldest bucket is dropped rather than the newest, so the
        # data that survives is the data closest to now.
        if len(self.buckets) >= self.max_buckets:
            oldest = min(self.buckets, key=lambda key: key[1])
            del self.buckets[oldest]
            self.stats["dropped"] += 1
        key = (symbol, bucket_of(event_time, self.bucket_ms))
        bucket = _Bucket()
        self.buckets[key] = bucket
        return bucket

    def drain_due(self, now_ms=None):
        """Whether enough time has passed for another drain attempt."""
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        return (now - self.last_drain_ms) >= self.drain_interval_ms

    def drain(self, now_ms=None, keep=1):
        """Finished buckets, as storage rows.

        The bucket covering the current minute is still being written to, so it is held
        back: emitting it now would persist a partial minute and the next drain would
        overwrite it with a smaller number if the process restarted.
        """
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        self.last_drain_ms = now
        cutoff = bucket_of(now, self.bucket_ms) - (max(0, int(keep) - 1) * self.bucket_ms)
        finished = [key for key in self.buckets if key[1] < cutoff]
        rows = []
        for key in sorted(finished, key=lambda item: item[1]):
            symbol, event_time = key
            rows.append(self.buckets.pop(key).row(symbol, event_time))
        self.stats["buckets"] += len(rows)
        return rows

    def health(self):
        return {**self.stats, "open_buckets": len(self.buckets),
                "bucket_ms": self.bucket_ms,
                "drain_interval_ms": self.drain_interval_ms}
