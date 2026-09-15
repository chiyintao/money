"""Forward paper tracking for the cross-sectional momentum signal.

The backtest in `app/strategy/cross_sectional.py` reports +78.2 bps per trade at t = 3.37
on 31 symbols over 366 days. That number is measured *inside a single historical sample*,
and split-half stability is not the same thing as out-of-sample evidence: both halves are
still data the configuration was chosen by looking at. The only way to convert it into
evidence is to write the decision down before the outcome exists, which is what this module
does.

**What it is not.** It does not place orders, it does not touch the paper account, and it
does not influence the trading decision loop. It reads the same candle store the exchange
feed writes, computes the signal at each daily boundary using only data available at that
moment, records the target weights, and later settles them against realised prices. A signal
cannot be revised after the fact because the record is append-only and carries the prices it
was computed from.

**Why the point-in-time discipline matters here.** A momentum signal is trivially easy to
leak. If the daily close used for ranking is written by the same ingest pass that later
revises it, or if the "day" boundary is evaluated after the day is complete, the forward test
reproduces the backtest and proves nothing. Three guards enforce otherwise:

* a target is only recorded once the decision day has **closed** (its close bar is in the
  store and the wall clock is past the day boundary),
* the prices used for ranking are read from the store **at the moment of recording** and
  stored with the target, so a later settlement cannot silently re-rank,
* settling requires a *later* day than the decision day, and refuses to settle otherwise.

**What it measures.** Each target becomes one pending observation with a fixed horizon. When
that horizon closes, the realised portfolio return is computed from the stored weights and
written back. The report then answers the question the backtest could not: does the edge
survive on data that did not exist when the signal was chosen.
"""
import json
import time

from ..core.domain import Event
from . import cross_sectional as cs

# Event types. Distinct from the trading event types so a forward-tracking record can never
# be confused with an order, and so the retention policy can treat them independently.
SIGNAL_EVENT = "momentum_signal"
SETTLEMENT_EVENT = "momentum_settlement"

# Days after a target is recorded within which it must be settled. Past this the target is
# marked stale rather than settled against a much later price, because settling a one-day
# holding at a ten-day price is not the strategy that was recorded.
MAX_SETTLE_DELAY_DAYS = 3


def _now_ms(clock=None):
    return int(clock() * 1000) if clock else int(time.time() * 1000)


# Bars in a day at each interval, so the read window scales with the interval rather than
# assuming five minutes.
BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6}

# Calendar days of bars the tracker reads per symbol: the lookback plus the holding period,
# tripled for safety, never fewer than 30. A scheduled run only needs this much, because it
# evaluates one day and settles one day; loading a year per symbol to compute a seven-day
# return would be wasteful.
MIN_READ_DAYS = 30

# Days read when replaying history rather than tracking forward. A replay has to reconstruct
# the cross-section at every past decision point, so it needs the whole sample, not the tail.
REPLAY_READ_DAYS = 800


def read_universe(store, symbols=None, min_bars=None, interval="5m", days=None):
    """Daily closes per symbol from the candle store, bounded to what the signal needs.

    min_bars used to gate eligibility while the read window was capped at a few thousand
    bars, so every symbol was read and then rejected for having too few. The two numbers are
    now derived from the same quantity -- the read window -- and eligibility is decided by
    the store query rather than by a threshold that contradicts it.
    """
    per_day = BARS_PER_DAY.get(interval, 288)
    window_days = int(days) if days else max(MIN_READ_DAYS,
                                             3 * cs.DEFAULT_LOOKBACK_DAYS + 5)
    limit = window_days * per_day + per_day
    floor = int(min_bars) if min_bars else _eligible_floor(store, interval)
    universe = {}
    names = symbols or _listed_symbols(store, interval=interval, min_bars=floor)
    for symbol in names:
        bars = store.candles(symbol, interval, limit=limit, closed_only=True)
        if len(bars) >= per_day * 2:  # at least two days of bars to form a daily series
            universe[symbol] = bars
    return universe


def _eligible_floor(store, interval="5m", fallback=50_000):
    """The bar count separating a symbol with real history from a fresh listing.

    Read from the store rather than fixed, because the right floor depends on how much
    history the venue actually has. A symbol with at least half of the longest history is
    eligible; a listing that started last week is not, and including it would truncate the
    whole cross-section to its own length.
    """
    rows = store.db.execute(
        "SELECT COUNT(*) FROM candles WHERE interval=? GROUP BY symbol "
        "ORDER BY COUNT(*) DESC LIMIT 1", (interval,)).fetchone()
    longest = int(rows[0]) if rows else 0
    if longest <= 0:
        return fallback
    return max(2000, int(longest * cs.MIN_HISTORY_SHARE))



def _listed_symbols(store, interval="5m", min_bars=50_000):
    rows = store.db.execute(
        "SELECT symbol, COUNT(*) n FROM candles WHERE interval=? "
        "GROUP BY symbol HAVING n >= ? ORDER BY n DESC", (interval, min_bars)).fetchall()
    return [row[0] for row in rows]


class ForwardTracker:
    """Append-only record of momentum targets and their realised outcomes.

    The tracker holds no state between runs beyond what is in the store, so it is safe to
    call on a schedule and safe to interrupt: a target is either recorded in full or not at
    all, and settlement is idempotent because it matches on the recorded day.
    """

    def __init__(self, store, lookback=None, hold=None, quantile=None, cost_bps=None,
                 min_bars=None, max_settle_delay_days=MAX_SETTLE_DELAY_DAYS):
        self.store = store
        self.lookback = int(lookback or cs.DEFAULT_LOOKBACK_DAYS)
        self.hold = int(hold or cs.DEFAULT_HOLD_DAYS)
        self.quantile = float(quantile if quantile is not None else cs.DEFAULT_QUANTILE)
        self.cost_bps = float(cost_bps if cost_bps is not None else 6.0)
        # None rather than a fixed floor, so the eligibility threshold is derived from the
        # store by read_universe. A hardcoded 50,000 was measured to admit 34 symbols where
        # the store-derived floor admits the 31 that carry a full year -- and admitting the
        # three short listings is exactly the effect measured earlier, where the whole
        # cross-section is truncated to the newest listing's length and the edge fell from
        # 78.2 bps at t = 3.37 to 30.0 bps at t = 1.17.
        self.min_bars = int(min_bars) if min_bars else None
        self.max_settle_delay_days = int(max_settle_delay_days)

    # ---------------------------------------------------------------- recording

    def signal_days(self):
        """Decision days already recorded, as a set."""
        return {int(row["day"]) for row in self.records(SIGNAL_EVENT)}

    @property
    def ledger(self):
        """The database the signal record lives in.

        Market data and the forward record are allowed to live in different stores: a
        tracking process should be able to run against a *read-only* candle database and
        write its observations somewhere else entirely. Assuming one connection for both is
        what made the first version record 355 signals and then report zero of them, because
        the write went to the events store while the read went to the candle store. The
        ledger is therefore resolved explicitly: a store that exposes an events sink uses it,
        and a plain store is its own ledger.
        """
        return getattr(self.store, "events", self.store).db

    def records(self, event_type, limit=20000):
        """Parsed payloads of one event type, oldest first.

        The limit is a safety bound rather than a page size: a long-running tracker
        accumulates one event per day and the report has to see all of them, so the previous
        bound of two thousand would have silently truncated a record after five years. The
        ordering is by event_time and then by rowid, because two events can legitimately
        share a timestamp -- a signal and its settlement are stamped to the day they belong
        to -- and an unstable order would make the record read differently between runs.
        """
        rows = self.ledger.execute(
            "SELECT payload FROM events WHERE type=? ORDER BY event_time ASC, rowid ASC "
            "LIMIT ?", (event_type, limit)).fetchall()
        out = []
        for row in rows:
            try:
                out.append(json.loads(row["payload"]))
            except (TypeError, ValueError):
                continue
        return out

    def latest_complete_day(self, days, now_ms=None):
        """The most recent day whose close is final and already in the past.

        A day is only usable once the wall clock has passed its end. Using the newest day in
        the store without this check would evaluate a signal on a partially formed day, which
        is exactly the look-ahead the forward test exists to exclude.
        """
        if not days:
            return None
        now_ms = now_ms if now_ms is not None else _now_ms()
        current_day = cs.day_index(now_ms)
        complete = [day for day in days if day < current_day]
        return complete[-1] if complete else None

    def record(self, universe=None, now_ms=None, force_day=None):
        """Compute and persist one target for the latest complete day, if not already done.

        Returns a dict describing what happened; `recorded` is False when the day is already
        present or when there is not yet a complete day, and the reason says which.
        """
        universe = universe if universe is not None else read_universe(
            self.store, min_bars=self.min_bars)
        days, prices, meta = cs.align(universe)
        if not meta.get("ready"):
            return {"recorded": False, "reason": meta.get("reason"), "meta": meta}
        day = force_day if force_day is not None else self.latest_complete_day(days, now_ms)
        if day is None:
            return {"recorded": False, "reason": "no_complete_day"}
        if day in self.signal_days():
            return {"recorded": False, "reason": "already_recorded", "day": day}
        index = days.index(day)
        if index < self.lookback:
            return {"recorded": False, "reason": "insufficient_history", "day": day}
        at = {symbol: prices[symbol][index] for symbol in prices}
        past = {symbol: prices[symbol][index - self.lookback] for symbol in prices}
        weights, info = cs.rank_weights(at, past, self.quantile)
        if not info.get("ready"):
            return {"recorded": False, "reason": info.get("reason"), "day": day}
        payload = {
            "day": int(day),
            "lookback": self.lookback,
            "hold": self.hold,
            "quantile": self.quantile,
            "cost_bps": self.cost_bps,
            "weights": weights,
            "longs": info["longs"],
            "shorts": info["shorts"],
            "universe": info["universe"],
            # The prices the ranking was computed from, stored with the target so the
            # decision can be audited against the data that was actually visible.
            "prices_at": at,
            "recorded_at": _now_ms(now_ms),
            "settled": False,
        }
        self._emit(SIGNAL_EVENT, payload, event_time=cs.day_index(day) * 86_400_000 + 86_399_999)
        return {"recorded": True, "day": day, "weights": weights, "longs": info["longs"],
                "shorts": info["shorts"], "universe": info["universe"]}

    def _emit(self, event_type, payload, event_time=None):
        """Append one event, with the exchange-time stamp the payload implies.

        Event is a frozen dataclass, so the timestamp is set through reconstruct rather than
        by assignment. Building the event with the stamp already in the payload is what the
        class reads in its post-init, so the reconstruction is not a workaround: it is the
        only way to give an event a time other than "now", which is needed because a target
        decided on day D belongs on day D's timeline and not on the timeline of the run that
        happened to compute it.
        """
        stamped = dict(payload)
        if event_time is not None:
            stamped["event_time"] = int(event_time)
        event = Event(event_type, stamped)
        self.store.record_event(event.json())
        return event

    # ---------------------------------------------------------------- settling

    def settle(self, universe=None, now_ms=None):
        """Settle every pending target whose horizon has closed.

        Idempotent: a target already marked settled is skipped, and settlement is recorded
        as a separate event keyed on the day rather than by mutating the original record, so
        the signal as it was written down is never overwritten.
        """
        universe = universe if universe is not None else read_universe(
            self.store, min_bars=self.min_bars)
        days, prices, meta = cs.align(universe)
        if not meta.get("ready"):
            return {"settled": 0, "reason": meta.get("reason")}
        settled_days = {int(row["day"]) for row in self.records(SETTLEMENT_EVENT)}
        outcomes = []
        soonest = days[0] if days else None
        for payload in self.records(SIGNAL_EVENT):
            day = int(payload.get("day") or 0)
            if day in settled_days:
                continue
            hold = int(payload.get("hold") or self.hold)
            if day not in days:
                # The decision day has aged out of the store entirely. Recording the miss
                # rather than skipping is the point: a silently dropped target would make
                # the forward record look cleaner than it is, and the count of targets the
                # tracker failed to observe is itself part of the evidence.
                if soonest is not None and day < soonest:
                    outcomes.append(self._settle_missed(payload, days, prices, 0, hold))
                continue
            index = days.index(day)
            target_index = index + hold
            if target_index >= len(days):
                continue
            if days[target_index] - day > self.max_settle_delay_days:
                outcomes.append(self._settle_missed(payload, days, prices, index, hold))
                continue
            outcomes.append(self._settle_one(payload, days, prices, index, hold))
        return {"settled": len(outcomes), "outcomes": outcomes}

    def _settle_one(self, payload, days, prices, index, hold):
        weights = payload.get("weights") or {}
        cost = 2.0 * float(payload.get("cost_bps") or self.cost_bps) / 10000.0
        gross = 0.0
        detail = {}
        for symbol, weight in weights.items():
            series = prices.get(symbol)
            if not series:
                continue
            entry = series[index]
            exit_price = series[index + hold]
            if entry <= 0:
                continue
            move = exit_price / entry - 1.0
            gross += weight * move
            detail[symbol] = round(move, 8)
        net = gross - cost
        record = {"day": int(payload["day"]), "settle_day": int(days[index + hold]),
                  "gross": round(gross, 8), "net": round(net, 8), "cost": cost,
                  "symbols": len(detail), "moves": detail,
                  "settled_at": _now_ms()}
        self._emit(SETTLEMENT_EVENT, record,
                   event_time=cs.day_index(days[index + hold]) * 86_400_000 + 86_399_999)
        return record

    def _settle_missed(self, payload, days, prices, index, hold):
        """A target whose settlement window passed while the tracker was not running.

        Recorded as a miss rather than settled late. Settling a one-day holding at whatever
        price happens to be newest would silently substitute a different strategy for the one
        that was written down, and would bias the forward record toward whichever direction
        the market moved while the tracker was down.
        """
        record = {"day": int(payload["day"]), "missed": True,
                  "reason": "settlement_window_passed",
                  "available_day": int(days[-1]), "settled_at": _now_ms()}
        self._emit(SETTLEMENT_EVENT, record,
                   event_time=cs.day_index(days[-1]) * 86_400_000 + 86_399_999)
        return record

    # ---------------------------------------------------------------- reporting

    def report(self):
        """The forward record: how many observations, and what they realised."""
        signals = self.records(SIGNAL_EVENT)
        settlements = self.records(SETTLEMENT_EVENT)
        realised = [s for s in settlements if not s.get("missed")]
        missed = [s for s in settlements if s.get("missed")]
        pending = len(signals) - len(settlements)
        report = {"signals": len(signals), "settled": len(realised), "missed": len(missed),
                  "pending": max(0, pending), "lookback": self.lookback, "hold": self.hold,
                  "quantile": self.quantile, "cost_bps": self.cost_bps}
        if realised:
            report["summary"] = cs.summarise(
                [{"day": s["day"], "net": s["net"]} for s in realised], self.hold)
            gross = [s["gross"] for s in realised]
            report["gross_bps_mean"] = round(sum(gross) / len(gross) * 10000, 4)
            days = sorted(s["day"] for s in realised)
            report["first_day"] = days[0]
            report["last_day"] = days[-1]
        return report

    def verdict(self):
        """Plain-language reading, calibrated to what a forward record can support.

        The thresholds are deliberately about *observation count* first. A forward record of
        ten trades cannot confirm or refute anything, and reporting a t-statistic on it would
        invite exactly the over-reading this whole exercise is meant to avoid.
        """
        report = self.report()
        lines = ["forward paper tracking: %d signals, %d settled, %d pending, %d missed"
                 % (report["signals"], report["settled"], report["pending"], report["missed"])]
        if not report.get("summary"):
            lines.append("no settled observations yet; nothing to conclude")
            return lines
        s = report["summary"]
        lines.append("backtest reference: +78.2 bps/trade at t = 3.37 (in-sample)")
        lines.append("forward realised:   %+.1f bps/trade gross, %+.1f bps net"
                     % (report.get("gross_bps_mean", 0.0), s["net_bps_mean"]))
        lines.append("hit rate %.1f%% over %d trades" % (s["hit_rate"] * 100, s["trades"]))
        if report["settled"] < 30:
            lines.append("VERDICT: too few forward trades to judge; keep recording")
        elif s["net_bps_mean"] > 0 and s.get("t_stat", 0.0) >= 2.0:
            lines.append("VERDICT: forward edge is positive and significant so far")
        elif s["net_bps_mean"] > 0:
            lines.append("VERDICT: forward edge is positive but not yet significant")
        else:
            lines.append("VERDICT: forward edge is not positive; the backtest did not hold")
        if report["missed"]:
            lines.append("note: %d targets missed their settlement window" % report["missed"])
        return lines
