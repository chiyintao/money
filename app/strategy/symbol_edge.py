"""Per-symbol edge measurement, and the meta-labeling gate built on it.

The audit that motivated this found the ensemble predicting the next move at roughly
coin-flip accuracy, but not uniformly: one symbol scored 64% at a 15 minute horizon while
another scored 33%. Every symbol was traded with identical rules and identical size, so
the losers were funded by the winners and the account bled.

The fix is the second layer of what Lopez de Prado calls meta-labeling. The primary model
keeps proposing a direction; this layer decides whether that proposal is worth acting on
for *this* symbol, based on that symbol's own measured, cost-adjusted record.

Two properties matter and are easy to get wrong:

1. Observations come from signals, not from trades. If the gate only learned from closed
   trades, a blocked symbol would never generate evidence and could never earn its way
   back. Recording every candidate forecast means a blocked symbol keeps accumulating
   proof while it is not being traded.

2. Nothing is scored until its horizon has actually elapsed. A forecast is resolved by
   looking up the price N bars later, and the outcome is net of the full round-trip cost,
   because a symbol whose gross edge is 8bps against a 12bps cost is a losing symbol no
   matter how good its hit rate looks.
"""
import bisect
import json
import math
from collections import deque

# How many scored outcomes to keep per symbol. Long enough to average out noise, short
# enough that a symbol which stopped working is dropped rather than defended by history.
DEFAULT_WINDOW = 200
# Bounded so an unbounded backlog of unresolved forecasts cannot grow without limit on a
# symbol that stopped receiving bars.
MAX_PENDING_PER_SYMBOL = 500


def signed_return(side, entry_price, exit_price):
    """The return this side would actually have earned, before costs."""
    entry_price = float(entry_price or 0.0)
    exit_price = float(exit_price or 0.0)
    if entry_price <= 0 or exit_price <= 0:
        return None
    raw = (exit_price - entry_price) / entry_price
    return raw if side == 'LONG' else -raw


class SymbolEdge:
    """Rolling, cost-adjusted track record for each symbol."""

    def __init__(self, horizon_bars=3, window=DEFAULT_WINDOW, min_samples=30,
                 min_net_bps=0.0, enabled=True, require_samples=True,
                 round_trip_cost=0.0, min_t=2.0, fdr=0.0):
        self.horizon_bars = max(1, int(horizon_bars))
        self.window = max(10, int(window))
        self.min_samples = max(0, int(min_samples))
        self.min_net_bps = float(min_net_bps)
        self.enabled = bool(enabled)
        # Deny until proven. Safe because observations are recorded whether or not the
        # symbol is tradable, so an unproven symbol still accumulates evidence.
        self.require_samples = bool(require_samples)
        self.round_trip_cost = float(round_trip_cost or 0.0)
        # Standard errors the mean must clear, and the Benjamini-Hochberg false discovery
        # rate across symbols. min_t = 2.0 is the conventional one-sided bound; fdr = 0.0
        # disables only the multiplicity correction, never significance itself.
        self.min_t = float(min_t or 0.0)
        self.fdr = float(fdr or 0.0)
        self.pending = {}
        self.scored = {}
        self.blocked = {}

    # ------------------------------------------------------------------ recording
    def observe(self, symbol, side, ref_time, price, expected_return=None):
        """Register a forecast to be scored once its horizon elapses.

        ref_time is when the forecast was made. It does not have to be a bar boundary:
        scoring anchors it to the bar that was forming at that moment, which is the last
        bar the decision could actually have seen.

        Called for every direction the primary model proposes, including ones the gate
        is about to reject, so rejection never blinds the tracker.
        """
        if not self.enabled or side not in ('LONG', 'SHORT'):
            return
        ref_time = int(ref_time or 0)
        price = float(price or 0.0)
        if ref_time <= 0 or price <= 0:
            return
        bucket = self.pending.setdefault(symbol, {})
        bucket[ref_time] = {'side': side, 'price': price,
                            'expected_return': expected_return}
        if len(bucket) > MAX_PENDING_PER_SYMBOL:
            for stale in sorted(bucket)[:len(bucket) - MAX_PENDING_PER_SYMBOL]:
                bucket.pop(stale, None)

    def resolve(self, symbol, times, closes):
        """Score every pending forecast whose horizon has now elapsed.

        times and closes are parallel arrays ordered oldest first. A forecast is anchored to
        the bar that was forming when it was made, and is only scored once horizon_bars
        further bars exist. Forecasts older than the series are dropped rather than scored
        against the wrong bar.
        """
        bucket = self.pending.get(symbol)
        if not bucket or not times:
            return 0
        resolved = 0
        for ref_time in sorted(bucket):
            index = bisect.bisect_right(times, ref_time) - 1
            if index < 0:
                continue
            target = index + self.horizon_bars
            if target >= len(times):
                continue
            item = bucket.pop(ref_time)
            gross = signed_return(item['side'], item['price'], closes[target])
            if gross is None:
                continue
            net_bps = (gross - self.round_trip_cost) * 10000.0
            series = self.scored.setdefault(symbol, deque(maxlen=self.window))
            series.append(round(net_bps, 6))
            resolved += 1
        return resolved

    # --------------------------------------------------------------------- stats
    def stats(self, symbol):
        series = self.scored.get(symbol)
        if not series:
            return {'symbol': symbol, 'n': 0, 'hit_rate': None, 'mean_net_bps': None,
                    'tradable': False, 'reason': 'no_evidence'}
        values = list(series)
        wins = sum(1 for v in values if v > 0)
        mean = sum(values) / len(values)
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        return {'symbol': symbol, 'n': len(values),
                'hit_rate': round(wins / len(values), 4),
                'mean_net_bps': round(mean, 4),
                'median_net_bps': round(median, 4),
                'worst_bps': round(ordered[0], 4),
                'best_bps': round(ordered[-1], 4),
                'tradable': self.allows(symbol)[0],
                'reason': self.allows(symbol)[1]}

    def statistics(self, symbol):
        """Mean, standard error and t-statistic of this symbol's scored net edge.

        The gate used to test only mean > floor on n = 30. That is not a test of anything:
        the standard error of a 30-sample mean of returns with a per-trade spread of tens
        of basis points is several basis points wide, so a pure-noise symbol clears a zero
        floor about half the time. Across a few hundred candidates, a rolling one-sided
        test with no dispersion term is guaranteed to promote a cohort of lucky ones.
        """
        series = self.scored.get(symbol) or []
        count = len(series)
        if count <= 0:
            return {'count': 0, 'mean': None, 'std': None, 'stderr': None, 't': None}
        mean = sum(series) / count
        if count < 2:
            return {'count': count, 'mean': mean, 'std': None, 'stderr': None, 't': None}
        variance = sum((value - mean) ** 2 for value in series) / (count - 1)
        deviation = variance ** 0.5
        stderr = deviation / math.sqrt(count)
        return {'count': count, 'mean': mean, 'std': deviation, 'stderr': stderr,
                't': (mean / stderr) if stderr > 0 else None}

    def q_values(self):
        """Benjamini-Hochberg adjusted significance across every scored symbol.

        The gate runs one test per symbol, every refresh. Without a multiplicity
        correction the expected number of false positives grows with the size of the
        universe, and a symbol promoted by chance is indistinguishable from one promoted
        by edge in every downstream report. BH controls the false discovery rate, which is
        the right error to control when the goal is that most of what is traded is real.
        """
        entries = []
        for symbol in self.scored:
            stats = self.statistics(symbol)
            if stats['count'] < self.min_samples or stats['t'] is None:
                continue
            # One-sided p-value for "this symbol's mean net edge exceeds the floor".
            z = (stats['mean'] - self.min_net_bps) / stats['stderr'] if stats['stderr'] else 0.0
            p = 0.5 * math.erfc(z / math.sqrt(2))
            entries.append((symbol, p))
        total = len(entries)
        if not total:
            return {}
        entries.sort(key=lambda item: item[1])
        adjusted = {}
        running = 1.0
        for rank in range(total, 0, -1):
            symbol, p = entries[rank - 1]
            running = min(running, p * total / rank)
            adjusted[symbol] = running
        return adjusted

    def allows(self, symbol):
        """Whether this symbol has earned the right to be traded. Returns (bool, reason)."""
        if not self.enabled:
            return True, 'edge_gate_disabled'
        series = self.scored.get(symbol)
        count = len(series) if series else 0
        if count < self.min_samples:
            if self.require_samples:
                return False, 'insufficient_edge_samples'
            return True, 'edge_sample_warmup'
        stats = self.statistics(symbol)
        mean = stats['mean']
        if mean <= self.min_net_bps:
            return False, 'symbol_edge_below_floor'
        # Significance, not just sign. The mean must clear min_t standard errors; without
        # this a 30-sample record means nothing.
        if self.min_t > 0 and (stats['t'] is None or stats['t'] < self.min_t):
            return False, 'symbol_edge_not_significant'
        if self.fdr > 0:
            q = self.q_values().get(symbol)
            if q is None or q > self.fdr:
                return False, 'symbol_edge_fails_fdr'
        return True, 'symbol_edge_ok'

    # Selection tiers. Ordering by these rather than by a yes/no answer is what stops a
    # session spending its slots on symbols already measured as losers: an unproven symbol
    # might become tradable, a proven-bad one has already been answered.
    TIER_ELIGIBLE = 2
    TIER_UNKNOWN = 1
    TIER_REJECTED = 0

    def tier(self, symbol):
        if not self.enabled:
            return self.TIER_ELIGIBLE
        allowed, reason = self.allows(symbol)
        if allowed:
            return self.TIER_ELIGIBLE
        if reason in ('insufficient_edge_samples', 'edge_sample_warmup'):
            return self.TIER_UNKNOWN
        return self.TIER_REJECTED

    def table(self):
        """Every symbol with evidence, worst first, for the dashboard and the report."""
        rows = [self.stats(symbol) for symbol in sorted(self.scored)]
        rows.sort(key=lambda row: (row['mean_net_bps'] if row['mean_net_bps'] is not None else 0.0))
        return rows

    # ---------------------------------------------------------------- persistence
    def snapshot(self):
        return {'horizon_bars': self.horizon_bars, 'window': self.window,
                'min_samples': self.min_samples, 'min_net_bps': self.min_net_bps,
                'round_trip_cost': self.round_trip_cost,
                'scored': {symbol: list(series) for symbol, series in self.scored.items()},
                'pending': {symbol: {str(k): v for k, v in bucket.items()}
                            for symbol, bucket in self.pending.items()}}

    def restore(self, payload):
        if not payload:
            return self
        self.round_trip_cost = float(payload.get('round_trip_cost') or 0.0)
        for symbol, series in (payload.get('scored') or {}).items():
            self.scored[symbol] = deque((float(v) for v in series), maxlen=self.window)
        for symbol, bucket in (payload.get('pending') or {}).items():
            self.pending[symbol] = {int(k): v for k, v in bucket.items()}
        return self


def _proposed_from_votes(decision):
    """Recover the direction the ensemble proposed, from the member votes.

    Rows written before proposed_side existed still carry every member's value, and the
    sign of their sum is the same direction the model proposed -- the gate only ever
    changed the *side* it reported, never the votes. Recovering it means the history
    already in the table is usable instead of being discarded, which matters because the
    gate denies until proven: throwing the history away restarts every symbol at zero
    samples and costs 165 minutes of refusals per symbol before anything can trade.
    """
    votes = decision.get('votes') or {}
    if not votes:
        return None
    try:
        total = sum(float(value) for value in votes.values())
    except (TypeError, ValueError):
        return None
    if total > 0:
        return 'LONG'
    if total < 0:
        return 'SHORT'
    return None


def seed_from_history(store, interval, horizon_bars, round_trip_cost, window=DEFAULT_WINDOW,
                      limit=200000):
    """Rebuild per-symbol evidence from decisions already in the audit trail.

    Without this the gate would start blind and refuse to trade anything, and the only
    way to earn evidence would be to trade -- which is the thing being decided. The
    stored decisions plus stored candles already contain everything needed, so the gate
    can begin from measured history instead of from a warmup period.
    """
    tracker = SymbolEdge(horizon_bars=horizon_bars, window=window,
                         min_samples=0, min_net_bps=0.0, round_trip_cost=round_trip_cost)
    rows = list(store.db.execute(
        "SELECT event_time, payload FROM events WHERE type='strategy_decision' ORDER BY event_time ASC LIMIT ?",
        (int(limit),)))
    seen = set()
    for event_time, payload in rows:
        try:
            event = json.loads(payload)
        except (TypeError, ValueError):
            continue
        event.setdefault('event_time', event_time)
        decision = event.get('decision') or {}
        market = event.get('market') or {}
        symbol = event.get('symbol')
        # The proposal, not the verdict. 'side' has already been flattened to FLAT by the
        # gate for every symbol the gate is blocking, so seeding from it rebuilds evidence
        # only for symbols that never needed it and leaves every blocked symbol with no
        # samples -- which is the state that keeps it blocked.
        side = decision.get('proposed_side') or _proposed_from_votes(decision) \
            or decision.get('side')
        ref_time = decision.get('bar_time') or market.get('bar_time') or event.get('event_time')
        price = market.get('price')
        if side not in ('LONG', 'SHORT') or not symbol or not ref_time or not price:
            continue
        # One observation per symbol per bar. The tick path repeats the same decision many
        # times per bar, and without this the window would fill with copies of one view.
        key = (symbol, int(ref_time) // 300000)
        if key in seen:
            continue
        seen.add(key)
        tracker.observe(symbol, side, ref_time, price, decision.get('expected_return'))
    for symbol in list(tracker.pending):
        series = store.candles(symbol, interval, 100000, closed_only=True)
        if not series:
            tracker.pending.pop(symbol, None)
            continue
        times = [int(row['open_time']) for row in series]
        closes = [float(row['close']) for row in series]
        tracker.resolve(symbol, times, closes)
    tracker.pending.clear()
    return tracker


# --------------------------------------------------------------- horizon analysis

DEFAULT_HORIZONS = (1, 2, 3, 6, 12, 24, 48)


def horizon_analysis(store, interval='5m', horizons=DEFAULT_HORIZONS, cost=0.0,
                     min_samples=10, symbols=None, limit=200000):
    """Cost-adjusted edge for every symbol at every horizon, from stored decisions.

    This is the measurement that decides how long a trade should be held. The audit that
    started this work found the ensemble's edge peaking in the 5-15 minute band and going
    negative by 30 minutes, while the exit rules were aiming at targets that take hours to
    reach. A reward:risk ratio is meaningless if the edge it is waiting for has already
    decayed by the time the target is touched, so the holding horizon has to come from
    measurement rather than from the geometry looking tidy.

    Returns a table plus the horizon with the best pooled result, which the service uses
    to set the default time stop.
    """
    import json as _json

    # The bucket that collapses repeated views of one bar. It was the literal 300000, so on
    # a 1m or 15m configuration every decision was folded into a five-minute bucket and the
    # de-duplication kept an arbitrary one of them.
    from ..core.config import interval_to_ms

    bucket = max(1, interval_to_ms(interval))
    rows = list(store.db.execute(
        "SELECT event_time, payload FROM events WHERE type='strategy_decision' ORDER BY event_time ASC LIMIT ?",
        (int(limit),)))
    decisions = {}
    for event_time, payload in rows:
        try:
            event = _json.loads(payload)
        except (TypeError, ValueError):
            continue
        decision = event.get('decision') or {}
        market = event.get('market') or {}
        symbol = event.get('symbol')
        side = decision.get('side')
        ref_time = decision.get('bar_time') or market.get('bar_time') or event_time
        price = market.get('price')
        if side not in ('LONG', 'SHORT') or not symbol or not ref_time or not price:
            continue
        if symbols and symbol not in symbols:
            continue
        # One observation per symbol per bar: the tick path repeats one view many times.
        decisions[(symbol, int(ref_time) // bucket)] = (symbol, side, int(ref_time), float(price))

    series_cache = {}
    def series_for(symbol):
        if symbol not in series_cache:
            candles = store.candles(symbol, interval, 100000, closed_only=True)
            series_cache[symbol] = (
                [int(row['open_time']) for row in candles],
                [float(row['close']) for row in candles],
            )
        return series_cache[symbol]

    table = {}
    for (symbol, side, ref_time, price) in decisions.values():
        times, closes = series_for(symbol)
        if not times:
            continue
        index = bisect.bisect_right(times, ref_time) - 1
        if index < 0:
            continue
        slot = table.setdefault(symbol, {h: [] for h in horizons})
        for horizon in horizons:
            target = index + horizon
            if target >= len(times):
                continue
            gross = signed_return(side, price, closes[target])
            if gross is None:
                continue
            slot[horizon].append((gross - cost) * 10000.0)

    report = {}
    for symbol, by_horizon in table.items():
        entry = {}
        for horizon, values in by_horizon.items():
            if len(values) < min_samples:
                continue
            mean = sum(values) / len(values)
            entry[horizon] = {
                'n': len(values),
                'hit_rate': round(sum(1 for v in values if v > 0) / len(values), 4),
                'mean_net_bps': round(mean, 4),
            }
        if entry:
            report[symbol] = entry

    pooled = {}
    for horizon in horizons:
        values = [v for symbol in table for v in table[symbol].get(horizon, [])]
        if len(values) < min_samples:
            continue
        mean = sum(values) / len(values)
        entry = {'n': len(values),
                 'hit_rate': round(sum(1 for v in values if v > 0) / len(values), 4),
                 'mean_net_bps': round(mean, 4)}
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            stderr = (variance ** 0.5) / math.sqrt(len(values))
            entry['stderr_bps'] = round(stderr, 4)
            entry['t'] = round(mean / stderr, 3) if stderr > 0 else None
            # Lower bound of a one-sided 95% interval. Picking the horizon with the best
            # point estimate out of seven, on overlapping windows, is a selection over
            # noise: the maximum of seven noisy means is positive even when none of them
            # is. The bound is what keeps a lucky argmax from becoming a live parameter.
            entry['lower_bound_bps'] = round(mean - 1.645 * stderr, 4)
        pooled[horizon] = entry

    eligible = {horizon: stats for horizon, stats in pooled.items()
                if stats['mean_net_bps'] > 0
                and (stats.get('lower_bound_bps') is None or stats['lower_bound_bps'] > 0)}
    best = None
    for horizon in eligible:
        if best is None or eligible[horizon]['mean_net_bps'] > eligible[best]['mean_net_bps']:
            best = horizon
    # Reported either way: "nothing cleared its own error bar" is a result, and it is a
    # different result from "no horizon had a positive mean".
    positive_but_unproven = sorted(horizon for horizon, stats in pooled.items()
                                   if stats['mean_net_bps'] > 0 and horizon not in eligible)
    return {'interval': interval, 'cost': cost, 'horizons': list(horizons),
            'pooled': pooled, 'symbols': report, 'best_horizon': best,
            'positive_but_unproven': positive_but_unproven,
            'selection': 'lower_bound' if best is not None else 'none_cleared_error_bar'}
