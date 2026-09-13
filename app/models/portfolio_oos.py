"""Out-of-sample portfolio evidence for the promotion gate.

The registry refuses to promote a model without a portfolio_oos block: out-of-sample,
costs included, at least 30 trades, positive net return, drawdown within 20%. Nothing in
the repository produced one. So the gate could not be passed, no candidate was ever
promoted, and the service ran unvetted weights for its whole history while reporting the
condition only as a mode string.

Producing the block means answering a different question than the walk-forward answers.
The walk-forward scores each out-of-sample prediction on its own: does this prediction,
taken alone, beat the round-trip cost? That is necessary and not sufficient. A promotion
decision is about an account -- one balance, concurrent positions competing for margin and
for the same risk budget, position sizing that shrinks when the book is full, and a
drawdown that accumulates across trades rather than resetting at each one.

So the out-of-sample predictions are replayed through the same PaperAccount, RiskEngine and
PortfolioLimits the live service uses. That is what makes the number a promotion decision
can be based on: it is the result of the code that will run, on data that model has not
seen.
"""
import math

from ..backtest.backtest import live_limits, live_risk, run_portfolio_backtest
from ..market.funding import funding_series

# Below this, out-of-sample trade counts are too small for the drawdown and the return to
# mean anything. The registry enforces its own floor; this one stops the replay earlier so
# the caller is told the evidence is thin rather than handed a number and left to notice.
MIN_TRADES = 30
MIN_SYMBOLS = 2


def oos_decisions(oos, cost_bps=12.0):
    """Turn out-of-sample predictions into per-symbol entry decisions.

    The threshold is the round-trip cost, exactly as in the walk-forward: a predicted move
    smaller than the cost of capturing it is not a trade, and pretending otherwise is how a
    strategy with no edge reports a positive gross return. The side is signed the same way
    in both places so the two cannot disagree about what was traded.
    """
    threshold = float(cost_bps) / 10000.0
    decisions = {}
    for record in oos or ():
        symbol = record.get("symbol")
        if not symbol:
            continue
        predicted = float(record.get("predicted_return") or 0.0)
        side = "LONG" if predicted > threshold else "SHORT" if predicted < -threshold else None
        if side is None:
            continue
        decisions.setdefault(symbol, {})[int(record["timestamp"])] = {
            "side": side, "predicted_return": predicted, "timestamp": int(record["timestamp"]),
            "fold": record.get("fold")}
    return decisions


def decision_strategy(decisions, cost_bps=12.0, stop_distance=0.01, target_multiple=2.0):
    """A backtest strategy that replays the recorded decisions and nothing else.

    It reads the decision for the bar it is asked about, so the replay cannot look ahead:
    the decision exists only for timestamps the walk-forward actually predicted, and those
    are out-of-sample by construction.

    The returned mapping uses the same field names as the live plan -- entry, stop,
    take_profit, expected_cost, expected_edge -- because the replay goes through the same
    RiskEngine and bracket checks. A plan carrying "price" and "target" instead is refused
    as invalid_plan and the replay reports zero trades with no indication why.
    """
    round_trip = float(cost_bps) / 10000.0

    # How many lookups actually found a decision. A replay whose keys do not match the
    # dataset's reports zero trades and no reason; this turns that into a number the
    # evidence block can carry.
    seen = {"hits": 0, "misses": 0}

    def decide(symbol, history):
        if not history:
            return None
        # The dataset identifies a bar by its close_time -- build_dataset writes
        # row["timestamp"] = bars[index]["close_time"] -- and the walk-forward keys every
        # out-of-sample prediction by that same field. The lookup here used the bar's
        # open_time, which for a five-minute bar is 299,999 ms earlier. The two never met,
        # so a replay holding 1,244 real decisions executed none of them, reported zero
        # trades, and failed the promotion gate for insufficient evidence -- while the
        # walk-forward that produced those decisions reported 1,244 of them.
        # close_time first, open_time kept as a fallback for a caller that keyed the other way.
        last = history[-1]
        chosen = decisions.get(symbol, {}).get(int(last.get("close_time") or 0))
        if not chosen:
            chosen = decisions.get(symbol, {}).get(int(last.get("open_time") or 0))
        if not chosen:
            seen["misses"] += 1
            return None
        seen["hits"] += 1
        entry = float(history[-1].get("close") or 0)
        if entry <= 0:
            return None
        distance = abs(entry) * float(stop_distance)
        # Targeted at a multiple of the stop distance, which is the geometry the live exit
        # policy uses, and floored at the round-trip cost so the bracket check cannot refuse
        # it for being inside the entry.
        move = max(distance * float(target_multiple), entry * round_trip * 1.5)
        if chosen["side"] == "LONG":
            stop, target = entry - distance, entry + move
        else:
            stop, target = entry + distance, entry - move
        return {"symbol": symbol, "side": chosen["side"], "entry": entry, "price": entry,
                "stop": stop, "take_profit": target, "target": target,
                "expected_return": chosen["predicted_return"],
                "expected_cost": entry * round_trip,
                "expected_edge": move - entry * round_trip,
                "reason": "oos_replay"}
    decide.stats = seen
    return decide


def funding_for(store, symbols):
    """Published funding rates per symbol, shaped for the replay.

    The live loop charges funding on every tick and the account advances each symbol's
    eight-hour (or four, or one) window only when the charge is booked. The replay did not
    charge it at all, so its net_return excluded a real perpetual cost while the evidence
    block asserted that costs were included. For a strategy that holds the crowded side of
    the book this is not a rounding error; it is the difference between an edge and no edge,
    and the sign of it is the whole reason the rate is published.
    """
    if store is None:
        return None
    rates = {}
    for symbol in symbols or ():
        try:
            times, values, _marks = funding_series(store, symbol)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        points = sorted(zip((int(value) for value in times),
                            (float(value) for value in values)))
        if points:
            rates[str(symbol)] = points
    return rates or None


def funding_windows(store, symbols):
    """Per-symbol settlement windows, when something can supply them.

    The venue runs eight-hour funding on most contracts and four or one hour on some. A
    single account-wide window either charges a four-hour contract half as often as it
    should or charges an eight-hour contract twice.
    """
    reader = getattr(store, "funding_intervals", None)
    if not callable(reader):
        return None
    try:
        windows = reader(symbols)
    except Exception:
        return None
    return {str(key): int(value) for key, value in (windows or {}).items()} or None


def series_from_rows(rows):
    grouped = {}
    for row in rows or ():
        symbol = row.get("symbol")
        if symbol:
            grouped.setdefault(symbol, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item.get("open_time") or 0))
    return grouped


def portfolio_evidence(rows, oos, starting_equity=10000.0, fee_rate=0.0004,
                       slippage_bps=2.0, cost_bps=12.0, interval=None, settings=None,
                       funding=None, windows=None):
    """Replay the out-of-sample decisions through one shared account.

    Returns the block the registry gate reads, plus the supporting numbers. A caller that
    cannot produce evidence gets an explicit reason rather than an empty dict, so a missing
    result is never mistaken for a passing one.
    """
    # costs_included is only True when funding was actually charged. It used to be a
    # constant: the replay applied fees and slippage, which are the costs the paper account
    # applies at fill time, and never funding, which the live loop charges every tick on
    # every open position. So the flag asserted that all costs were in a number that omitted
    # the one cost a perpetual position pays for as long as it is held -- and the promotion
    # gate reads exactly this flag.
    evidence = {"costs_included": bool(funding), "trades": 0, "net_return": 0.0,
                "max_drawdown": 0.0, "status": "unavailable"}
    series = series_from_rows(rows)
    if len(series) < MIN_SYMBOLS:
        evidence["reason"] = "portfolio_needs_multiple_symbols:%d" % len(series)
        return evidence
    decisions = oos_decisions(oos, cost_bps=cost_bps)
    replayable = {symbol: values for symbol, values in series.items() if decisions.get(symbol)}
    if len(replayable) < MIN_SYMBOLS:
        evidence["reason"] = "too_few_symbols_with_decisions:%d" % len(replayable)
        return evidence
    strategy = decision_strategy(decisions)
    try:
        result = run_portfolio_backtest(
            replayable, starting_equity=starting_equity, fee_rate=fee_rate,
            slippage_bps=slippage_bps, strategy=strategy,
            interval=interval, risk=live_risk(starting_equity, settings),
            limits=live_limits(settings), funding=funding, funding_windows=windows)
    except Exception as exc:
        evidence["reason"] = "replay_failed:%s: %s" % (type(exc).__name__, exc)
        return evidence
    # Whether the replay could actually see the decisions. A key mismatch between the
    # dataset's timestamps and the replay's lookup produced a full, plausible-looking
    # evidence block reporting zero trades, and nothing in it said the decisions had been
    # invisible rather than unprofitable. These two counts say which it was.
    hits = int(getattr(strategy, "stats", {}).get("hits", 0))
    misses = int(getattr(strategy, "stats", {}).get("misses", 0))
    evidence["decision_lookups_hit"] = hits
    evidence["decision_lookups_missed"] = misses
    if decisions and not hits:
        evidence["reason"] = "decisions_never_matched_a_bar:%d" % sum(
            len(v) for v in decisions.values())
    metrics = result.get("metrics") or {}
    trades = result.get("trades") or []
    equity = result.get("equity_curve") or []
    # net_return is the account return after fees, funding and slippage, which is what the
    # gate means by costs_included. Read from the curve rather than from the metric so the
    # two cannot drift apart.
    start = float(starting_equity)
    end = float(equity[-1]) if equity else start
    net_return = (end / start - 1.0) if start else 0.0
    max_drawdown = float(metrics.get("max_drawdown_pct") or 0.0) / 100.0
    evidence.update({
        "status": "ok" if trades else "no_trades",
        "trades": len(trades),
        "net_return": net_return,
        "max_drawdown": max_drawdown,
        "symbols": sorted(replayable),
        "oos_records": len(oos or ()),
        "decisions": sum(len(values) for values in decisions.values()),
        "starting_equity": start,
        "ending_equity": end,
        "total_fees": float(metrics.get("total_fees") or 0.0),
        "total_funding": float(metrics.get("total_funding") or 0.0),
        "win_rate_pct": float(metrics.get("win_rate_pct") or 0.0),
        "sharpe": metrics.get("sharpe"),
        "turnover": float(metrics.get("turnover") or 0.0),
        "cost_bps": float(cost_bps),
    })
    replay_funding = result.get("funding") or {}
    if funding:
        evidence["total_funding"] = float(replay_funding.get("charged") or 0.0)
        evidence["funding_settlements_missed"] = int(replay_funding.get("missed") or 0)
    else:
        # Named, so a reader does not have to infer it from a zero. Kept out of "reason",
        # which reports the replay's primary outcome: a run can be short of trades and short
        # of funding at the same time and one field cannot hold both.
        evidence["costs_missing"] = ["funding"]
        # total_funding reads 0.0 either because funding was charged and summed to nothing or
        # because it was never modelled, and those are different facts. Report the second one
        # by not reporting a number at all.
        evidence.pop("total_funding", None)
    if len(trades) < MIN_TRADES:
        evidence["status"] = "insufficient_trades"
        evidence["reason"] = "trades_below_floor:%d" % len(trades)
    return evidence


def gate_verdict(evidence, max_drawdown=0.2, min_trades=MIN_TRADES):
    """Whether this evidence would pass the registry gate, and why not if it would not.

    Mirrors the registry conditions deliberately, so a training run can say "this would be
    promoted" or "this would be refused because X" without actually promoting anything.
    """
    failures = []
    if not evidence or evidence.get("status") != "ok":
        # Absent evidence is the common case and must not raise: this is called from a
        # training run to report what it would have been refused for, and a run with no
        # evidence at all is exactly the situation it needs to describe.
        return {"passes": False,
                "failures": [(evidence or {}).get("reason") or "no_evidence"]}
    if evidence.get("costs_included") is not True:
        failures.append("costs_not_included")
    if int(evidence.get("trades") or 0) < min_trades:
        failures.append("insufficient_trades")
    for key in ("net_return", "max_drawdown"):
        value = evidence.get(key)
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            failures.append("non_finite_" + key)
    if failures:
        return {"passes": False, "failures": failures}
    if float(evidence["net_return"]) <= 0:
        failures.append("net_return_not_positive")
    if not 0 <= float(evidence["max_drawdown"]) <= max_drawdown:
        failures.append("drawdown_outside_limit")
    return {"passes": not failures, "failures": failures}
