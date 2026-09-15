"""A/B the two sizing fixes on the same traded history.

The comparison is deliberately narrow. It does not re-run the simulation, re-train a model
or re-derive signals: it takes the trades that actually happened, each with the entry price,
stop, exit price, quantity and net edge recorded at the moment the decision was taken, and
asks one question -- what would this account have returned under each sizing rule?

Holding the trades fixed is the point. A re-simulation would change which signals fired as
well as how they were sized, and the two effects would be inseparable. Here the only thing
that varies between arms is the fraction of the risk budget each signal received.

The rules
---------
baseline   the shipped behaviour: risk_budget * edge_scale, where edge_scale is the linear
           ramp edge_bps / sizing_reference_edge_bps, capped at 1.
reference  fix A -- lower the reference from 25 bps to a value taken from the measured
           distribution of the edges that actually traded.
floor      fix B -- keep the reference, but guarantee a minimum fraction of the budget to
           any signal that cleared the gate.

Both fixes act through the same RiskEngine, so the arm definitions are constructor
arguments rather than reimplementations. A reimplementation would measure the author's
understanding of the rule instead of the rule.
"""
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.trading.risk import RiskEngine  # noqa: E402

DB = "data/research.sqlite3"
INITIAL_EQUITY = 10000.0
# The per-trade stop is where the risk is realised, so "risk" below always means
# quantity * |entry - stop| -- the cash the position loses if the stop is hit -- and never
# the notional, which for a 10x book is a different number by an order of magnitude.
FEE_RATE = 0.0004


def load_trades(db=DB):
    """Every trade that happened, with the edge recorded when the decision was taken.

    The edge is looked up from the strategy_decision event stream rather than stored on the
    trade, because that is where the serving layer writes it and reading it from anywhere
    else would be reading a copy that can drift from the original.
    """
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    events = []
    # Both tables, and that is not optional. \`events\` holds only the live session's decisions;
    # every earlier session's were moved to \`events_archive\` by retention. Reading one table
    # left 140 of 142 trades with no edge at all, and the arms then sized them identically --
    # producing a three-way comparison that was really three copies of the same run. The
    # giveaway was a maximum difference of a few dollars between arms that were supposed to
    # differ by a factor of ten.
    for table in ("events", "events_archive"):
        for event_time, payload in cursor.execute(
                "SELECT event_time,payload FROM %s WHERE type='strategy_decision' "
                "ORDER BY event_time" % table):
            data = json.loads(payload)
            decision = data.get("decision") or {}
            if decision.get("side") not in ("LONG", "SHORT"):
                continue
            events.append({"time": event_time, "symbol": data.get("symbol"),
                           "session": data.get("session_id"),
                           "side": decision.get("side"),
                           "edge_bps": decision.get("edge_bps"),
                           "price": (data.get("market") or {}).get("price")})
    events.sort(key=lambda item: item["time"])
    trades = []
    for row in cursor.execute(
            "SELECT id,timestamp,symbol,side,entry,exit,qty,pnl,fees,payload FROM trades "
            "ORDER BY id"):
        trade_id, stamp, symbol, side, entry, exit_price, qty, pnl, fees, payload = row
        detail = json.loads(payload or "{}")
        stop = detail.get("initial_stop") or detail.get("stop_price")
        if not (entry and exit_price and qty and stop):
            continue
        # The edge for this trade is the most recent decision event on the same symbol and
        # side at or before the entry, which is the value the sizer was handed.
        candidates = [e for e in events
                      if e["symbol"] == symbol and e["side"] == side
                      and e["edge_bps"] is not None
                      and e["time"] <= stamp + 60000]
        edge = candidates[-1]["edge_bps"] if candidates else None
        trades.append({"id": trade_id, "time": stamp, "symbol": symbol, "side": side,
                       "entry": float(entry), "exit": float(exit_price), "qty": float(qty),
                       "stop": float(stop), "pnl": pnl or 0.0, "fees": fees or 0.0,
                       "edge_bps": edge, "session": detail.get("session_id"),
                       "notional": detail.get("entry_notional") or float(entry) * float(qty),
                       "leverage": detail.get("leverage")})
    return trades


def simulate(trades, engine, equity=INITIAL_EQUITY):
    """Replay the trades through one sizing rule, chronologically, compounding equity.

    Equity is carried forward rather than fixed at the starting value. The first pass used a
    constant 10000 and that quietly changed the answer: every position was sized off the
    initial balance, so a losing run never reduced the next bet and the arms could not
    diverge in the way the live account would. Compounding is what makes the comparison a
    statement about an account rather than about a sequence of bets.
    """
    ordered = sorted(trades, key=lambda t: t["time"])
    current = float(equity)
    rows = []
    for trade in ordered:
        plan = {"entry": trade["entry"], "stop": trade["stop"],
                "target": trade["entry"] + (trade["entry"] - trade["stop"]) * 1.8,
                "symbol": trade["symbol"], "side": trade["side"]}
        sized = engine.size(plan, current, edge_bps=trade["edge_bps"])
        if not sized or sized["quantity"] <= 0:
            rows.append({**trade, "scale_rule": 0.0, "risk_cash": 0.0,
                         "applied_notional": 0.0, "applied_pnl": 0.0})
            continue
        unit_risk = abs(trade["entry"] - trade["stop"])
        new_qty = sized["quantity"]
        # The realised P&L is rescaled by the quantity ratio, which is exact for a linear
        # instrument: the recorded trade was one specific size, and a different size earns
        # the same per-unit amount minus the fees, which also scale.
        ratio = new_qty / trade["qty"] if trade["qty"] else 0.0
        applied_pnl = trade["pnl"] * ratio
        applied_notional = trade["entry"] * new_qty
        current += applied_pnl
        rows.append({**trade, "scale_rule": new_qty / trade["qty"] if trade["qty"] else 0.0,
                     "risk_cash": new_qty * unit_risk, "applied_notional": applied_notional,
                     "applied_pnl": applied_pnl, "equity_after": current,
                     "binding": sized.get("binding")})
    return {"final_equity": current, "return_pct": (current / equity - 1.0) * 100.0,
            "rows": rows}


def summarise(name, result, trades):
    rows = result["rows"]
    traded = [r for r in rows if r["applied_notional"] > 0]
    gross = sum(r["applied_notional"] for r in rows)
    fees = sum((r["fees"] or 0.0) * (r["applied_notional"] / r["notional"] if r["notional"] else 0)
               for r in rows)
    wins = [r for r in traded if r["applied_pnl"] > 0]
    risk = [r["risk_cash"] for r in traded] or [0.0]
    return {
        "arm": name,
        "final_equity": result["final_equity"],
        "return_pct": result["return_pct"],
        "trades": len(traded),
        "gross_notional": gross,
        "fees": fees,
        "net": result["final_equity"] - INITIAL_EQUITY,
        "hit_rate": (len(wins) / len(traded)) if traded else 0.0,
        "mean_risk_cash": sum(risk) / len(risk),
        "max_risk_cash": max(risk),
        "mean_notional": (sum(r["applied_notional"] for r in traded) / len(traded))
        if traded else 0.0,
    }


def reference_from_history(trades, percentile=0.75):
    """The reference edge fix A uses, taken from the edges that actually traded.

    The shipped 25 bps was never observed: the maximum net edge on any traded signal in
    this session is 9.89 bps and the median is 1.57, so the ramp spent its whole life in
    its first tenth. Taking a percentile of the realised distribution replaces a number
    that was assumed with one that was measured.
    """
    edges = sorted(t["edge_bps"] for t in trades
                   if t["edge_bps"] is not None and t["edge_bps"] > 0)
    if not edges:
        return None
    index = min(len(edges) - 1, max(0, int(round(percentile * (len(edges) - 1)))))
    return edges[index]


def build_arms(trades, floor=0.30, percentile=0.75):
    """The three configurations, all through the same engine."""
    reference = reference_from_history(trades, percentile)
    common = dict(max_risk=0.01, max_portfolio_risk=0.04, target_exposure=3.0,
                  max_symbol_leverage=4.0, max_gross_leverage=8.0)
    return {
        "baseline": RiskEngine(sizing_reference_edge_bps=25.0, **common),
        "reference": RiskEngine(sizing_reference_edge_bps=reference if reference else 25.0,
                                **common),
        "floor": RiskEngine(sizing_reference_edge_bps=25.0, sizing_floor_scale=floor,
                            **common),
    }, reference


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="A/B the two position-sizing fixes.")
    parser.add_argument("--floor", type=float, default=0.30,
                        help="minimum fraction of the risk budget for a gated signal")
    parser.add_argument("--percentile", type=float, default=0.75,
                        help="percentile of realised traded edges used as the reference")
    parser.add_argument("--session", default=None, help="restrict to one session id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    trades = load_trades()
    if args.session:
        trades = [t for t in trades if t["session"] == args.session]
    if not trades:
        print("no trades to replay")
        return 1
    arms, reference = build_arms(trades, floor=args.floor, percentile=args.percentile)
    results, replays = {}, {}
    for name, engine in arms.items():
        replays[name] = simulate(trades, engine)
        results[name] = summarise(name, replays[name], trades)

    payload = {"trades": len(trades), "initial_equity": INITIAL_EQUITY,
               "reference_edge_bps": reference, "floor": args.floor,
               "percentile": args.percentile, "arms": results}
    if args.json:
        print(json.dumps(payload, indent=1, default=str))
        return 0
    edged = [t for t in trades if t["edge_bps"] is not None]
    print("trades replayed: %d   with a recorded edge: %d   initial equity: %.2f"
          % (len(trades), len(edged), INITIAL_EQUITY))
    if len(edged) < len(trades):
        # Said out loud because it bounds the conclusion. A trade with no edge is sized at
        # scale 1.0 by every arm alike, so the arms only separate on the trades that carry
        # one -- and decisions were not always logged: the largest session's event stream
        # covers its last 25 minutes while its trades span twice that, so 58 trades have no
        # edge and no arm can be judged on them.
        print("  NOTE: %d trades have no recorded edge (decision logging did not cover "
              "them); they are sized identically in every arm, so the comparison rests on "
              "the other %d." % (len(trades) - len(edged), len(edged)))
    print("reference edge for fix A: %s bps (%.0fth percentile of traded edges; shipped "
          "default 25.0)" % (reference, args.percentile * 100))
    print()
    print("%-11s %12s %10s %12s %11s %11s %8s" % (
        "arm", "final eq", "return", "mean risk$", "max risk$", "gross notl", "hit"))
    for name in ("baseline", "reference", "floor"):
        s = results[name]
        print("%-11s %12.2f %9.3f%% %12.2f %11.2f %11.0f %7.0f%%" % (
            s["arm"], s["final_equity"], s["return_pct"], s["mean_risk_cash"],
            s["max_risk_cash"], s["gross_notional"], s["hit_rate"] * 100))
    print()
    base = results["baseline"]
    for name in ("reference", "floor"):
        s = results[name]
        print("%-11s net %+.4f  vs baseline %+.4f  -> difference %+.4f" % (
            name, s["net"], base["net"], s["net"] - base["net"]))
    print()
    # \`applied_pnl\` is the recorded net P&L, which already has fees and funding taken out
    # of it. Adding the fee column back on gives the gross figure; reporting the net one as
    # "gross" made the fee share read 1247%, which is how the mistake was caught.
    print("%-11s %10s %12s %12s %11s" % ("arm", "fees", "gross pnl", "net pnl", "fee/gross"))
    for name in ("baseline", "reference", "floor"):
        s = results[name]
        net_pnl = s["net"]
        # \`net_pnl\` here is the account result (final equity minus the start), and the fees
        # are a cost inside it rather than something to add back. The gross P&L the fees are
        # a share OF is therefore \`net + fees\`, and when that is negative the ratio is not a
        # meaningful percentage at all -- printing one anyway produced the 1247% that gave
        # this line away. The raw numbers are shown instead, which is the honest form.
        gross_pnl = net_pnl + s["fees"]
        share = (s["fees"] / gross_pnl) * 100 if gross_pnl > 0 else None
        shown = ("%9.1f%%" % share) if share is not None else "n/a(loss)"
        print("%-11s %10.4f %12.4f %12.4f %11s" % (name, s["fees"], gross_pnl, net_pnl, shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

