"""Command line for the replacement prediction pipeline.

Audits a dataset, checks whether its label can pay for its own trading costs, validates a
model with combinatorial purged CV, and evaluates the cross-sectional momentum signal that
the horizon study identified. Nothing here places an order and nothing here promotes a model.

    python -m scripts.research_v2 audit    --dataset data/research_v3/training_dataset_mainstream.jsonl
    python -m scripts.research_v2 validate --dataset data/research_v3/training_dataset_mainstream.jsonl
    python -m scripts.research_v2 momentum --cost-bps 6
"""
import argparse
import json
import sqlite3
import sys

from app.models import improved_training, label_economics
from app.models.feature_audit import audit_dataset
from app.strategy import cross_sectional as cs

# Symbols with enough history to be worth ranking. Taken from the store rather than
# hardcoded downstream, so the cross-section grows as collection does.
MIN_BARS = 50_000


def load_universe(db="data/research.sqlite3", symbols=None, min_bars=MIN_BARS):
    """Read daily closes per symbol straight from the candle store."""
    connection = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        if symbols:
            names = list(symbols)
        else:
            names = [row[0] for row in connection.execute(
                "SELECT symbol, COUNT(*) n FROM candles WHERE interval='5m' "
                "GROUP BY symbol HAVING n >= ? ORDER BY n DESC", (min_bars,))]
        universe = {}
        for symbol in names:
            rows = connection.execute(
                "SELECT open_time, close FROM candles WHERE symbol=? AND interval='5m' "
                "ORDER BY open_time", (symbol,)).fetchall()
            if len(rows) >= min_bars:
                universe[symbol] = [{"open_time": r[0], "close": r[1]} for r in rows]
        return universe
    finally:
        connection.close()


def command_audit(args):
    report = audit_dataset(args.dataset)
    print("rows %d, features %d" % (report["rows"], report["features"]))
    print("counts:", report["counts"])
    print("wasted share: %.1f%%" % (report["wasted_share"] * 100))
    for name in report["dead"]:
        print("  DEAD       %s" % name)
    for name in report["degenerate"]:
        print("  DEGENERATE %s" % name)
    print("informative features: %d -> %s"
          % (len(report["informative"]), ", ".join(report["informative"][:40])))
    return 0


def command_economics(args):
    from app.models.dataset_io import iter_rows
    outcomes, barriers = [], []
    for count, row in enumerate(iter_rows(args.dataset)):
        outcomes.append(row.get("future_return") or 0.0)
        barriers.append(row.get("barrier"))
        if args.limit and count + 1 >= args.limit:
            break
    report = label_economics.assess(outcomes, barriers, profile=args.profile,
                                    target_r2=args.target_r2)
    for line in label_economics.verdict_lines(report):
        print("  " + line)
    print()
    print("feasible: %s" % report.get("feasible"))
    return 0 if report.get("feasible") else 1


def command_validate(args):
    report = improved_training.run(args.dataset, args.output, limit=args.limit,
                                   groups=args.groups, k=args.k, cost_bps=args.cost_bps,
                                   rounds=args.rounds, allow_infeasible=True)
    print("status: %s   rows: %d   elapsed: %.1fs"
          % (report["status"], report["rows"], report["elapsed_s"]))
    print("audit: %d dead, %d degenerate, %d informative"
          % (len(report["audit"]["dead"]), len(report["audit"]["degenerate"]),
             len(report["audit"]["informative"])))
    validation = report.get("validation")
    if validation:
        print("CPCV: %d splits, %d paths, mean edge %.2f bps"
              % (validation["splits"], validation["paths"], validation["net_edge_bps"]))
        print("paths positive %d/%d, path Sharpe %.2f"
              % (validation.get("paths_positive", 0), validation["paths"],
                 validation.get("path_sharpe", 0.0)))
    print()
    for line in report["verdict"]:
        print("  " + line)
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    return 0


def command_momentum(args):
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",")) if args.symbols else None
    universe = load_universe(args.db, symbols, args.min_bars)
    print("universe: %d symbols" % len(universe))
    result = cs.backtest(universe, lookback=args.lookback, hold=args.hold,
                         quantile=args.quantile, cost_bps=args.cost_bps)
    for line in cs.evidence(result):
        print("  " + line)
    if args.scan:
        print()
        print("%8s %14s %14s %14s" % ("lookback", "hold=1", "hold=3", "hold=7"))
        for lookback in (1, 2, 3, 5, 7, 10, 14, 21, 30):
            cells = []
            for hold in (1, 3, 7):
                scan = cs.backtest(universe, lookback=lookback, hold=hold,
                                   quantile=args.quantile, cost_bps=args.cost_bps)
                cells.append("%.0f/%.1f bps/t" % (scan["summary"]["net_bps_mean"],
                                                  scan["summary"].get("t_stat", 0.0))
                             if scan.get("ready") else "-")
            print("%8d %14s %14s %14s" % (lookback, *cells))
    return 0 if result.get("ready") and result["summary"]["net_bps_mean"] > 0 else 1


def command_track(args):
    """Record today's momentum target, settle what is due, and report the forward record."""
    from app.storage.storage import Store
    from app.strategy import forward_track as ft

    store = Store(args.data_dir)
    tracker = ft.ForwardTracker(store, lookback=args.lookback, hold=args.hold,
                                quantile=args.quantile, cost_bps=args.cost_bps)
    if args.settle_only:
        result = tracker.settle()
        print("settled %d targets" % result["settled"])
    else:
        record = tracker.record()
        if record.get("recorded"):
            print("recorded day %s: %d longs, %d shorts, universe %d"
                  % (record["day"], len(record["longs"]), len(record["shorts"]),
                     record["universe"]))
        else:
            print("not recorded: %s" % record.get("reason"))
        settled = tracker.settle()
        if settled["settled"]:
            print("settled %d targets" % settled["settled"])
    print()
    for line in tracker.verdict():
        print("  " + line)
    if args.json:
        print(json.dumps(tracker.report(), indent=1, default=str))
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description="Research pipeline v2.")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="report features that carry no information")
    audit.add_argument("--dataset", default="data/research_v3/training_dataset_mainstream.jsonl")
    audit.set_defaults(func=command_audit)

    economics = sub.add_parser("economics", help="can this label pay for its costs")
    economics.add_argument("--dataset", default="data/research_v3/training_dataset_mainstream.jsonl")
    economics.add_argument("--profile", default="taker", choices=sorted(label_economics.COST_PROFILES))
    economics.add_argument("--target-r2", type=float, default=0.003)
    economics.add_argument("--limit", type=int, default=200000)
    economics.set_defaults(func=command_economics)

    validate = sub.add_parser("validate", help="combinatorial purged CV")
    validate.add_argument("--dataset", default="data/research_v3/training_dataset_mainstream.jsonl")
    validate.add_argument("--output", default=None)
    validate.add_argument("--limit", type=int, default=120000)
    validate.add_argument("--groups", type=int, default=6)
    validate.add_argument("--k", type=int, default=2)
    validate.add_argument("--cost-bps", type=float, default=6.0)
    validate.add_argument("--rounds", type=int, default=150)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=command_validate)

    momentum = sub.add_parser("momentum", help="cross-sectional momentum evidence")
    momentum.add_argument("--db", default="data/research.sqlite3")
    momentum.add_argument("--symbols", default=None)
    momentum.add_argument("--min-bars", type=int, default=MIN_BARS)
    momentum.add_argument("--lookback", type=int, default=cs.DEFAULT_LOOKBACK_DAYS)
    momentum.add_argument("--hold", type=int, default=cs.DEFAULT_HOLD_DAYS)
    momentum.add_argument("--quantile", type=float, default=cs.DEFAULT_QUANTILE)
    momentum.add_argument("--cost-bps", type=float, default=6.0)
    momentum.add_argument("--scan", action="store_true")
    momentum.set_defaults(func=command_momentum)

    track = sub.add_parser("track", help="record and settle the forward momentum record")
    track.add_argument("--data-dir", default="data")
    track.add_argument("--lookback", type=int, default=cs.DEFAULT_LOOKBACK_DAYS)
    track.add_argument("--hold", type=int, default=cs.DEFAULT_HOLD_DAYS)
    track.add_argument("--quantile", type=float, default=cs.DEFAULT_QUANTILE)
    track.add_argument("--cost-bps", type=float, default=6.0)
    track.add_argument("--settle-only", action="store_true")
    track.add_argument("--json", action="store_true")
    track.set_defaults(func=command_track)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())