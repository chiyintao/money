"""Out-of-sample walk-forward report for the tabular candidates.

A single chronological split cannot tell an edge from a lucky period. This retrains on an
expanding window and evaluates each successive block, then reports how often the edge was
positive and how wide the uncertainty really is.

    python -m scripts.walkforward_report --dataset data/research_v3/training_dataset.jsonl
"""
import argparse
import json

from app.core.config import Settings
from app.models.fit_model import load
from app.models.walkforward import walk_forward_tabular


def report(result):
    aggregate = result["aggregate"]
    print("%s  folds=%d  purge=%d bars  cost=%s bps"
          % (result["backend"], result["folds"], result["purge_bars"], result["cost_bps"]))
    print("  aggregate: rows=%d active=%d (%.3f%%) dir_acc=%.2f%% edge=%.2f bp"
          % (aggregate["rows"], aggregate["active_samples"], aggregate["active_share_pct"],
             aggregate["dir_acc_pct"], aggregate["net_edge_bps"]))
    print("  folds with activity: %d | positive edge: %d | negative edge: %d"
          % (aggregate["folds_with_activity"], aggregate["folds_positive_edge"],
             aggregate["folds_negative_edge"]))
    boot = result["bootstrap"]
    print("  bootstrap(%d, block=%d): mean=%.2f bp  95%% CI [%.2f, %.2f]  P(edge>0)=%.2f"
          % (boot["iterations"], boot["block"], boot["mean_bps"], boot["low_bps"],
             boot["high_bps"], boot["prob_positive"]))
    print()
    print("  %-5s %-14s %10s %9s %9s %10s"
          % ("fold", "test_start", "test_rows", "active", "dir_acc", "net_edge"))
    for fold in result["per_fold"]:
        print("  %-5d %-14d %10d %9d %8.2f%% %7.2f bp"
              % (fold["fold"], fold["test_start"], fold["test_rows"],
                 fold["active_samples"], fold["directional_accuracy_pct"],
                 fold["active_net_edge_bps"]))


def verdict(result):
    aggregate, boot = result["aggregate"], result["bootstrap"]
    lines = []
    if aggregate["folds_with_activity"] < result["folds"] / 2:
        lines.append("only %d of %d folds traded at all: the threshold rarely fires"
                     % (aggregate["folds_with_activity"], result["folds"]))
    if boot["low_bps"] <= 0 <= boot["high_bps"]:
        lines.append("bootstrap interval straddles zero: no edge is established")
    if aggregate["folds_positive_edge"] and aggregate["folds_negative_edge"]:
        lines.append("edge changes sign across folds: the result is period-dependent")
    if aggregate["active_samples"] < 1000:
        lines.append("only %d active trades across the whole run" % aggregate["active_samples"])
    return lines or ["no structural objection raised by these checks"]


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="Walk-forward a trained candidate.")
    parser.add_argument("--dataset", default="data/research_v3/training_dataset.jsonl")
    parser.add_argument("--backends", default="lightgbm")
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--cost-bps", type=float, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args(argv)

    rows = load(args.dataset)
    print("dataset %s rows=%d" % (args.dataset, len(rows)), flush=True)
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        result = walk_forward_tabular(rows, backend, args.folds, args.cost_bps,
                                      args.rounds, args.horizon)
        report(result)
        for line in verdict(result):
            print("  ! %s" % line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
