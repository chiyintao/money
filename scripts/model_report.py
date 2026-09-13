"""Print a per-symbol report for trained candidates.

A pooled accuracy figure hides which symbol carries the model and which loses on every
trade. This prints the breakdown that the manifest already records, plus the numbers
that decide whether an edge survives the round-trip cost.

    python -m scripts.model_report --candidate data/research_v3/candidates/lightgbm-xxxx
"""
import argparse
import json
from pathlib import Path

from app.core.config import Settings


def load_manifest(candidate):
    path = Path(candidate)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def latest(root, backend):
    folders = sorted(Path(root).glob(backend + "-*"), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return folders[0] if folders else None


def summarise(manifest):
    print("%s  version=%s  features=%s"
          % (manifest["backend"], manifest.get("feature_version"), manifest.get("features")))
    print("  dataset: %d rows, %d symbols, cost=%s bps"
          % (manifest.get("dataset_rows", 0), len(manifest.get("dataset_symbols", [])),
             manifest.get("cost_bps")))
    split = manifest.get("split", {})
    print("  split: train=%s validation=%s test=%s purged=%s"
          % (split.get("train"), split.get("validation"), split.get("test"),
             split.get("purged")))
    print()
    print("  %-11s %8s %9s %11s %10s %10s" % ("partition", "rows", "dir_acc",
                                              "active", "active%", "net_edge"))
    for part in ("train", "validation", "test"):
        stats = manifest["metrics"][part]
        print("  %-11s %8d %8.2f%% %11d %9.2f%% %7.2f bp"
              % (part, stats["rows"], stats["directional_accuracy_pct"],
                 stats["active_samples"], stats["active_share_pct"],
                 stats["active_net_edge_bps"]))
    print()
    print("  test by symbol:")
    print("  %-11s %10s %9s %11s %10s" % ("symbol", "dir_acc", "active", "net_edge",
                                          "hit_rate"))
    for symbol, stats in sorted(manifest.get("metrics_by_symbol", {}).items()):
        print("  %-11s %9.2f%% %9d %8.2f bp %9.1f%%"
              % (symbol, stats["directional_accuracy_pct"], stats["active_samples"],
                 stats["active_net_edge_bps"], stats["active_hit_rate_pct"]))


def verdict(manifest):
    test = manifest["metrics"]["test"]
    lines = []
    if test["directional_accuracy_pct"] < 50:
        lines.append("directional accuracy is below 50%: no demonstrated directional skill")
    if test["active_samples"] < 1000:
        lines.append("only %d active test samples: too few to support any claim"
                     % test["active_samples"])
    edges = [s["active_net_edge_bps"] for s in manifest.get("metrics_by_symbol", {}).values()
             if s["active_samples"] > 0]
    if edges and min(edges) < 0 < max(edges):
        lines.append("per-symbol net edge changes sign: the pooled figure is not stable")
    return lines


def main(argv=None):
    settings = Settings()
    root = Path(settings.data_dir) / "research_v3" / "candidates"
    parser = argparse.ArgumentParser(description="Report trained candidates.")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--backends", default="lightgbm,catboost")
    args = parser.parse_args(argv)

    targets = ([args.candidate] if args.candidate
               else [latest(args.root, b) for b in args.backends.split(",")])
    for target in [t for t in targets if t]:
        manifest = load_manifest(target)
        summarise(manifest)
        for line in verdict(manifest):
            print("  ! %s" % line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
