"""Download history, build a pooled dataset, and train candidates.

One command for the offline research path, which otherwise takes three invocations and
a hand-managed argument list:

    python -m scripts.train_pipeline --symbols BTCUSDT,ETHUSDT --interval 5m --days 365

Nothing here places an order: the pipeline only writes candles, a dataset and candidate
model artifacts. Promotion to production stays a deliberate, separate step.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from app.core.config import Settings
from app.features.feature_spec import FEATURE_VERSION
from app.models.fit_model import load
from app.market.ingest import DEFAULT_SYMBOLS, coverage, ingest
from app.models.tabular_model import train_tabular
from app.models.train import DEFAULT_HORIZON, build_dataset

DATASET_PATH = "research_v3/training_dataset.jsonl"


def download(settings, symbols, interval, days, concurrency):
    began = time.time()
    results = asyncio.run(ingest(list(symbols), interval, days, settings.base_url,
                                 settings.data_dir, concurrency))
    failures = [r for r in results if r["errors"]]
    print("ingest: %d symbols, %d rows in %.0fs, %d failures"
          % (len(results), sum(r["written"] for r in results), time.time() - began,
             len(failures)), flush=True)
    for row in coverage(__import__("app.storage", fromlist=["Store"]).Store(settings.data_dir),
                        list(symbols), interval):
        print("  %-12s bars=%-7d coverage=%6.2f%% missing=%-6d largest_gap=%d"
              % (row["symbol"], row["bars"], row["coverage_pct"], row["missing"],
                 row["largest_gap_bars"]), flush=True)
    return failures


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="History -> dataset -> candidates.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--window", type=int, default=settings.lookback)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--cost-bps", type=float, default=12)
    parser.add_argument("--backends", default="lightgbm,catboost")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    dataset_path = args.dataset or str(Path(settings.data_dir) / DATASET_PATH)

    if not args.skip_download:
        failures = download(settings, symbols, args.interval, args.days, args.concurrency)
        if failures:
            print("refusing to build a dataset while ingestion reported failures")
            return 1

    if not args.skip_dataset:
        result = build_dataset(settings.data_dir, symbols, args.interval,
                               horizon=args.horizon, window=args.window,
                               output=dataset_path)
        print("dataset: %s rows=%d window=%d horizon=%d"
              % (result["output"], result["rows"], result["window"], result["horizon"]),
              flush=True)
        empty = [e["symbol"] for e in result["symbols"] if not e["rows"]]
        if empty:
            print("symbols contributing no rows: %s" % ", ".join(empty))

    rows = load(dataset_path)
    output_root = str(Path(settings.data_dir) / "research_v3" / "candidates")
    trained = []
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            result = train_tabular(rows, backend, output_root, args.rounds, args.cost_bps)
        except ValueError as exc:
            print("%s: refused (%s)" % (backend, exc), flush=True)
            continue
        test = result["metrics"]["test"]
        print("%s -> %s" % (backend, result["path"]), flush=True)
        print("  feature_version=%s rows=%d symbols=%d"
              % (result["feature_version"], result["dataset_rows"],
                 len(result["dataset_symbols"])), flush=True)
        print("  test: dir_acc=%.2f%% active=%d (%.1f%%) net_edge=%.2f bps hit=%.2f%%"
              % (test["directional_accuracy_pct"], test["active_samples"],
                 test["active_share_pct"], test["active_net_edge_bps"],
                 test["active_hit_rate_pct"]), flush=True)
        trained.append(result)

    if not trained:
        print("no candidate was trained")
        return 1
    print(json.dumps({"candidates": [r["path"] for r in trained],
                      "feature_version": FEATURE_VERSION}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
