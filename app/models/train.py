"""Build a pooled training dataset from stored candles.

Rows from every symbol are pooled into one file so a single model can be trained
across markets. Two details matter for that to work:

* the feature window matches the live decision loop (settings.lookback), because a
  model trained on a different window than it is served would be quietly wrong;
* each label records label_end_time, the bar the forward return actually resolves at,
  so the chronological split can purge rows whose label crosses a boundary.

The builder walks a bounded window instead of the whole history before each row, which
turns the previous O(bars^2) pass into O(bars * window).
"""
import argparse
import json
import os
from pathlib import Path

from ..core.config import Settings
from ..features.feature_spec import FEATURES, LABEL_CLIP
from ..features.feature_source import FeatureSource
from ..features.feature_spec import FEATURE_VERSION, LABEL_VERSION
from ..market.funding import funding_series
from . import labels as labelling
from ..trading.exit_policy import get_policy, plan_levels
from ..features.quality import validate_ohlcv
from ..storage.storage import Store

DEFAULT_HORIZON = 12  # forward bars; 12 x 5m = one hour


def build_dataset(data_dir="data", symbols=("BTCUSDT", "ETHUSDT"), interval="5m",
                  limit=250000, horizon=DEFAULT_HORIZON, window=None, output=None,
                  strict=True, label_clip=LABEL_CLIP, max_basis_age_ms=None,
                  labels="triple_barrier", exit_policy=None):
    """Write a pooled jsonl dataset and return a per-symbol report.

    Raises when a symbol fails OHLCV validation and ``strict`` is set, because a
    silently skipped symbol shows up later as an unexplained coverage hole.
    """
    if horizon < 1:
        raise ValueError("invalid_horizon")
    window = int(window or Settings().lookback)
    if window < 50:
        raise ValueError("window_below_warmup")

    # One worker per symbol. The work is independent per symbol and it is the longest
    # stage of a run: a single-threaded walk over fifteen symbols left fifteen cores idle
    # for sixteen minutes. Each worker opens its own store, because a SQLite connection
    # cannot be shared across processes. MODEL_DATASET_WORKERS=1 restores the serial path
    # exactly, and the rows and the report are identical either way.
    options = dict(data_dir=data_dir, interval=interval, limit=limit, horizon=horizon,
                   window=window, strict=strict, max_basis_age_ms=max_basis_age_ms,
                   labels=labels, exit_policy=exit_policy, label_clip=label_clip)
    workers = _dataset_workers(len(symbols))
    rows, report = [], []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [(symbol, pool.submit(_build_symbol, symbol, options))
                       for symbol in symbols]
            for symbol, future in pending:
                symbol_rows, entry = future.result()
                rows.extend(symbol_rows)
                report.append(entry)
    else:
        for symbol in symbols:
            symbol_rows, entry = _build_symbol(symbol, options)
            rows.extend(symbol_rows)
            report.append(entry)

    rows.sort(key=lambda item: (item["timestamp"], item["symbol"]))
    destination = (Path(output) if output is not None
                   else Path(data_dir) / "training_dataset.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"output": str(destination), "rows": len(rows), "window": window,
            "horizon": horizon, "interval": interval, "symbols": report,
            "label_version": LABEL_VERSION, "feature_version": FEATURE_VERSION,
            "label_entry": "next_bar_open", "label_scheme": (
                labelling.LABEL_SCHEME if labels == "triple_barrier" else "fixed-horizon-v1"),
            "barrier_counts": _barrier_counts(rows)}


def _dataset_workers(symbols):
    """How many symbols to build at once.

    Half the machine by default: the service is streaming quotes in the same process tree
    while a run is in flight, and a build that takes every core makes the dashboard stop
    answering. MODEL_DATASET_WORKERS overrides it, and 1 is the old serial behaviour.
    """
    raw = os.getenv("MODEL_DATASET_WORKERS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return max(1, min(value, symbols))
    # Capped at four, and that cap is about memory rather than cores. Each worker holds
    # one symbol's candles and builds its rows in memory, and the service is holding the
    # whole dataset from the previous stage at the same time. Twelve workers on a
    # sixteen-gigabyte host pushed it to 91% and the operating system started killing
    # processes. The speed-up from four is most of the eight-worker speed-up because the
    # per-symbol work is not uniform; the rest is not worth an out-of-memory kill.
    return max(1, min(symbols, 4, (os.cpu_count() or 2) // 2))


def _build_symbol(symbol, options):
    """Every row one symbol contributes, and the entry that describes it.

    Split out of build_dataset so it can run in a worker process. It opens its own store:
    the connection belongs to the process that made it, and a worker is not that process.
    """
    data_dir = options["data_dir"]
    interval = options["interval"]
    limit = options["limit"]
    horizon = options["horizon"]
    window = options["window"]
    strict = options["strict"]
    max_basis_age_ms = options["max_basis_age_ms"]
    labels = options["labels"]
    exit_policy = options["exit_policy"]
    label_clip = options["label_clip"]
    rows = []
    store = Store(data_dir)
    try:
        bars = store.candles(symbol, interval, limit, True)
        entry = {"symbol": symbol, "bars": len(bars), "rows": 0, "quality": None}
        if len(bars) < window + horizon:
            entry["quality"] = {"ready": False, "reason": "insufficient_bars"}
            report.append(entry)
            if strict:
                raise ValueError({"symbol": symbol, "reason": "insufficient_bars",
                                  "bars": len(bars), "need": window + horizon})
            return rows, report
        step = int(bars[0]["close_time"]) - int(bars[0]["open_time"]) + 1
        quality = validate_ohlcv(bars, step)
        entry["quality"] = {"ready": quality["ready"],
                            "error_count": quality["error_count"],
                            "errors": quality["errors"][:3]}
        if not quality["ready"] and strict:
            raise ValueError({"symbol": symbol, "quality": quality})
        funding_times, _funding_rates, _funding_marks = funding_series(store, symbol)
        entry["funding_records"] = len(funding_times)
        # The same object the live service uses. Two separate feature paths is exactly
        # how the funding family came to be present in training and absent at serving.
        # reuse=True: the store is not written during the build, so the derivative
        # series is constant for the whole walk. Without it every row re-reads the
        # funding and mark series -- 150,000 rows times two queries -- to guard against
        # a write that cannot happen, which measured at 313 rows/sec.
        source = FeatureSource(store, ttl_ms=0, max_basis_age_ms=max_basis_age_ms,
                               reuse=True)
        # The barriers come from the same exit-policy function the live path calls, so a
        # change to the policy moves the labels with it. Copying its constants here is
        # how the training label and the traded outcome drifted apart in the first place.
        policy = exit_policy if exit_policy is not None else get_policy()
        produced = 0
        degraded = 0
        # Which features fell back, not just how often. A dataset whose funding
        # columns are placeholders trains a model that cannot be served; one whose
        # collected families are empty is a different model, and the two used to
        # report the same single number.
        missing_seen = {}
        # One extra bar of headroom: the label now enters at bar index+1 and resolves
        # horizon bars later, so it needs index+1+horizon to exist.
        for index in range(window - 1, len(bars) - horizon - 1):
            bar_time = int(bars[index]["close_time"])
            # Funding state as published at or before this bar; a later settlement
            # would be look-ahead. FeatureSource applies that rule itself.
            features, fell_back = source.snapshot(
                bars[index - window + 1:index + 1], symbol, bar_time,
                float(bars[index]["close"]))
            if fell_back:
                degraded += 1
                for name in fell_back:
                    missing_seen[name] = missing_seen.get(name, 0) + 1
            row = {name: features[name] for name in FEATURES}
            row["symbol"] = symbol
            row["timestamp"] = bars[index]["close_time"]
            # The bar the decision becomes tradeable on. Its open is the first price a
            # market order could have got after the signal, so it is the entry the
            # label is measured from; measuring from close[index] assumed a fill at the
            # very price that produced the signal.
            entry_bar = bars[index + 1]
            row["entry_time"] = int(entry_bar["open_time"])
            entry_price = float(entry_bar["open"])
            row["entry_price"] = entry_price
            if labels == "triple_barrier":
                # The label is what the exit policy would actually have produced:
                # whichever of the stop, the target or the time limit is touched first,
                # measured from the price the entry could really have got. A
                # fixed-horizon return describes a strategy nobody runs -- there is a
                # stop and a target, and the trade leaves when one of them is hit.
                #
                # The barriers are SYMMETRIC, and that is not a simplification. The
                # label is built before anyone knows which way the model will call it,
                # so putting the policy's 1.8R target above and its 1R stop below would
                # mean the upper barrier is 1.8x further away than the lower one -- the
                # label would then be biased upward for reasons that have nothing to do
                # with the market, and a model trained on it would learn to predict
                # that bias. Using the risk unit on both sides makes the label an
                # unbiased statement about which way price moved first.
                atr = float(row.get("atr_pct") or 0.0) * entry_price
                levels = plan_levels(policy, entry_price, atr, "LONG")
                if levels is None:
                    continue
                distance = levels["stop_distance"]
                outcome = labelling.triple_barrier(
                    bars, index + 1, distance, distance, horizon,
                    entry_price=entry_price)
            else:
                exit_bar = bars[index + 1 + horizon]
                outcome = {
                    "future_return": float(exit_bar["close"]) / entry_price - 1,
                    "barrier": "fixed_horizon", "bars_held": horizon,
                    "barrier_time": int(exit_bar["close_time"])}
            row["label_end_time"] = int(outcome["barrier_time"])
            row["barrier"] = outcome["barrier"]
            row["bars_held"] = int(outcome["bars_held"])
            row["label_start"] = int(bar_time)
            forward = outcome["future_return"]
            if label_clip is not None:
                # A single -56% flash bar otherwise contributes a squared error some
                # 775x a typical label and dominates the fit.
                forward = max(-label_clip, min(label_clip, forward))
            row["future_return"] = forward
            rows.append(row)
            produced += 1
        entry["rows"] = produced
        # A dataset whose feature columns are placeholders trains a model that cannot
        # be served, so the count is reported rather than left implicit -- and named,
        # because "9.7% of rows are degraded" does not say whether to rebuild the
        # funding history or wait for the flow collector.
        entry["degraded_rows"] = degraded
        entry["degraded_features"] = dict(sorted(missing_seen.items()))
    finally:
        store.close()
    return rows, entry


def _barrier_counts(rows):
    """How the labels resolved, so a report says what the model is predicting."""
    counts = {}
    for row in rows:
        name = row.get("barrier") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def main(argv=None):
    settings = Settings()
    parser = argparse.ArgumentParser(description="Build a pooled training dataset.")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default=settings.interval)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--window", type=int, default=settings.lookback)
    parser.add_argument("--limit", type=int, default=250000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-partial", action="store_true",
                        help="skip symbols that fail quality instead of failing")
    parser.add_argument("--no-clip-label", action="store_true",
                        help="keep unbounded forward returns (flash bars dominate the fit)")
    args = parser.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    result = build_dataset(args.data_dir, symbols, args.interval, args.limit,
                           args.horizon, args.window, args.output,
                           strict=not args.allow_partial,
                           label_clip=None if args.no_clip_label else LABEL_CLIP)
    print(json.dumps({k: v for k, v in result.items() if k != "symbols"},
                     ensure_ascii=False))
    for entry in result["symbols"]:
        quality = entry["quality"] or {}
        print("  %-12s bars=%-7d rows=%-7d quality=%s"
              % (entry["symbol"], entry["bars"], entry["rows"], quality.get("ready")),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
