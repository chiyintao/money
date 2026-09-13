"""Long-running training runs, persisted so the dashboard can follow them.

A run is a sequence of stages: pick a universe, refresh its history, build a pooled
dataset, evaluate out-of-sample, and only then train. The order matters -- training first
and reporting afterwards is how a model with no edge gets shipped.

The evaluation is walk-forward rather than one train/test split. A single split gave
+35.8bp on this project while eleven walk-forward folds gave -17.1bp; the split was luck,
and it is exactly the number a single split would have put on the dashboard.

Heavy stages are CPU-bound, so they run in a thread executor and the job only reports
progress from the event loop. State is persisted on every transition, so a restart loses
at most the stage in flight.
"""
import asyncio
import gc
import json
import time
import uuid
from pathlib import Path

from ..core.config import Settings
from ..core.config import interval_to_ms
from ..features.universe import RULES, TIER_EXCLUDED, TIER_MAINSTREAM, TIER_SPECULATIVE, classify, summarise, symbols_for

RUN_KEY = "training_runs"
MAX_LOGS = 400

DEFAULTS = {"interval": "5m", "days": 365, "horizon": 12, "folds": 12, "rounds": 200,
            "cost_bps": 12, "max_symbols": 15, "min_prob_positive": 0.95,
            "min_active_samples": 500}


def _now():
    return int(time.time() * 1000)


def dataset_path(data_dir, tier):
    """Per-tier dataset file.

    Written as an explicit function because the obvious inline expression
    Path(data_dir) / "research_v3" / "training_dataset_%s.jsonl" % tier binds the
    percent operator to the Path: / and % share precedence and associate left to right.
    That raised a TypeError and aborted every run before any model was trained.
    """
    return str(Path(data_dir) / "research_v3" / ("training_dataset_%s.jsonl" % tier))


def candidates_root(data_dir, tier):
    return str(Path(data_dir) / "research_v3" / ("candidates_%s" % tier))


def live_candidates_root(data_dir):
    """The folder the live runtime actually loads models from.

    Training writes per-tier folders so runs never overwrite each other, which means a
    result is invisible to the agent until it is copied here. Promotion is that bridge,
    and it only happens for a run whose out-of-sample evidence passed.
    """
    return str(Path(data_dir) / "research_v3" / "candidates")


def promote_candidate(data_dir, candidate_path, backend):
    """Copy a passing candidate into the live folder so the runtime will load it.

    Returns the destination path. Only the named backend's newest artifact is copied:
    dropping the whole run in would let a second backend silently replace the first.
    """
    import shutil
    source = Path(candidate_path)
    if not source.is_dir():
        raise ValueError("candidate_missing:" + str(candidate_path))
    destination = Path(live_candidates_root(data_dir)) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return str(destination)


class TrainingJob:
    """Mutable state for one run; serialised to the store after every stage."""

    def __init__(self, tier, settings=None, **options):
        self.settings = settings or Settings()
        self.tier = tier
        self.options = {**DEFAULTS, **options}
        self.run_id = uuid.uuid4().hex[:12]
        self.status = "running"
        self.stage = "universe"
        self.started_at = _now()
        self.finished_at = None
        self.logs = []
        self.symbols = []
        self.universe_counts = {}
        self.dataset = {}
        self.walk_forward = None
        self.point_in_time = None
        self.portfolio_oos = None
        self.registry = None
        self.candidates = []
        self.verdict = []
        self.promoted = False
        self.promoted_to = None
        self.error = None

    def log(self, message):
        self.logs.append("%s  %s" % (time.strftime("%H:%M:%S"), message))
        del self.logs[:-MAX_LOGS]

    def snapshot(self):
        return {"run_id": self.run_id, "tier": self.tier, "status": self.status,
                "stage": self.stage, "options": self.options, "started_at": self.started_at,
                "finished_at": self.finished_at, "symbols": self.symbols,
                "universe_counts": self.universe_counts, "dataset": self.dataset,
                "walk_forward": self.walk_forward, "point_in_time": self.point_in_time,
                "portfolio_oos": self.portfolio_oos, "registry": self.registry,
                "candidates": self.candidates,
                "verdict": self.verdict, "promoted": self.promoted,
                "promoted_to": self.promoted_to, "error": self.error,
                "logs": self.logs[-60:]}


def verdict_for(walk_forward, options):
    """The pass/fail reading of a walk-forward result, stated in plain terms."""
    if not walk_forward:
        return ["no walk-forward result"]
    aggregate = walk_forward.get("aggregate") or {}
    boot = walk_forward.get("bootstrap") or {}
    lines = []
    edge = aggregate.get("net_edge_bps", 0.0)
    probability = boot.get("prob_positive", 0.0)
    active = aggregate.get("active_samples", 0)
    if active < options["min_active_samples"]:
        lines.append("only %d out-of-sample trades: too few to score" % active)
    if probability < options["min_prob_positive"]:
        lines.append("P(edge>0)=%.2f below the %.2f bar: no edge established"
                     % (probability, options["min_prob_positive"]))
    if boot.get("low_bps", 0) <= 0 <= boot.get("high_bps", 0):
        lines.append("confidence interval straddles zero")
    if edge <= 0:
        lines.append("out-of-sample edge is negative (%.2f bp): this would lose money" % edge)
    positive = aggregate.get("folds_positive_edge", 0)
    negative = aggregate.get("folds_negative_edge", 0)
    if negative > positive:
        lines.append("more losing folds than winning ones (%d vs %d): period-dependent"
                     % (negative, positive))
    if not lines:
        lines.append("passed every check: edge %.2f bp, P(edge>0)=%.2f" % (edge, probability))
    return lines


def passes(walk_forward, options):
    if not walk_forward:
        return False
    aggregate = walk_forward.get("aggregate") or {}
    boot = walk_forward.get("bootstrap") or {}
    return (aggregate.get("active_samples", 0) >= options["min_active_samples"]
            and boot.get("prob_positive", 0.0) >= options["min_prob_positive"]
            and aggregate.get("net_edge_bps", 0.0) > 0)


def _release_memory():
    """Return freed training memory to the operating system between stages.

    A finished stage leaves a million-row list to the collector, and CPython keeps the
    arenas rather than handing them back. On a sixteen-gigabyte host running a fifteen-
    symbol fit that difference decided whether the service stayed up."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("msvcrt")._heapmin()
    except Exception:
        pass


class TrainingRunner:
    """Owns the single in-flight run and the history of finished ones."""

    def __init__(self, settings, store, market_rows_fn=None):
        self.settings = settings
        self.store = store
        self.market_rows_fn = market_rows_fn
        self.current = None
        self.task = None
        self.history = list(store.get_runtime(RUN_KEY, []) or [])

    # ---------------------------------------------------------------- persistence
    def _persist(self):
        try:
            self.store.set_runtime(RUN_KEY, self.history[-20:])
        except Exception:
            pass

    def state(self):
        return {"current": self.current.snapshot() if self.current else None,
                "history": list(reversed(self.history[-20:])),
                "running": bool(self.current and self.current.status == "running")}

    # ---------------------------------------------------------------- run control
    def start(self, tier, **options):
        if tier not in (TIER_MAINSTREAM, TIER_SPECULATIVE):
            raise ValueError("unknown_tier:" + str(tier))
        if self.current and self.current.status == "running":
            raise ValueError("a_run_is_already_in_flight")
        job = TrainingJob(tier, self.settings, **options)
        self.current = job
        job.log("run %s started for tier=%s" % (job.run_id, tier))
        return job

    async def execute(self, job, market_rows):
        try:
            await self._run(job, market_rows)
            job.status = "done"
        except Exception as exc:  # a failed run must not take the server down
            job.status = "failed"
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.log("FAILED " + job.error)
        finally:
            job.finished_at = _now()
            job.stage = "done" if job.status == "done" else job.stage
            self.history.append(job.snapshot())
            self._persist()

    @staticmethod
    def _attach_portfolio_evidence(candidate_path, evidence, walk_forward=None, verdict=None):
        """Write the evidence into the candidate manifest the gate and the reader look at.

        The gate reads metrics.portfolio_oos; the training run computed it after the
        manifest was written, so without this the two never meet and the gate reports a
        missing file for a model that has the evidence.

        The walk-forward verdict is attached for the same reason and a different reader. It
        is the answer to "why was this not promoted", and it lived only in the training
        job's in-memory record -- so the artifact that a later shift or a budget tool
        selects between carried no statement of why it should not be traded.
        """
        path = Path(candidate_path) / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metrics = dict(manifest.get("metrics") or {})
        if evidence:
            metrics["portfolio_oos"] = evidence
        if walk_forward:
            metrics["walk_forward"] = walk_forward
        if verdict:
            manifest["verdict"] = list(verdict)
        if len(metrics) == len(manifest.get("metrics") or {}) and "verdict" not in manifest:
            return None
        manifest["metrics"] = metrics
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        return str(path)

    def _register_note(self, job, best):
        """Why a passing tabular candidate does not also become the registry's model.

        The two are different artifact formats. The registry holds the gradient-boosted
        stumps model that `ProductionMember` loads and calls `predict_advanced` on, which
        requires `base_score`, `stumps` and `normalization`; a LightGBM or CatBoost
        booster has none of those. Registering the tabular candidate's manifest as if it
        were a model put metadata where the loader expected weights, so a promotion through
        that path produced something that raised `legacy_model_requires_retraining` on every
        call. This reports the split instead of performing it.
        """
        manifest_path = Path(best["path"]) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        job.log("registry: the %s candidate is a %s artifact and stays in the candidates "
                "folder; data/models holds the gradient-boosted stumps format, produced by "
                "`python -m app.advanced_model <dataset> --register`"
                % (best.get("backend"), manifest.get("backend") or "tabular"))
        return {"status": "not_applicable", "backend": best.get("backend"),
                "reason": "different_artifact_format"}

    def _portfolio_oos(self, rows, oos, options):
        """Replay the out-of-sample predictions through one shared account.

        `rows` are dataset rows -- features and a label. The replay needs candles, because it
        runs the live PaperAccount over a price series, and a feature row has no open, high,
        low or close on it. Passing the dataset straight through made `validate_ohlcv` reject
        every row as `missing_fields`, so the replay returned zero trades for every model and
        the promotion gate read that as "the account did not make money". The candles are
        therefore read back from the store over the window the predictions cover: the dataset
        keeps only the features, and the prices live in the table they came from.
        """
        from .portfolio_oos import (funding_for, funding_windows, gate_verdict,
                                    portfolio_evidence)

        rows = self._candles_for(oos, options)
        # The symbol list is derived from the rows rather than from the row list itself:
        # `sorted(rows)` compares dicts, which raises rather than sorting.
        symbols = sorted({str(r.get("symbol")) for r in rows if r.get("symbol")})
        evidence = portfolio_evidence(rows, oos, cost_bps=options["cost_bps"],
                                      interval=options["interval"], settings=self.settings,
                                      funding=funding_for(self.store, symbols),
                                      windows=funding_windows(self.store, symbols))
        evidence["gate"] = gate_verdict(evidence)
        return evidence

    def _candles_for(self, oos, options):
        """The stored candles covering every symbol and timestamp the predictions touch.

        Raw rows, not the symbol-grouped mapping: `portfolio_evidence` groups them itself,
        and handing it an already-grouped dict made it call `.get` on a symbol name.
        """
        stamps = [int(r["timestamp"]) for r in (oos or ()) if r.get("timestamp") is not None]
        symbols = sorted({str(r.get("symbol")) for r in (oos or ()) if r.get("symbol")})
        if not symbols or not stamps:
            return []
        interval = options["interval"]
        # One bar of slack on each side: the strategy is asked about the bar after the
        # decision, and a series that stops exactly at the last prediction would truncate the
        # final trades rather than score them.
        span = interval_to_ms(interval)
        return self.store.candles_range(symbols, min(stamps) - span, max(stamps) + 2 * span,
                                        interval=interval)

    def _point_in_time(self, symbols, options):
        """The universe as it stood at the start of the training window.

        Read from the candles already stored rather than from the live snapshot. The window
        start is derived from the requested history length, so it moves with the request
        instead of being a fixed date that silently goes stale.
        """
        from ..features.universe_history import extent_from_windows, point_in_time, series_from_windows, universe_changes

        interval_ms = interval_to_ms(options["interval"])
        now = _now()
        start = now - int(options["days"]) * 86_400_000
        # One extra day before the window start. The reconstruction needs the trailing 24h
        # ending at the start, and fetching from the start itself leaves that window empty
        # -- which marks every symbol as not tradeable and empties the whole universe.
        fetch_start = start - 86_400_000
        # Two bounded window reads, not the whole range. A year of 5-minute bars across a
        # tier is 1,391,212 rows -- 7.7 seconds and 1.6 GB to materialise -- and the range
        # reader carried a default LIMIT of 500,000 with ORDER BY open_time ASC, so the
        # reconstruction was reading the *oldest* 36% of the window and measuring the
        # trailing 24 hours 234 days before the data ended.
        probe = self.store.candle_windows(symbols, (start,), interval=options["interval"])
        if not probe:
            return {"status": "unavailable", "reason": "no_stored_candles"}
        extent = extent_from_windows(probe)
        horizon = int(extent["horizon"])
        later = self.store.candle_windows(symbols, (horizon,), interval=options["interval"])
        series = series_from_windows([probe, later], (start, horizon))
        # The moment the data reaches, not the wall clock: the training tier is refreshed
        # by the history stage of a run, so between runs its candles fall behind while the
        # live symbols stay current. Evaluated at "now", every tier symbol reads as having
        # stopped trading, and that is a fact about our ingest being reported as a fact
        # about the market.
        beginning = point_in_time(series, start, fetch_start_ms=fetch_start)
        ending = point_in_time(series, horizon, fetch_start_ms=fetch_start)
        changes = universe_changes(beginning, ending)
        listed_late = sorted(set(symbols) - {item["symbol"] for item in beginning["graded"]})
        notes = []
        # A symbol that is in the universe at the start and absent at the horizon has two
        # explanations the verdict alone cannot separate, and they point opposite ways: it
        # stopped trading (survivorship -- its losses belong in the result), or our own
        # feed for it is behind the rest of the store (a gap in the data, not the market).
        # The ticker-by-ticker lag is what tells them apart.
        stale_or_delisted = []
        for item in beginning["graded"]:
            member = extent["per_symbol"].get(item["symbol"]) or {}
            last = member.get("last")
            if last is None:
                continue
            lag = max(0, horizon - int(last))
            if lag > 86_400_000:
                stale_or_delisted.append({"symbol": item["symbol"], "feed_lag_ms": lag,
                                          "feed_lag_days": round(lag / 86_400_000, 2)})
        if not beginning["graded"] and extent["first"] is not None:
            # Bars exist but nothing qualified, and the reasons say which way: a symbol
            # listed later is the market answering, a symbol with no bars in the window is
            # usually the ingest. Reporting the count alone cannot tell them apart.
            notes.append("no symbol qualified at the window start (%s); check the trailing "
                         "window was fetched (fetch_start=%d)"
                         % (beginning["untraded_reasons"] or {}, fetch_start))
        if listed_late:
            notes.append("not tradeable at window start: " + ", ".join(listed_late))
        if changes["removed"]:
            notes.append("left the universe during the window: " + ", ".join(changes["removed"]))
        bounded = sorted(item["symbol"] for item in beginning["graded"]
                         if item.get("listed_age_is_lower_bound"))
        # The outcome, as one field. Every symbol measuring one day old at the window start
        # is not a universe with nothing in it: the age gate is reading our fetch boundary
        # rather than the listing, which means the request reaches further back than the
        # store does. That distinction decides whether to shorten the window or to ingest
        # more history, and it was previously an exclusion count that looks the same either
        # way. Note it is the *measured age* that separates the two, not the lower-bound
        # flag: a symbol is flagged whenever the store does not reach its listing, which is
        # true of a two-year store as well, and its age is still two years.
        # The measured age is the thing that separates the two cases, not the lower-bound
        # flag: a symbol is flagged whenever the store does not reach its listing, which is
        # true of a two-year store as well, and its age is still two years.
        graded_list = beginning["graded"]
        boundary_only = bool(graded_list) and all(
            fetch_start - 86_400_000 <= int(item.get("listed_at") or 0) <= fetch_start + 86_400_000
            for item in graded_list)
        unmeasured = beginning.get("unmeasured_criteria") or {}
        all_unmeasured = bool(unmeasured) and sum(unmeasured.values()) >= len(graded_list)
        if not beginning["counts"]:
            diagnosis = "unmeasured"
        elif boundary_only:
            diagnosis = "request_reaches_before_the_store"
        elif beginning["tradeable"]:
            diagnosis = "universe_measured"
        elif all_unmeasured:
            # Every candidate excluded by a criterion that nothing measured. The universe
            # is not thin, it is unread -- and the response is to re-ingest, not to trade a
            # smaller set.
            diagnosis = "criteria_unmeasured"
        else:
            diagnosis = "nothing_tradeable_at_start"
        if unmeasured:
            notes.append("tier criteria decided by an empty column: %s; these symbols were "
                         "not excluded on what they are, but on what was never recorded"
                         % ", ".join("%s x%d" % (name, count)
                                     for name, count in sorted(unmeasured.items())))
        # A refusal that says only "no" is not actionable. The window the store can support
        # is a number: the data reaches back this far, the tier needs this much age at the
        # window start, and the difference is the longest request that can be established.
        min_age = int((RULES.get(self.current.tier if self.current else "", {}) or {})
                      .get("min_age_days") or 0)
        reach_days = (horizon - int(extent["first"] or horizon)) / 86_400_000.0
        max_window_days = int(reach_days - 1 - min_age)
        if diagnosis == "request_reaches_before_the_store" and max_window_days > 0:
            notes.append("the store reaches back %.0f days and the %s tier needs %d days of "
                         "age at the window start, so the longest window it can establish is "
                         "%d days; ingest older candles or shorten the window"
                         % (reach_days, self.current.tier, min_age, max_window_days))
        if diagnosis != "universe_measured":
            notes.append("point-in-time universe not established (%s): the dataset will be "
                         "built from the tier chosen from today's snapshot" % diagnosis)
        if bounded:
            # Age is measured from the earliest candle in hand. When that is the fetch
            # boundary rather than the listing, the gate reads a symbol as one day old and
            # excludes it -- conservative, but it silently empties a tier, so it is said out
            # loud rather than left to be inferred from an exclusion count.
            notes.append("listing age is a lower bound for: " + ", ".join(bounded)
                         + " (extend the fetch window or supply onboardDate to measure it)")
        stale_symbols = {item["symbol"] for item in stale_or_delisted}
        # A removal reported while the symbol's own feed is hours behind the rest of the
        # store is not evidence of anything about the market. Keeping the two lists apart
        # is the difference between "the universe shrank" and "the ingest stopped".
        removed_stale = [s for s in changes["removed"] if s in stale_symbols]
        removed_current = [s for s in changes["removed"] if s not in stale_symbols]
        if stale_or_delisted:
            notes.append("%d symbol(s) in the universe at the window start have no recent "
                         "bars in the store; a removal among them is an ingest gap, not a "
                         "delisting" % len(stale_or_delisted))
        return {"status": "ok", "as_of": start, "fetch_start": fetch_start,
                "horizon": horizon, "feed_lag_ms": max(0, now - horizon),
                "extent": {"first": extent["first"], "horizon": extent["horizon"],
                           "symbols": extent["symbols"]},
                "at_start": beginning["counts"],
                "at_end": ending["counts"], "graded_at_start": beginning["graded"],
                "measurable": bool(beginning["measurable"]),
                "max_window_days": max_window_days,
                "tradeable": bool(beginning["tradeable"]),
                "diagnosis": diagnosis,
                "untraded_reasons_at_start": beginning["untraded_reasons"],
                "unmeasured_criteria_at_start": unmeasured,
                "not_tradeable_at_start": listed_late,
                "stale_or_delisted": stale_or_delisted,
                # In the universe at the start, gone at the horizon, and its own feed is
                # current: this is a removal the data supports. The other list is suspect.
                "removed_with_current_feed": removed_current,
                "removed_with_stale_feed": removed_stale,
                "changes": changes, "notes": notes, "interval_ms": interval_ms}

    def _restrict(self, job, rows):
        """Cut the dataset down to what the reconstruction says was tradeable.

        The reconstruction above reads the universe at the start of the window; this is
        where that verdict is used. Until now it was computed, stored on the job, printed in
        the log, and then ignored -- the dataset was built from the tier list chosen from
        today's snapshot, so every symbol that listed inside the window contributed rows for
        the part of the window before it existed, and the window itself ran back past the
        requested start. Both inflate the result and neither is visible in the output.

        It refuses rather than empties. A reconstruction that graded every candidate into
        the age or liquidity gate has not established that nothing was tradeable; it has
        failed to measure, and cutting the dataset to zero rows on the strength of it would
        turn a request that is too long for the store into a silent empty dataset.
        """
        universe = job.point_in_time or {}
        start = universe.get("as_of")
        graded = universe.get("graded_at_start") or []
        # Four refusals, named apart, because they call for four different responses: build
        # a dataset, fix the fetch window, fix the ingest, or accept that the request reaches
        # further back than the store does. One "could not restrict" covers none of them.
        if not rows:
            return rows, {"status": "skipped", "reason": "no_rows"}
        if not start:
            return rows, {"status": "skipped", "reason": "no_start_universe"}
        if not universe.get("measurable"):
            reason = "start_universe_unmeasured"
            job.log("point-in-time restriction skipped: %s (%s)"
                    % (reason, universe.get("untraded_reasons_at_start")))
            return rows, {"status": "skipped", "reason": reason}
        # Graded is not the same as tradeable. Every candidate that read as one day old at
        # the window start is in the graded list, in the excluded tier -- so taking the list
        # as the allow-set made an all-excluded universe restrict nothing and report success,
        # which is the exact failure this method exists to prevent.
        allowed = {item["symbol"] for item in graded
                   if item.get("tier") != TIER_EXCLUDED}
        if not allowed:
            # Carried through from the reconstruction rather than flattened to "nothing was
            # tradeable": a universe emptied by an unread column calls for re-ingesting, and
            # one emptied by the age gate calls for a shorter window. Same outcome, opposite
            # next step.
            reason = ("criteria_unmeasured"
                      if universe.get("diagnosis") == "criteria_unmeasured"
                      else "nothing_tradeable_at_start")
            job.log("point-in-time restriction skipped: %s (%s)"
                    % (reason, universe.get("at_start")))
            return rows, {"status": "skipped", "reason": reason,
                          "at_start": universe.get("at_start"),
                          "unmeasured_criteria": universe.get("unmeasured_criteria_at_start")}
        before = len(rows)
        kept = [row for row in rows
                if row.get("symbol") in allowed and int(row.get("timestamp") or 0) >= start]
        dropped_symbols = sorted({row.get("symbol") for row in rows} - allowed)
        # The two causes are counted apart on purpose: a symbol that was not listed yet and
        # a window that reaches back further than the request are different mistakes, and a
        # single "rows dropped" number would hide whichever one is doing the work.
        by_symbol = sum(1 for row in rows if row.get("symbol") not in allowed)
        return kept, {"status": "ok", "as_of": start, "allowed_symbols": sorted(allowed),
                      "rows_before": before, "rows_after": len(kept),
                      "rows_dropped_symbol": by_symbol,
                      "rows_dropped_before_start": before - len(kept) - by_symbol,
                      "dropped_symbols": dropped_symbols}

    async def _run(self, job, market_rows):
        loop = asyncio.get_running_loop()
        options = job.options

        job.stage = "universe"
        graded = classify(market_rows or [], now_ms=_now())
        job.universe_counts = summarise(graded)
        symbols = symbols_for(job.tier, market_rows or [], limit=options["max_symbols"],
                              now_ms=_now())
        if not symbols:
            raise ValueError("no_symbols_in_tier:" + job.tier)
        job.symbols = symbols
        job.log("tier %s -> %d symbols: %s" % (job.tier, len(symbols), ", ".join(symbols)))

        job.stage = "history"
        from ..market.ingest import ingest
        results = await ingest(symbols, options["interval"], options["days"],
                               self.settings.base_url, self.settings.data_dir,
                               concurrency=6, on_progress=job.log)
        written = sum(r["written"] for r in results)
        failures = [r["symbol"] for r in results if r["errors"]]
        job.log("history refreshed: %d new bars, %d failures" % (written, len(failures)))
        if failures:
            job.log("failed symbols: " + ", ".join(failures))

        # The tier was chosen from today's snapshot, and the history that follows reaches
        # back further than some of these contracts have existed. Rebuilding the universe
        # at the start of the window is what makes that visible: a symbol that was not yet
        # listed, or had already stopped trading, is not part of the period being evaluated.
        job.stage = "point_in_time"
        try:
            universe = await loop.run_in_executor(
                None, lambda: self._point_in_time(symbols, options))
            job.point_in_time = universe
            for line in universe.get("notes") or ():
                job.log(line)
        except Exception as exc:
            job.point_in_time = {"status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)}
            job.log("point-in-time universe unavailable: " + job.point_in_time["error"])

        job.stage = "dataset"
        from .train import build_dataset
        destination = dataset_path(self.settings.data_dir, job.tier)
        built = await loop.run_in_executor(None, lambda: build_dataset(
            self.settings.data_dir, tuple(symbols), options["interval"],
            horizon=options["horizon"], window=self.settings.lookback,
            output=destination, strict=False,
            max_basis_age_ms=self.settings.max_basis_age_ms))
        job.dataset = {"path": built["output"], "rows": built["rows"],
                       "window": built["window"], "horizon": built["horizon"],
                       "symbols": [{"symbol": e["symbol"], "bars": e["bars"],
                                    "rows": e["rows"]} for e in built["symbols"]]}
        job.log("dataset built: %d rows" % built["rows"])
        if built["rows"] < 1000:
            raise ValueError("dataset_too_small:%d" % built["rows"])

        job.stage = "walk_forward"
        from .fit_model import load
        from .walkforward import walk_forward_tabular
        rows = await loop.run_in_executor(None, load, destination)
        rows, restriction = self._restrict(job, rows)
        job.dataset["restriction"] = restriction
        if restriction["status"] == "ok" and restriction["rows_after"] != restriction["rows_before"]:
            job.log("point-in-time restriction: %d of %d rows kept (%d symbols dropped: %s)"
                    % (restriction["rows_after"], restriction["rows_before"],
                       len(restriction["dropped_symbols"]),
                       ", ".join(restriction["dropped_symbols"]) or "none"))
        if not rows:
            raise ValueError("no_rows_after_point_in_time_restriction")
        result = await loop.run_in_executor(None, lambda: walk_forward_tabular(
            rows, "lightgbm", options["folds"], options["cost_bps"],
            options["rounds"], options["horizon"], return_oos=True))
        job.walk_forward = {k: result[k] for k in
                            ("folds", "backend", "cost_bps", "purge_bars", "aggregate",
                             "bootstrap", "per_fold")}
        aggregate, boot = result["aggregate"], result["bootstrap"]
        job.log("walk-forward: %d folds, %d trades, edge %.2f bp, P(edge>0)=%.2f"
                % (result["folds"], aggregate["active_samples"],
                   aggregate["net_edge_bps"], boot["prob_positive"]))

        # The per-prediction walk-forward says whether the signal beats the cost. It cannot
        # say whether an account trading it makes money, because it never runs an account.
        # The promotion gate refuses without a portfolio result, and nothing produced one, so
        # no candidate could ever be promoted. This is that producer.
        job.stage = "portfolio_oos"
        # Imported here, not only inside _portfolio_oos. The gate verdict was read in this
        # scope while the name was bound only in that method's local scope, so every run
        # raised NameError *after* the evidence had been computed and *before* it was kept:
        # the except below replaced a real portfolio result with a failure record. Nothing
        # downstream could tell the difference between "the account lost money" and "the
        # code crashed", and no candidate ever carried portfolio evidence.
        from .portfolio_oos import gate_verdict
        # The out-of-sample predictions are roughly a million dicts, and nothing after this
        # stage reads them: the walk-forward above already consumed them, and the candidate
        # fits below read the dataset rows instead. Keeping them alive through training put
        # the service at 93% of physical memory with the dataset still resident and the fit
        # about to allocate its matrices on top, which is how the operating system came to
        # kill processes during a run. Released as soon as the replay that needs them ends.
        oos = result.pop("oos", None) or []
        try:
            evidence = await loop.run_in_executor(None, lambda: self._portfolio_oos(
                rows, oos, options))
            job.portfolio_oos = evidence
            verdict = gate_verdict(evidence)
            job.log("portfolio: %d trades, net %.2f%%, max drawdown %.2f%% -> %s"
                    % (evidence.get("trades") or 0,
                       (evidence.get("net_return") or 0.0) * 100,
                       (evidence.get("max_drawdown") or 0.0) * 100,
                       "passes" if verdict["passes"] else ", ".join(verdict["failures"])))
        except Exception as exc:
            job.portfolio_oos = {"status": "failed",
                                 "error": "%s: %s" % (type(exc).__name__, exc)}
            job.log("portfolio evidence unavailable: " + job.portfolio_oos["error"])
        finally:
            # The candle rows the replay loaded are the largest single allocation of the
            # run, and the fit below needs that room more than the replay does.
            oos = None
            _release_memory()

        job.verdict = verdict_for(job.walk_forward, options)
        job.promoted = passes(job.walk_forward, options)
        for line in job.verdict:
            job.log("verdict: " + line)

        job.stage = "train"
        from .tabular_model import train_tabular
        output_root = candidates_root(self.settings.data_dir, job.tier)
        for backend in ("lightgbm", "catboost"):
            try:
                trained = await loop.run_in_executor(None, lambda b=backend: train_tabular(
                    rows, b, output_root, options["rounds"], options["cost_bps"]))
            except Exception as exc:
                job.log("%s training failed: %s" % (backend, exc))
                continue
            job.candidates.append({"backend": backend, "path": trained["path"],
                                   "feature_version": trained["feature_version"],
                                   "in_sample_test": trained["metrics"]["test"]})
            job.log("%s candidate written to %s" % (backend, trained["path"]))
            _release_memory()
        # Evidence is recorded on every candidate, not only the promoted one. Two reasons,
        # both about being able to answer questions later. The registry gate reads
        # metrics.portfolio_oos, and the walk-forward verdict is the only written answer to
        # "why was this rejected" -- keeping them only for runs that already passed meant a
        # rejected artifact carried no statement of why, which is exactly the case an
        # operator needs to read. Recording is cheap and writes no weights.
        for candidate in job.candidates:
            try:
                self._attach_portfolio_evidence(candidate["path"], job.portfolio_oos,
                                                job.walk_forward, job.verdict)
            except Exception as exc:
                job.log("could not record evidence on %s: %s" % (candidate["backend"], exc))
        if job.promoted and job.candidates:
            # A run that passed is the only thing allowed to reach the live folder.
            best = job.candidates[0]
            try:
                live_path = promote_candidate(self.settings.data_dir, best["path"],
                                              best["backend"])
                job.promoted_to = live_path
                job.log("PROMOTED %s to live candidates: %s" % (best["backend"], live_path))
                job.log("restart the service to load it")
            except Exception as exc:
                job.promoted_to = None
                job.log("promotion failed: %s" % exc)
            job.registry = self._register_note(job, best)
        else:
            job.log("not promoted: the live folder is left unchanged")
        job.log("run finished; promoted=%s" % job.promoted)
