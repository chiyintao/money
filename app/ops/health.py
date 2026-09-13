"""What "/health" should mean.

The endpoint returned a fixed status of "ok" plus five counters. Every one of them was a
count, and a count cannot be wrong: zero open orders and zero live events look the same,
and a service whose feed had died, whose weights were expired, and whose reconciliation
had found the ledger inconsistent still answered ok. A monitor polling it learned nothing,
and a reader of the dashboard learned that the process had not crashed -- which the HTTP
response already said.

This builds a report from state that is already in memory, on purpose. A liveness probe is
the most frequently hit endpoint and the reason the old handler avoided the dashboard
payload is correct: assembling the full snapshot is the most expensive thing the server
does. Nothing here computes anything new, calls the venue, or touches the database.

Each check answers a different question, and the status is the worst answer among them.
"degraded" means the service is running but one of its inputs or guarantees is not what it
should be. "down" means entries cannot be trusted at all -- no live prices, or the store
failing -- and is the condition a monitor should page on.
"""
import time

OK = "ok"
DEGRADED = "degraded"
DOWN = "down"
SKIPPED = "skipped"

# How many times the market-data freshness window may be exceeded before the feed is
# treated as dead rather than late. The guard already refuses entries at one window; the
# larger bound is what distinguishes "this symbol is quiet" from "the socket is gone".
DEAD_FEED_MULTIPLE = 5.0


def _now(now_ms=None):
    return int(now_ms if now_ms is not None else time.time() * 1000)


def _check(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def _age(now, stamp):
    try:
        return max(0, now - int(stamp))
    except (TypeError, ValueError):
        return None


def market_check(runtime, now, guard_ms):
    """Is there a live price at all, and how old is the newest one?"""
    observed = getattr(getattr(runtime, "guard", None), "last_event_ms", None) or {}
    if not observed:
        return _check("market_data", DOWN, "no_live_events")
    newest = max(observed.values())
    age = _age(now, newest)
    if age is None:
        return _check("market_data", DEGRADED, "unreadable_event_time")
    symbols = len(observed)
    if guard_ms > 0 and age > guard_ms * DEAD_FEED_MULTIPLE:
        return _check("market_data", DOWN, "feed_silent", age_ms=age, symbols=symbols)
    if guard_ms > 0 and age > guard_ms:
        return _check("market_data", DEGRADED, "feed_stale", age_ms=age, symbols=symbols)
    return _check("market_data", OK, "live", age_ms=age, symbols=symbols)


def feed_check(runtime, now):
    """Has the socket produced anything since startup?

    A service that starts, subscribes and receives nothing is running and useless, and
    every counter the old payload reported was zero in that state.
    """
    feeds = getattr(runtime, "feeds", None)
    events = int(getattr(feeds, "ws_events", 0) or 0)
    if events <= 0:
        return _check("feed", DEGRADED, "no_events_since_start")
    return _check("feed", OK, "streaming", events=events,
                  last_event=getattr(feeds, "last_event", None))


def _model_runtime(runtime):
    """The object that actually holds the loaded weights and the promotion verdict.

    Both live on the model runtime. This check read them off the outer Runtime, which has
    no such attributes, so every deployment reported "trading_unpromoted_weights" with
    mode=None -- including one running a properly promoted model, and one that had loaded
    nothing at all. A status read from the wrong object, defaulting to a plausible value.
    """
    models = getattr(runtime, "models", None)
    # Accept either shape: the live service holds the model runtime on Runtime.models, and
    # the tests that construct this check directly pass the model runtime itself.
    if models is not None and hasattr(models, "promotion"):
        return models
    return runtime


def model_check(runtime, now):
    """Are the loaded weights allowed to trade, and are they current?"""
    holder = _model_runtime(runtime)
    members = getattr(holder, "models", None)
    promotion = getattr(holder, "promotion", None) or {}
    if not members:
        # "No model runtime" and "a model runtime with nothing loaded" are different
        # states and the second one is not a skip: it is a service that has been told to
        # trade and cannot. A promoted model that expired, or that lost its calibrator,
        # lands here, and SKIPPED would have read as "not configured".
        if promotion.get("status") in ("fallback", "error"):
            return _check("models", DEGRADED,
                          "production_unusable:%s" % (promotion.get("reason") or "unknown"),
                          version=promotion.get("version"), mode=getattr(holder, "mode", None))
        return _check("models", SKIPPED, "no_model_runtime")
    decisions = getattr(runtime, "decisions", None)
    problem = None
    if decisions is not None and hasattr(decisions, "provenance_problem"):
        try:
            problem = decisions.provenance_problem()
        except Exception as exc:
            problem = "provenance_check_failed:%s" % type(exc).__name__
    if problem:
        # Unpromoted or expired weights are not a crash, but every number the service
        # produces is suspect, which is exactly what "degraded" is for.
        return _check("models", DEGRADED, problem)
    status = promotion.get("status")
    if status != "active":
        # Not a failure: the service is configured to accept candidates, and refusing to
        # trade at all is worse than trading unvetted weights while the model matures. But
        # "tradable" on its own reads as a clean bill of health, and this is the one fact
        # that decides whether any number the service reports passed review.
        return _check("models", OK, "trading_unpromoted_weights", mode=status,
                      version=promotion.get("version"),
                      reason=promotion.get("reason"))
    return _check("models", OK, "tradable", mode=status, version=promotion.get("version"))


def features_check(runtime, now):
    """Can the service fill every feature the loaded weights were fit on?

    This is the question the whole feature layer exists to answer, and nothing asked it.
    The service trained a model on ten features and served it four constants; the model
    panel, the out-of-range gate and every aggregate looked plausible from either side,
    because a placeholder zero sits comfortably inside the training bounds. The source
    reports which names it had to fill, so the answer is now a fact rather than an
    assumption -- and it has to be a health check, because "the weights loaded" and "the
    weights are being fed what they were fit on" are different statements.
    """
    decisions = getattr(runtime, "decisions", None)
    # The loaded weights may declare input columns this code cannot compute. That is not a
    # partial outage -- the model reads exactly those columns and every one of them would be
    # absent -- and it used to leave required_features() empty, which switches the
    # degeneracy check off and makes this very check report a healthy feature layer.
    unusable = list(getattr(decisions, "unusable_features", None) or ())
    if unusable:
        return _check("features", DEGRADED, "declared_features_unproducible",
                      unproducible=unusable)
    source = getattr(decisions, "feature_source", None) if decisions is not None else None
    if source is None or not hasattr(source, "health"):
        return _check("features", SKIPPED, "not_configured")
    try:
        health = dict(source.health())
    except Exception as exc:
        return _check("features", DEGRADED, "feature_source_unreadable:%s" % type(exc).__name__)
    required = []
    if decisions is not None and hasattr(decisions, "required_features"):
        try:
            required = list(decisions.required_features())
        except Exception:
            required = []
    detail = {"required": len(required), "contract": health.get("reported_features"),
              "rows": health.get("rows"), "degraded_rows": health.get("degraded_rows"),
              "families": health.get("families"),
              "missing_features": health.get("degraded_features")}
    if not int(health.get("rows") or 0):
        # No signal has been built yet. Reporting "ok" here would be a check that passes
        # before it has looked at anything.
        detail["required_features"] = required
        return _check("features", SKIPPED, "no_rows_yet", **detail)
    if int(health.get("degraded_rows") or 0) >= int(health.get("rows") or 1):
        return _check("features", DEGRADED, "every_row_degraded", **detail)
    if int(health.get("degraded_rows") or 0):
        return _check("features", DEGRADED, "partially_degraded", **detail)
    if not health.get("extra_rows"):
        # Every modelled column was filled, but from the candle and funding history alone:
        # the collected families are declared and empty, so the model is being served a
        # narrower input than the contract names.
        return _check("features", DEGRADED, "collected_families_empty", **detail)
    return _check("features", OK, "all_inputs_measured", **detail)


def drift_check(runtime, now):
    monitor = getattr(runtime, "drift", None)
    if monitor is None:
        return _check("drift", SKIPPED, "not_configured")
    try:
        health = monitor.health() or {}
    except Exception as exc:
        return _check("drift", DEGRADED, "drift_check_failed:%s" % type(exc).__name__)
    drifted = health.get("drifted") or []
    blocked = False
    try:
        blocked = bool(monitor.blocks_entries())
    except Exception:
        blocked = False
    # Whether anything would act on the verdict. A drift finding that neither stops entries
    # nor asks for a retrain is a row in the event table, and reporting it alone reads as if
    # the system has responded to it.
    retrainer = getattr(runtime, "retrainer", None)
    wired = bool(getattr(retrainer, "drift_fn", None)) if retrainer is not None else False
    detail = {"drifted": drifted, "retrain_on_drift": wired}
    if blocked:
        return _check("drift", DEGRADED, "drift_blocking_entries", **detail)
    if drifted:
        if not wired and retrainer is not None:
            return _check("drift", DEGRADED, "feature_drift_unactionable", **detail)
        return _check("drift", DEGRADED, "feature_drift", **detail)
    return _check("drift", OK, "stable", compared=health.get("compared"), **detail)


def ood_check(runtime, now):
    """Whether the input-range gate can refuse anything.

    A bound equal to the feature's a-priori clip range is not a bound: every value the
    pipeline can produce is inside it, so the gate is unreachable for that feature and
    nothing downstream is told. Reported here rather than repaired, because the fix is to
    rebuild the dataset profile and a gate that quietly fixed itself would hide the fact
    that the stored one is stale. Run scripts/profile_dataset.py.
    """
    models = getattr(runtime, "models", None)
    space = getattr(models, "feature_space", None) if models else None
    if not space:
        return _check("ood_gate", SKIPPED, "no_feature_space")
    if space.get("status") != "ok":
        return _check("ood_gate", DEGRADED, str(space.get("reason") or space.get("status")))
    quality = space.get("bounds_quality") or {}
    degenerate = quality.get("degenerate") or []
    if degenerate:
        return _check("ood_gate", DEGRADED, "unreachable_for:" + ",".join(degenerate),
                      degenerate=degenerate)
    # A feature the model reads and the profile has no bounds for is not checked at all,
    # and ood_check used to return ok with a count that only covered the ones it looked at.
    # The profile is keyed on the dataset the weights were trained on, so a dataset older
    # than the feature contract leaves the newer columns unguarded -- silently, because
    # "checked: 10" reads like a complete answer.
    unbounded = set(quality.get("unbounded") or ())
    if unbounded:
        needed = set()
        decisions = getattr(runtime, "decisions", None)
        if decisions is not None and hasattr(decisions, "required_features"):
            try:
                needed.update(decisions.required_features())
            except Exception:
                needed = set()
        if not needed:
            # runtime.models is the model runtime, which holds the loaded members; falling
            # back to the runtime's own .values() raises on an object that has none, and a
            # health check that raises takes the whole report down with it.
            members = getattr(models, "models", None) or {}
            try:
                for member in members.values():
                    needed.update(getattr(member, "features", ()) or ())
            except AttributeError:
                needed = set()
        missing = sorted(unbounded & needed)
        if missing:
            return _check("ood_gate", DEGRADED, "bounds_missing_for:" + ",".join(missing),
                          unbounded=missing, checked=len(quality.get("checked") or []))
    if not space.get("verified"):
        # Bounds exist but were not checked against the loaded weights' dataset digest, so
        # they describe some other dataset.
        return _check("ood_gate", DEGRADED, "dataset_digest_mismatch")
    return _check("ood_gate", OK, "informative", checked=len(quality.get("checked") or []))


def reconcile_check(runtime, now):
    report = getattr(runtime, "reconciliation", None)
    if not report:
        return _check("reconciliation", SKIPPED, "not_yet_run")
    findings = report.get("findings") or []
    if findings:
        return _check("reconciliation", DEGRADED, ",".join(str(f) for f in findings),
                      checked_at=report.get("checked_at"))
    return _check("reconciliation", OK, "consistent", checked_at=report.get("checked_at"))


def venue_check(runtime, now):
    venue = getattr(runtime, "venue", None)
    if not venue:
        return _check("venue", SKIPPED, "not_checked")
    status = venue.get("status")
    if status != "ok":
        # The venue comparison is what caught a testnet socket being read against a
        # production REST host. It is a startup check and its result must stay visible.
        return _check("venue", DEGRADED, str(venue.get("reason") or status or "unknown"),
                      ws_kind=venue.get("ws_kind"), rest_kind=venue.get("rest_kind"))
    return _check("venue", OK, "matched", ws_kind=venue.get("ws_kind"),
                  spread_bps=venue.get("spread_bps"))


# Sources whose failure means the service cannot safely trade, as opposed to running with
# degraded inputs. A store that will not write loses fills; a broker that will not match
# has no book. Both are worse than a stale price, which at least refuses entries.
CRITICAL_SOURCES = ("store", "storage", "database", "persistence", "broker", "execution")


def error_check(runtime, now):
    log = getattr(runtime, "errors", None)
    entries = list(getattr(log, "entries", {}).values()) if log is not None else []
    if not entries:
        return _check("components", OK, "no_failures")
    critical = [e for e in entries
                if any(word in str(e.get("source", "")).lower() for word in CRITICAL_SOURCES)]
    sources = sorted(str(e.get("source")) for e in entries)
    if critical:
        return _check("components", DOWN, "critical_component_failed", sources=sources)
    return _check("components", DEGRADED, "component_failures", sources=sources)


# How many collection intervals may pass with nothing written before the stream is treated
# as lost rather than late.
DEAD_COLLECTOR_MULTIPLE = 3.0


def derivatives_check(runtime, now):
    """The one data source whose loss is permanent, and the one nothing was watching.

    The venue publishes open interest, long/short ratios and taker volume ratios on a
    rolling thirty-day window. There is no archive and no historical endpoint, so a day not
    collected is a day that cannot be bought back at any price. The collector caught its
    per-symbol failures into counters, returned normally, and the decision loop then
    recorded the source as healthy -- so "/health" answered ok while every cycle failed and
    the only place the failure appeared was a Prometheus counter nobody was paging on.

    Reads the collector in memory, the same numbers /metrics exports, so the two cannot
    disagree. No query, no network, no dashboard payload.
    """
    collector = getattr(runtime, "derivatives_collector", None)
    flow = getattr(runtime, "flow", None)
    if collector is None and flow is None:
        return _check("derivatives", SKIPPED, "not_configured")
    try:
        stats = dict(collector.health()) if collector is not None else {}
    except Exception as exc:
        return _check("derivatives", DEGRADED, "collector_unreadable:%s" % type(exc).__name__)
    runs = int(stats.get("runs") or 0)
    failed = int(stats.get("failed") or 0)
    rows = int(stats.get("rows") or 0)
    last_at = stats.get("last_run_at")
    age_ms = _age(now, last_at) if last_at else None
    interval_ms = float(stats.get("interval_seconds") or 0) * 1000
    detail = {"runs": runs, "rows": rows, "failed": failed,
              "last_run_age_ms": age_ms, "last_error": stats.get("last_error"),
              "symbols": stats.get("symbols")}
    if failed:
        # The failure is per symbol and the write may still be happening for others, but a
        # failing collection is how this stream is lost and there is no second chance.
        return _check("derivatives", DEGRADED, "collection_failing", **detail)
    if runs and not rows:
        return _check("derivatives", DEGRADED, "collection_empty", **detail)
    if age_ms is not None and interval_ms and age_ms > interval_ms * DEAD_COLLECTOR_MULTIPLE:
        return _check("derivatives", DEGRADED, "collection_overdue", **detail)
    if flow is not None:
        try:
            flow_stats = dict(flow.health())
        except Exception as exc:
            flow_stats = {"unreadable": type(exc).__name__}
        detail["flow"] = {key: flow_stats.get(key)
                          for key in ("trades", "liquidations", "buckets", "written",
                                      "dropped", "open_buckets", "unreadable")
                          if key in flow_stats}
        # A drain that has never written while trades have been seen is the flow table
        # failing the same way, and it is the same class of loss.
        if int(flow_stats.get("trades") or 0) and not int(flow_stats.get("written") or 0):
            return _check("derivatives", DEGRADED, "flow_never_written", **detail)
    return _check("derivatives", OK, "collecting", **detail)


def checks(runtime, now_ms=None):
    now = _now(now_ms)
    guard_ms = int(getattr(getattr(runtime, "settings", None), "market_guard_max_age_ms", 0) or 0)
    return [market_check(runtime, now, guard_ms), feed_check(runtime, now),
            model_check(runtime, now), features_check(runtime, now),
            ood_check(runtime, now), drift_check(runtime, now),
            reconcile_check(runtime, now), venue_check(runtime, now),
            derivatives_check(runtime, now), error_check(runtime, now)]


def worst(statuses):
    for level in (DOWN, DEGRADED):
        if level in statuses:
            return level
    return OK


def counters(runtime):
    """The cheap numbers the old handler returned, kept so existing consumers still work.

    These are counts, not verdicts; the checks above are the verdicts. Reported together
    so a reader can see the evidence behind a "degraded" without a second request.
    """
    cache = getattr(runtime, "cache", None)
    session = getattr(runtime, "session", None)
    broker = getattr(runtime, "broker", None)
    orders = {}
    try:
        orders = broker.order_counts() if broker is not None else {}
    except Exception:
        orders = {}
    return {
        "market_count": len((getattr(cache, "market", None) or {}).get("all") or []),
        "session_id": getattr(session, "session_id", None),
        "mode": getattr(getattr(runtime, "settings", None), "mode", None),
        "leverage": getattr(session, "leverage", None),
        "symbols": len(getattr(session, "selected_symbols", None) or ()),
        "open_orders": orders.get("working"),
        "order_status_counts": orders,
        "live_symbols": len(getattr(getattr(runtime, "guard", None), "last_event_ms", {}) or {}),
        "last_error": getattr(getattr(runtime, "errors", None), "latest", None),
    }


def report(runtime, now_ms=None):
    """The /health payload: a verdict, the per-check evidence, and the raw counters."""
    results = []
    try:
        results = checks(runtime, now_ms=now_ms)
    except Exception as exc:
        # A health endpoint that raises is indistinguishable to a monitor from one that
        # is not running, so it reports the failure instead of propagating it.
        results = [_check("health", DOWN, "report_failed:%s" % type(exc).__name__)]
    statuses = [item["status"] for item in results]
    counts = {level: statuses.count(level) for level in (OK, DEGRADED, DOWN, SKIPPED)}
    try:
        raw = counters(runtime)
    except Exception as exc:
        raw = {"error": "%s: %s" % (type(exc).__name__, exc)}
    return {"status": worst(statuses), "checked_at": _now(now_ms),
            "checks": results, "summary": counts, "counters": raw}
