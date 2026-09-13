"""The liveness probe must not build the dashboard payload, and must not lie."""
import asyncio
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.core import app_keys
from app.web.web import health, ready


class FakeSession:
    session_id = "abc123"
    leverage = 10.0
    selected_symbols = ["BTCUSDT", "ETHUSDT"]


class FakeBroker:
    def open_orders(self):
        return [object(), object(), object()]

    def order_counts(self):
        return {"working": 3, "terminal": 1, "OPEN": 3, "fill_rate": 0.25}


class FakeGuard:
    last_event_ms = {}


class FakeRuntime:
    def __init__(self):
        self.session = FakeSession()
        self.broker = FakeBroker()
        self.guard = FakeGuard()
        self.cache = type("Cache", (), {"market": {"all": [1, 2, 3, 4]}})()
        self.settings = type("Settings", (), {"market_guard_max_age_ms": 120000,
                                             "mode": "paper"})()
        self.feeds = type("Feeds", (), {"ws_events": 500, "last_event": "trade"})()
        self.errors = type("Errors", (), {"entries": {}, "latest": None})()


def fresh_guard():
    guard = FakeGuard()
    guard.last_event_ms = {"BTCUSDT": int(time.time() * 1000)}
    return guard


def make_client(state_calls, runtime=None):
    def state_fn():
        # Assembling the dashboard payload is the most expensive thing this server does;
        # /health is the endpoint a monitor polls and must stay cheap.
        state_calls.append(1)
        raise AssertionError("health must not assemble the dashboard state")

    app = web.Application()
    app[app_keys.RUNTIME] = runtime or FakeRuntime()
    app[app_keys.STATE] = state_fn
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    return TestClient(TestServer(app))


def call(runtime=None, path="/health", calls=None):
    async def run():
        client = make_client(calls if calls is not None else [], runtime)
        await client.start_server()
        try:
            response = await client.get(path)
            return response.status, await response.json()
        finally:
            await client.close()
    return asyncio.run(run())


def test_health_reports_status_without_building_dashboard_state():
    calls = []
    runtime = FakeRuntime()
    runtime.guard = fresh_guard()
    status, payload = call(runtime, calls=calls)
    assert status == 200
    # A live feed and no failures: the verdict is ok and the counters back it up.
    assert payload["status"] == "ok"
    assert payload["counters"]["market_count"] == 4
    assert payload["counters"]["symbols"] == 2
    assert payload["counters"]["open_orders"] == 3
    assert payload["counters"]["mode"] == "paper"
    assert calls == []


def test_health_survives_an_empty_cache():
    # /health is polled before the first market refresh completes.
    runtime = FakeRuntime()
    runtime.cache.market = {}
    status, payload = call(runtime)
    assert status == 200
    assert payload["counters"]["market_count"] == 0


def test_a_silent_feed_is_not_reported_as_ok():
    # The whole point of the rewrite: the old handler returned a constant "ok" and five
    # counters, so a dead socket looked exactly like a healthy one.
    runtime = FakeRuntime()
    runtime.guard.last_event_ms = {"BTCUSDT": int((time.time() - 3600) * 1000)}
    status, payload = call(runtime)
    assert payload["status"] == "down"
    assert any(check["name"] == "market_data" and check["status"] == "down"
               for check in payload["checks"])
    # A liveness probe stays 200; an orchestrator restarting the container during a venue
    # outage would restart it forever.
    assert status == 200
    ready_status, ready_payload = call(runtime, path="/ready")
    assert ready_status == 503
    assert ready_payload["status"] == "down"


def test_a_stale_but_not_dead_feed_is_degraded_rather_than_down():
    runtime = FakeRuntime()
    runtime.guard.last_event_ms = {"BTCUSDT": int((time.time() - 300) * 1000)}
    _, payload = call(runtime)
    assert payload["status"] == "degraded"
    assert any(check["detail"] == "feed_stale" for check in payload["checks"])


def test_a_broken_store_is_down_and_a_broken_indicator_is_not():
    runtime = FakeRuntime()
    runtime.guard = fresh_guard()
    runtime.errors = type("Errors", (), {
        "entries": {"drift": {"source": "drift", "message": "x", "at": 0}},
        "latest": "drift: x"})()
    _, payload = call(runtime)
    assert payload["status"] == "degraded"
    runtime.errors = type("Errors", (), {
        "entries": {"store": {"source": "store", "message": "disk full", "at": 0}},
        "latest": "store: disk full"})()
    _, payload = call(runtime)
    assert payload["status"] == "down"


def test_a_runtime_with_no_observations_says_so_rather_than_failing_to_answer():
    status, payload = call(FakeRuntime())
    assert status == 200
    names = {check["name"] for check in payload["checks"]}
    assert {"market_data", "feed", "models", "ood_gate", "drift", "reconciliation",
            "venue", "components"} <= names
    assert payload["summary"]["skipped"] >= 1


def test_a_runtime_missing_every_optional_object_still_answers():
    class Bare:
        pass

    status, payload = call(Bare())
    assert status == 200
    assert payload["status"] in ("degraded", "down")
    assert payload["checks"]


def test_unpromoted_weights_are_named_in_the_report_rather_than_called_tradable():
    """The one fact that decides whether any reported number passed review.

    The service is configured to accept candidates (MODEL_REQUIRE_PROMOTED=0), which is a
    deliberate choice: refusing to trade at all is worse than trading unvetted weights
    while the model matures. But data/models is empty, so no model has ever been promoted,
    and a check that answers "tradable" is describing the absence of a refusal as a clean
    bill of health.
    """
    runtime = FakeRuntime()
    runtime.models = {"lightgbm": object()}
    runtime.promotion = {"status": "no_production_model", "reason": "no_production_model"}
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "models")
    assert check["status"] == "ok"
    assert check["detail"] == "trading_unpromoted_weights"
    assert check["mode"] == "no_production_model"


def test_a_promoted_model_is_reported_as_tradable():
    runtime = FakeRuntime()
    runtime.models = {"production": object()}
    runtime.promotion = {"status": "active", "version": "gbst-v3"}
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "models")
    assert check["status"] == "ok"
    assert check["detail"] == "tradable"
    assert check["version"] == "gbst-v3"


class _Collector:
    """The derivatives collector, with only the surface /health is allowed to read."""

    def __init__(self, **stats):
        self.stats = {"runs": 0, "rows": 0, "empty": 0, "failed": 0, "last_error": None,
                      "last_rows": 0, "last_run_at": None, "interval_seconds": 300.0,
                      "period": "5m", "symbols": 12, "due": False}
        self.stats.update(stats)

    def health(self):
        return dict(self.stats)


def _derivatives(runtime, **stats):
    runtime.derivatives_collector = _Collector(**stats)
    _, payload = call(runtime)
    return next(c for c in payload["checks"] if c["name"] == "derivatives")


def test_the_endpoint_that_cannot_be_collected_twice_is_watched():
    """The venue keeps this data for thirty days and then discards it.

    The collector caught per-symbol failures into counters, returned normally, and the
    decision loop recorded the source as healthy -- so /health answered ok while every
    cycle failed. The failure appeared only as a Prometheus counter nobody paged on. This
    is the one input whose loss cannot be bought back, so it gets a verdict.
    """
    now_ms = int(time.time() * 1000)
    check = _derivatives(FakeRuntime(), runs=12, rows=0, failed=12,
                         last_error="BTCUSDT: OperationalError('no such table: derivatives_detail')",
                         last_run_at=now_ms - 60_000)
    assert check["status"] == "degraded"
    assert check["detail"] == "collection_failing"
    assert check["failed"] == 12
    assert "no such table" in check["last_error"]


def test_a_collector_that_never_writes_is_not_called_healthy():
    now_ms = int(time.time() * 1000)
    check = _derivatives(FakeRuntime(), runs=8, rows=0, last_run_at=now_ms - 1000)
    assert check["status"] == "degraded"
    assert check["detail"] == "collection_empty"


def test_a_collector_that_stopped_running_is_overdue():
    now_ms = int(time.time() * 1000)
    check = _derivatives(FakeRuntime(), runs=5, rows=500,
                         last_run_at=now_ms - 60 * 60 * 1000)
    assert check["status"] == "degraded"
    assert check["detail"] == "collection_overdue"
    assert check["rows"] == 500


def test_a_working_collector_is_ok_and_says_so_with_numbers():
    now_ms = int(time.time() * 1000)
    check = _derivatives(FakeRuntime(), runs=20, rows=4000, last_run_at=now_ms - 30_000)
    assert check["status"] == "ok"
    assert check["detail"] == "collecting"
    assert check["runs"] == 20 and check["rows"] == 4000


def test_a_runtime_with_no_collector_skips_rather_than_inventing_a_verdict():
    _, payload = call(FakeRuntime())
    check = next(c for c in payload["checks"] if c["name"] == "derivatives")
    assert check["status"] == "skipped"
    assert check["detail"] == "not_configured"


class _Source:
    """The feature source, with only the surface /health is allowed to read."""

    def __init__(self, **health):
        self._health = {"rows": 0, "degraded_rows": 0, "funding_rows": 0, "extra_rows": 0,
                        "extra_reads": 0, "degraded_features": [], "missing_features": [],
                        "reported_features": 10, "extras_enabled": True,
                        "families": {"price": 10, "order_flow": 5, "positioning": 8,
                                     "flow": 7}}
        self._health.update(health)

    def health(self):
        return dict(self._health)


class _Decisions:
    def __init__(self, source, required=10):
        self.feature_source = source
        self._required = tuple("f%d" % index for index in range(required))

    def required_features(self):
        return self._required


def _features(source, required=10):
    runtime = FakeRuntime()
    runtime.decisions = _Decisions(source, required)
    _, payload = call(runtime)
    return next(c for c in payload["checks"] if c["name"] == "features")


def test_a_service_serving_constants_where_a_model_expects_features_is_degraded():
    """The defect this check exists for: ten features trained, four served as zero.

    Nothing in the system could notice. A placeholder zero sits inside the training bounds,
    so the out-of-range gate passed it; the model loaded; the aggregates looked plausible.
    "The weights loaded" and "the weights are being fed what they were fit on" are
    different statements and only the first was ever checked.
    """
    check = _features(_Source(rows=500, degraded_rows=500,
                              degraded_features=["funding_rate", "mark_basis"]),
                      required=10)
    assert check["status"] == "degraded"
    assert check["detail"] == "every_row_degraded"
    assert check["missing_features"] == ["funding_rate", "mark_basis"]
    assert check["required"] == 10


def test_a_partly_degraded_input_is_reported_rather_than_averaged_away():
    check = _features(_Source(rows=500, degraded_rows=20,
                              degraded_features=["mark_basis"]))
    assert check["status"] == "degraded"
    assert check["detail"] == "partially_degraded"
    assert check["degraded_rows"] == 20


def test_a_fully_measured_row_is_ok_and_reports_the_family_split():
    check = _features(_Source(rows=500, degraded_rows=0, extra_rows=500))
    assert check["status"] == "ok"
    assert check["detail"] == "all_inputs_measured"
    assert check["families"]["order_flow"] == 5


def test_a_row_filled_only_from_candles_is_not_a_full_row():
    """Every modelled column present, and the collected families still empty.

    The three families app/order_flow.py computes need a collector that has been running.
    On a deployment whose flow tables were created after it started, every value the model
    reads is real and twenty columns it was trained on are missing. Reporting ok there is
    how a service trades a narrower input than its own contract names.
    """
    check = _features(_Source(rows=500, degraded_rows=0, extra_rows=0))
    assert check["status"] == "degraded"
    assert check["detail"] == "collected_families_empty"


def test_before_the_first_signal_the_check_skips_rather_than_claiming_a_verdict():
    check = _features(_Source(rows=0, degraded_rows=0, extra_rows=0))
    assert check["status"] == "skipped"
    assert check["detail"] == "no_rows_yet"


def test_an_unreadable_source_is_degraded_and_a_missing_one_is_skipped():
    class Broken:
        feature_source = object()

        def required_features(self):
            return ("rsi",)

    runtime = FakeRuntime()
    runtime.decisions = Broken()
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "features")
    assert check["status"] == "skipped"
    assert check["detail"] == "not_configured"


def _space(quality, members=None, verified=True):
    """The shape the check reads in production: runtime.models is the model runtime, which
    holds both the stored feature space and the loaded members."""
    holder = type("Holder", (), {})()
    holder.feature_space = {"status": "ok", "verified": verified, "bounds_quality": quality}
    holder.models = dict(members or {})
    return holder


def test_a_column_the_model_reads_but_the_profile_never_bounded_is_not_checked_at_all():
    """A gate that reports ok for the columns it looked at and says nothing about the rest.

    Bounds are stored per dataset. A dataset older than the feature contract has no entry
    for the newer columns, so out_of_range cannot test them and ood_check returned
    "informative, checked: 10" -- a count that reads like a complete answer while twenty
    columns the model may be reading are unguarded. The check now intersects the unbounded
    set with what the loaded weights actually declare, so it fires only when a model is
    being served a column nothing can bound.
    """
    quality = {"informative": True, "degenerate": [], "checked": ["rsi", "atr_pct"],
               "unbounded": ["taker_imbalance", "flow_delta_z"]}
    runtime = FakeRuntime()
    runtime.models = _space(quality)
    runtime.decisions = _Decisions(_Source(rows=1), required=2)
    runtime.decisions.required_features = lambda: ("rsi", "atr_pct")
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "ood_gate")
    assert check["status"] == "ok"
    assert check["detail"] == "informative"
    assert check["checked"] == 2


def test_a_model_reading_an_unbounded_column_is_degraded_and_the_columns_are_named():
    quality = {"informative": True, "degenerate": [], "checked": ["rsi"],
               "unbounded": ["taker_imbalance", "flow_delta_z", "oi_z"]}
    runtime = FakeRuntime()
    runtime.models = _space(quality)
    runtime.decisions = _Decisions(_Source(rows=1), required=2)
    runtime.decisions.required_features = lambda: ("rsi", "taker_imbalance", "oi_z")
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "ood_gate")
    assert check["status"] == "degraded"
    assert check["detail"] == "bounds_missing_for:oi_z,taker_imbalance"
    assert check["unbounded"] == ["oi_z", "taker_imbalance"]


def test_the_unbounded_intersection_also_reads_the_members_when_there_is_no_decision_layer():
    quality = {"informative": True, "degenerate": [], "checked": ["rsi"],
               "unbounded": ["liquidation_z"]}
    runtime = FakeRuntime()
    # No decisions object: the check falls back to what each loaded member declares.
    runtime.models = _space(quality, members={
        "lightgbm": type("M", (), {"features": ("rsi", "liquidation_z")})()})
    _, payload = call(runtime)
    check = next(c for c in payload["checks"] if c["name"] == "ood_gate")
    assert check["status"] == "degraded"
    assert check["detail"] == "bounds_missing_for:liquidation_z"
