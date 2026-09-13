"""Service assembly.

run() used to be a 416-line function holding fourteen nested closures over twenty
mutable locals, which made the paper decision path impossible to test in isolation.
Assembly now happens here and every behaviour lives in a module:

    runtime_context : shared mutable state, declared in one place
    snapshot        : HTTP state projection
    decision_loop   : model-driven entry logic (tick + bar)
    execution       : order settlement
    pumps           : background market-data loops

Trading behaviour is unchanged by this file; it only wires the pieces together.
"""
import asyncio
import dataclasses
import logging
from pathlib import Path

from aiohttp import web

from .core import app_keys
from .web.api_routes import build_app
from .core.config import Settings
from .strategy.decision_loop import evaluate_tick, run_decision_loop
from .features.feature_source import FeatureSource
from .strategy.live_models import ModelDecision, RealModelRuntime
from .core.logging_json import configure
from .market.market import BinancePublic
from .storage.persistence import BatchWriter
from .storage.database_worker import DatabaseWorker
from .market.pumps import mark_pump, price_pump, realtime_pump
from .trading.guards import PortfolioLimits
from .trading.risk import RiskEngine
from .trading import exit_policy, risk_profiles; from .strategy import symbol_edge
from .runtime_context import BackgroundTasks, Runtime
from .strategy.session_control import close_position, on_session_start, reset_risk, set_exit_policy, set_profile
from .strategy.shadow import ShadowModels
from .backtest.simulation_session import SimulationSession
from .backtest.snapshot import build_chart_data, build_realtime_state, build_state
from .storage.storage import Store

log = logging.getLogger(__name__)


def _build_models(settings, broker, broker_store):
    """Model runtime plus the decision layer, priced at the broker's real cost model."""
    models = RealModelRuntime(settings.data_dir, device=settings.model_device,
                              chronos_enabled=settings.model_chronos_enabled,
                              max_chronos_symbols=settings.model_max_chronos_symbols,
                              chronos_horizon=settings.model_chronos_horizon,
                              allow_candidate_fallback=settings.model_allow_candidate_fallback)
    decisions = ModelDecision(models, fee_rate=broker.fee_rate, slippage_bps=broker.slippage_bps,
                              maker_fee_rate=settings.maker_fee_rate,
                              entry_order_type=settings.entry_order_type,
                              min_edge_bps=settings.model_min_edge_bps,
                              min_agreement=settings.model_min_agreement,
                              min_edge_multiple=settings.min_edge_multiple,
                              rule_fallback=settings.model_rule_fallback,
                              ood_policy=settings.model_ood_policy,
                              chronos_timeout_ms=settings.model_chronos_timeout_ms,
                              # The same builder the training job uses. Serving without it
                              # is what left four of the ten modelled features constant.
                              feature_source=FeatureSource(
                                  broker_store,
                                  max_basis_age_ms=settings.max_basis_age_ms),
                              degraded_blocks=(settings.feature_degraded_policy != 'warn'),
                              require_promoted=settings.model_require_promoted,
                              max_model_age_ms=int(settings.model_max_age_hours * 3600 * 1000))
    # Seeded from decisions already in the audit trail, so the gate starts from measured
    # history instead of a warmup period in which it would have to trade blind.
    edge = symbol_edge.seed_from_history(
        broker_store, settings.interval, settings.symbol_edge_horizon_bars,
        decisions.round_trip_cost_pct, window=settings.symbol_edge_window)
    edge.min_samples = settings.symbol_edge_min_samples
    edge.min_net_bps = settings.symbol_edge_min_net_bps
    edge.min_t = settings.symbol_edge_min_t
    edge.fdr = settings.symbol_edge_fdr
    edge.enabled = settings.symbol_edge_enabled
    edge.require_samples = settings.symbol_edge_require_samples
    decisions.symbol_edge = edge
    models.on_chronos_result = decisions.invalidate
    return models, decisions


def _measured_horizon(settings, store, decisions, policy):
    """Return the exit policy with its time stop aligned to measured edge persistence.

    Returns None when alignment is off, when no horizon shows a positive cost-adjusted
    edge, or when the stored policy already carries a time stop. All three are reasons to
    leave the operator's configuration alone rather than to override it.
    """
    if not settings.exit_align_to_measured_horizon:
        return None
    if int(policy.time_stop_bars or 0) > 0:
        return None
    try:
        # The sample floor is the configured one, not the literal 8 this used to pass.
        # Eight overlapping observations per horizon, argmax over seven horizons and no
        # multiplicity correction is a parameter fitted to noise -- and it was written
        # straight into the live time stop.
        analysis = symbol_edge.horizon_analysis(
            store, settings.interval, cost=decisions.round_trip_cost_pct,
            min_samples=max(30, int(settings.symbol_edge_min_samples)))
    except Exception as exc:
        log.warning('horizon analysis unavailable: %r', exc)
        return None
    best = analysis.get('best_horizon')
    store.set_runtime('measured_horizon', {'best_horizon': best,
                                           'pooled': analysis.get('pooled') or {}})
    if not best:
        log.info('no horizon cleared its own error bar (positive but unproven: %s); '
                 'leaving the time stop disabled',
                 analysis.get('positive_but_unproven') or [])
        return None
    aligned = dataclasses.replace(policy, time_stop_bars=int(best))
    log.info('exit time stop aligned to measured horizon: %s bars', best)
    return aligned


def _static_venue_check(settings):
    """The free half: do the two configured hosts describe the same venue?"""
    from .market.venue_check import classify_host, host_mismatch

    ws_url = str(getattr(settings, 'ws_url', '') or '')
    rest_url = str(getattr(settings, 'base_url', '') or '')
    report = {'status': 'ok', 'findings': [], 'ws_url': ws_url, 'rest_url': rest_url,
              'ws_kind': classify_host(ws_url), 'rest_kind': classify_host(rest_url)}
    mismatch = host_mismatch(ws_url, rest_url)
    if mismatch:
        report['status'] = 'suspect'
        report['findings'] = ['mixed_venue']
        report['mismatch'] = mismatch
        print('[venue] WARNING market data from %s (%s) and prices from %s (%s)'
              % (mismatch['ws'], ws_url, mismatch['rest'], rest_url), flush=True)
    return report


async def _check_venue(api, settings):
    """Compare the configured stream against the configured REST host before startup.

    A failure here is a configuration error with no symptom: a wrong stream still produces
    prices, still fills orders and still reports a plausible equity curve.
    """
    from .market.venue_check import check_venue

    policy = str(getattr(settings, 'venue_check_policy', 'warn') or 'warn').lower()
    if policy == 'off':
        return {'status': 'skipped', 'policy': policy}
    symbol = (list(settings.symbols) or ['BTCUSDT'])[0]
    try:
        report = await check_venue(api, settings, symbol=symbol,
                                   seconds=float(getattr(settings, 'venue_check_seconds', 6.0)))
    except Exception as exc:
        return {'status': 'unknown', 'policy': policy, 'error': repr(exc)}
    report['policy'] = policy
    if report.get('status') == 'suspect':
        detail = ', '.join(report.get('findings') or [])
        message = ('market data venue disagrees with the price venue: %s (ws=%s rest=%s '
                   'mid=%s rest_last=%s deviation_bps=%s spread_bps=%s)'
                   % (detail, report.get('ws_kind'), report.get('rest_kind'),
                      report.get('mid'), report.get('rest_last'),
                      report.get('deviation_bps'), report.get('spread_bps')))
        if policy == 'block':
            raise RuntimeError('venue_check_failed: ' + message)
        print('[venue] WARNING ' + message, flush=True)
    return report


def profile_state(runtime, name=None, overrides=None, switch=True):
    """Read or switch the risk profile and report it in one shape.

    The response always carries the effective parameters alongside the catalogue, so the
    caller sees what is actually in force rather than what it asked for.
    """
    if switch and (name or overrides):
        set_profile(runtime, name, overrides)
    active = runtime.store.get_runtime('risk_profile') or risk_profiles.get_profile(
        runtime.store.get_runtime('risk_profile_name')).snapshot()
    return {'active': active,
            'name': runtime.store.get_runtime('risk_profile_name'),
            'effective': runtime.risk.snapshot(),
            **risk_profiles.describe()}


def symbol_edge_state(runtime):
    """Per-symbol track record plus the horizon the service measured at startup.

    This is what explains why a symbol is not being traded: the gate rejects on measured
    cost-adjusted edge, so the evidence has to be visible or the refusal looks arbitrary.
    """
    tracker = getattr(getattr(runtime, 'decisions', None), 'symbol_edge', None)
    measured = runtime.store.get_runtime('measured_horizon') or {}
    return {'enabled': bool(tracker and tracker.enabled),
            'horizon_bars': tracker.horizon_bars if tracker else None,
            'min_samples': tracker.min_samples if tracker else None,
            'min_net_bps': tracker.min_net_bps if tracker else None,
            'round_trip_cost_bps': round((tracker.round_trip_cost if tracker else 0.0) * 10000, 4),
            'measured_horizon': measured.get('best_horizon'),
            'measured_pooled': measured.get('pooled') or {},
            'symbols': tracker.table() if tracker else []}


def exit_state(runtime, name=None, overrides=None, switch=True):
    """Read or switch the exit policy, reporting what is actually in force."""
    if switch and (name or overrides):
        set_exit_policy(runtime, name, overrides)
    active = runtime.store.get_runtime('exit_policy') or exit_policy.get_policy(
        runtime.store.get_runtime('exit_policy_name')).snapshot()
    live = getattr(runtime.decisions, 'exit_policy', None)
    return {'active': active,
            'name': runtime.store.get_runtime('exit_policy_name'),
            'effective': live.snapshot() if live else active,
            **exit_policy.describe()}


async def run():
    configure()
    settings = Settings()
    if settings.mode != 'paper':
        raise RuntimeError('Only paper mode is implemented')

    api = BinancePublic(settings.base_url, settings.data_dir)
    # The configured stream and the configured REST host must be the same venue. They were
    # not: market data came from a testnet host while prices, funding and instrument filters
    # came from production, and nothing compared the two for the entire life of the project.
    #
    # The comparison has a cheap half and an expensive half. The host classification is
    # synchronous and free, so it always runs before anything is built on top of the
    # clients. Sampling the live book costs several seconds, so under 'warn' it runs as a
    # background task and never delays readiness; under 'block' it runs inline, because the
    # only way to refuse to start is to do it before starting.
    venue_policy = str(getattr(settings, 'venue_check_policy', 'warn') or 'warn').lower()
    venue = _static_venue_check(settings)
    if venue_policy == 'block':
        venue = await _check_venue(api, settings)
    store = Store(settings.data_dir)
    session = SimulationSession.restore(store, store.get_runtime('simulation_session'), settings.starting_equity)
    if not session.account.trades:
        # Only this session's own fills. Loading the whole trades table into the live
        # account book is what session.end() then summarised: win rate, profit factor,
        # average holding time and best/worst trade were computed over every trade the
        # service had ever recorded, while return_pct divided by *this* session's initial
        # cash. The two numbers came from different populations and neither was wrong
        # enough to look wrong.
        try:
            session.account.trades = list(reversed(store.trades_for_session(session.session_id, 500)))
        except Exception:
            session.account.trades = []
    risk = RiskEngine.restore(store.get_runtime('risk_state'), session.account.equity)
    # The active profile overrides whatever the engine was restored with, so a restart
    # cannot leave the service trading on parameters nobody chose. The profile is
    # persisted separately from the engine's running state.
    profile = risk_profiles.get_profile(store.get_runtime('risk_profile_name') or settings.risk_profile)
    risk.apply_profile(profile)
    # Limiter state, installed before anything can approve an entry. Both default to
    # disabled at 0 so an operator who has not chosen them keeps the old behaviour.
    risk.drawdown_limit = float(settings.max_drawdown_from_peak or 0.0)
    risk.loss_streak_limit = int(settings.loss_streak_limit or 0)
    risk.loss_streak_scale = float(settings.loss_streak_scale or 0.5)
    risk.daily_trade_limit = int(settings.max_daily_trades or 0)
    risk.target_volatility = float(settings.target_bar_volatility or 0.0)
    risk.observe_equity(session.account.equity)
    # The portfolio caps are a second copy of the same numbers, and only the runtime
    # profile switcher updated both. After a restart with 'maximum' persisted, the
    # dashboard reported 20x gross leverage while PortfolioLimits still enforced the
    # .env values, so the panel and the order path disagreed about the same session.
    limits = PortfolioLimits.from_profile(profile)
    # After from_profile, which resets the caps from the profile. Assigning these before it
    # was a no-op: the profile does not carry a concentration cap, so the constructor's
    # default of 0 was written back over the configured value.
    limits.max_correlated_leverage = float(settings.max_correlated_leverage or 0.0)
    limits.correlation_threshold = float(settings.correlation_threshold or 0.6)
    store.set_runtime('risk_state', risk.snapshot())
    store.set_runtime('risk_profile', profile.snapshot())
    exit_policy_active = exit_policy.get_policy(store.get_runtime('exit_policy_name') or settings.exit_policy)
    store.set_runtime('exit_policy', exit_policy_active.snapshot())
    store.set_runtime('exit_policy_name', exit_policy_active.name)
    from .trading.broker import PaperBroker
    from .trading.fill_models import FillModel

    # The matching models are stated here rather than left to the constructor defaults.
    # They were the defaults: the live account priced every fill with a hardcoded 0.0400%
    # taker fee and a 2 bp size-independent slippage, and nothing in .env could change
    # either, so no stored session could be explained by the costs it assumed.
    broker = PaperBroker(order_sink=store.save_order, order_ttl_ms=settings.order_ttl_ms,
                         fee_rate=settings.taker_fee_rate,
                         maker_fee_rate=settings.maker_fee_rate,
                         slippage_bps=settings.spread_bps,
                         fill_model=FillModel(spread_bps=settings.spread_bps,
                                              impact_coefficient_bps=settings.impact_bps,
                                              entry_latency_ms=settings.entry_latency_ms,
                                              latency_adverse_share=settings.latency_adverse_share))
    store.set_runtime('risk_profile_name', profile.name)
    models, decisions = _build_models(settings, broker, store)
    # Alignment needs the decisions object, because the measured edge is expressed net of
    # the round-trip cost that object owns. It runs after the models are built so the
    # holding period follows the data rather than the geometry.
    measured = _measured_horizon(settings, store, decisions, exit_policy_active)
    if measured is not None:
        exit_policy_active = measured
        store.set_runtime('exit_policy', exit_policy_active.snapshot())
    decisions.exit_policy = exit_policy_active
    runtime = Runtime.build(settings=settings, api=api, store=store, session=session, risk=risk,
                            broker=broker, models=models, decisions=decisions, limits=limits)
    runtime.shadow = ShadowModels(settings.data_dir, device=settings.model_device, runtime=models)
    runtime.venue = venue
    runtime.restore_orders()
    # Real contract filters before the first decision. Without them every symbol rounds
    # to the same hardcoded tick and lot step, which is only correct for a handful.
    try:
        broker.specs.update(await api.contract_specs())
    except Exception as exc:
        log.warning('contract specs unavailable, using defaults: %r', exc)
    # The venue's own maintenance brackets. A flat rate understates the requirement of
    # large positions, which are exactly the ones whose liquidation matters.
    try:
        brackets = await api.leverage_brackets()
        if brackets:
            session.account.margin.schedule.update(brackets)
            log.info('maintenance brackets loaded for %d symbols', len(brackets))
    except Exception as exc:
        log.warning('leverage brackets unavailable, using published defaults: %r', exc)

    if session.status not in ('running', 'paused'):
        risk.reset_for_session(session.account.equity)

    database_worker = DatabaseWorker(settings.data_dir)
    # Published on the runtime, not just captured in a closure: the retention sweep in
    # decision_loop reached for runtime.database_worker on every pass and raised
    # AttributeError into a bare except, so aged audit rows were never once pruned.
    runtime.database_worker = database_worker
    if settings.derivatives_collect_enabled:
        from .market.derivatives_collect import DerivativesCollector

        runtime.derivatives_collector = DerivativesCollector(
            api, store, period=settings.derivatives_collect_period,
            interval_seconds=settings.derivatives_collect_interval_seconds)
    if settings.flow_collect_enabled:
        from .market.flow_collect import FlowCollector

        runtime.flow = FlowCollector(
            drain_interval_ms=int(settings.flow_drain_seconds * 1000))
    if settings.drift_check_enabled:
        from .models.drift_monitor import DriftMonitor

        # The reference population is the dataset the loaded models were fitted on, not the
        # recent past: the question is whether the live inputs still look like the training
        # inputs, which comparing two recent windows can never answer.
        dataset = Path(settings.data_dir) / 'research_v3' / 'training_dataset.jsonl'
        runtime.drift = DriftMonitor.from_dataset(
            dataset, window=settings.drift_window, threshold=settings.drift_psi_threshold,
            interval_seconds=settings.drift_interval_seconds, policy=settings.drift_policy)
    runtime.event_writer = BatchWriter(lambda batches: database_worker.call('record_events', batches),
                                       batch_size=100, flush_interval=0.1)
    runtime.candle_writer = BatchWriter(lambda batches: database_worker.call('upsert_candles_batch', batches),
                                        batch_size=20, flush_interval=0.2)
    runtime.event_writer.start()
    runtime.candle_writer.start()

    tasks = BackgroundTasks(log)
    app = build_app(
        runtime,
        state_fn=lambda: build_state(runtime, connectors()),
        realtime_state_fn=lambda: build_realtime_state(runtime),
        chart_fn=lambda symbol: build_chart_data(runtime, symbol),
        on_session_start_fn=lambda session_id, name=None, overrides=None: on_session_start(runtime, name, overrides),
        close_position_fn=lambda symbol: close_position(runtime, symbol),
        reset_risk_fn=lambda: reset_risk(runtime),
        model_state_fn=lambda: model_api_state(runtime),
        set_risk_profile_fn=lambda name, overrides, switch=True: profile_state(runtime, name, overrides, switch),
        set_exit_policy_fn=lambda name, overrides, switch=True: exit_state(runtime, name, overrides, switch),
        symbol_edge_state_fn=lambda: symbol_edge_state(runtime),
    )

    from .models.training_job import TrainingRunner

    async def market_rows():
        """Live market rows, fetched on demand when the cache is still cold."""
        cached = list(runtime.cache.market.get('all') or [])
        if cached:
            return cached
        try:
            snapshot = await runtime.api.market_snapshot()
        except Exception:
            return []
        runtime.broker.specs.update(snapshot.pop('specs', None) or {})
        runtime.cache.market.update(snapshot)
        return list(snapshot.get('all') or [])

    training = TrainingRunner(settings, runtime.store)
    app[app_keys.TRAINING] = training
    app[app_keys.MARKET_ROWS] = market_rows

    # RetrainScheduler existed with a full run/stop/snapshot lifecycle and no caller, so a
    # model could only ever be refreshed by an operator pressing a button. Training here is
    # explicitly not an auto-promotion: the scheduler produces a new candidate and the
    # registry gate still decides whether it becomes production.
    if settings.retrain_enabled:
        from .features.feature_spec import FEATURE_VERSION
        from .models.model_registry import ModelRegistry
        from .models.retrain import RetrainScheduler
        from .models.training_job import TIER_MAINSTREAM

        def current_manifest():
            try:
                return ModelRegistry(str(models.registry_root)).current()
            except Exception:
                return None

        async def retrain_once():
            job = training.start(TIER_MAINSTREAM)
            await training.execute(job, market_rows())
            return job.snapshot()

        def drift_report():
            monitor = getattr(runtime, "drift", None)
            return getattr(monitor, "last_report", None) or {}

        runtime.retrainer = RetrainScheduler(
            retrain_once, current_manifest,
            interval_ms=int(settings.retrain_interval_seconds * 1000),
            feature_version=FEATURE_VERSION, drift_fn=drift_report)

    def connectors():
        ws = runtime.feeds.ws
        public = runtime.feeds.public
        return {
            'ws_reconnects': getattr(ws, 'reconnects', 0),
            'connector_health': ws.health.snapshot() if ws else {},
            'connector_routes': {'market': ws.health.snapshot() if ws else {},
                                 'public': public.health.snapshot() if public else {}},
            'feed_normalizer': ws.normalizer.stats() if ws else {},
        }

    from .market.websocket_feed import BinanceWebSocketFeed

    runtime.feeds.ws = BinanceWebSocketFeed(settings.ws_url, list(settings.symbols), on_ws(runtime),
                                            interval=settings.interval, category='market',
                                            proxy=settings.ws_proxy)
    runtime.feeds.public = BinanceWebSocketFeed(settings.ws_url, list(settings.symbols), on_ws(runtime),
                                                interval=settings.interval, category='public',
                                                proxy=settings.ws_proxy)
    tasks.spawn(runtime.feeds.ws.run())
    tasks.spawn(runtime.feeds.public.run())
    tasks.spawn(models.ensure_loaded())
    models.start()
    runtime.shadow.start()

    if venue_policy == 'warn':
        tasks.spawn(_venue_watch(runtime, api, settings))
    tasks.spawn(realtime_pump(runtime, app))
    tasks.spawn(price_pump(runtime))
    tasks.spawn(mark_pump(runtime))
    tasks.spawn(_decision_loop(runtime))
    if runtime.retrainer is not None:
        tasks.spawn(runtime.retrainer.start())

    app.on_cleanup.append(lambda _app: cleanup(runtime, tasks, database_worker))
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 8101).start()
        print({'event': 'dashboard', 'url': 'http://127.0.0.1:8101'}, flush=True)
        await asyncio.Event().wait()
    finally:
        try:
            await runner.cleanup()
        finally:
            if api.session is not None:
                await api.session.close()
            store.close()


async def _venue_watch(runtime, api, settings):
    """Sample the live book once, after startup, and record what it found."""
    try:
        report = await _check_venue(api, settings)
        static = dict(getattr(runtime, 'venue', None) or {})
        report['static'] = static.get('findings') or []
        runtime.venue = report
    except Exception as exc:
        runtime.venue = dict(getattr(runtime, 'venue', None) or {},
                             sampling_error=repr(exc))


async def _decision_loop(runtime):
    while True:
        try:
            await run_decision_loop(runtime)
        except Exception as exc:
            runtime.errors.note('decision_loop', exc)
        else:
            runtime.errors.ok('decision_loop')
        await asyncio.sleep(runtime.settings.poll_seconds)


async def cleanup(runtime, tasks, database_worker):
    runtime.feeds.ws.running = False
    runtime.feeds.public.running = False
    await tasks.cancel_all()
    await runtime.shadow.close()
    await runtime.models.close()
    try:
        await asyncio.gather(runtime.event_writer.stop(), runtime.candle_writer.stop())
    finally:
        await database_worker.close()
        runtime.store.set_runtime('paper_account', runtime.account.snapshot())


def on_ws(runtime):
    """Build the websocket event handler bound to this runtime."""

    async def handler(event):
        from .strategy.feeds_handlers import handle_ws_event

        await handle_ws_event(runtime, event, evaluate_tick)

    return handler


def model_api_state(runtime):
    research = runtime.shadow.status()
    decision = runtime.decisions.status()
    return {'mode': decision['mode'], 'production': decision['production'],
            'queued': research.get('queued', 0), 'models': research.get('models', []),
            'decision': decision}


if __name__ == '__main__':
    asyncio.run(run())
