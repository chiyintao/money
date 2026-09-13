"""Explicit container for the mutable state that used to be loose locals in run().

The previous design kept ~15 mutable values as closure variables inside a 416-line
run(), which made every helper depend on everything and made the data flow
impossible to follow. This container is dataclass-based so the fields are declared
in one place and can be inspected or reset deliberately.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core.background import BackgroundTasks

from .trading.broker import PaperBroker
from .core.config import Settings
from .trading.guards import MarketGuard, PortfolioLimits, portfolio_limits
from .strategy.live_models import ModelDecision, RealModelRuntime
from .market.market import BinancePublic
from .core import order_state
from .storage.persistence import BatchWriter
from .trading.risk import RiskEngine
from .backtest.simulation_session import SimulationSession
from .storage.storage import Store


@dataclass
class Feeds:
    """Transport-layer objects and their health counters."""

    ws: object = None
    public: object = None
    ws_events: int = 0
    last_event: str | None = None
    last_mark_event: dict = field(default_factory=dict)
    last_event_audit: dict = field(default_factory=dict)


@dataclass
class MarketCache:
    """Latest known quote per symbol plus the rolling closed-candle window."""

    market: dict = field(default_factory=dict)
    latest: dict = field(default_factory=dict)
    latest_rows: dict = field(default_factory=dict)
    closed_rows: dict = field(default_factory=dict)
    live_candles: dict = field(default_factory=dict)
    last_refresh: float = 0.0
    last_retention: float = 0.0
    last_reconcile: float = 0.0


@dataclass
class DecisionState:
    """Per-symbol throttling and last-emitted decision, used to avoid duplicate orders."""

    per_symbol: dict = field(default_factory=dict)
    last_tick_strategy: dict = field(default_factory=dict)
    processed_bars: dict = field(default_factory=dict)
    stored_trades: int = 0
    # Last decision actually written to the audit trail, per symbol, so the tick path
    # can tell a repeat of the previous side from a change worth recording.
    last_signal: dict = field(default_factory=dict)
    # When each symbol's current position was opened, so the exit policy's time stop can
    # count bars held. Cleared when the position closes.
    position_opened_at: dict = field(default_factory=dict)


@dataclass
class ErrorLog:
    """Recent failures, keyed by the component that produced them.

    This replaces a single `last_error` string that the decision loop cleared at the top
    of every iteration. Any component whose failure was written by a different one -- the
    mark-price pump, the REST price fallback, the bar loop -- had its evidence erased the
    next time the decision loop succeeded, so a permanently broken feed still showed a
    clean dashboard. Failures are now attributed and only the reporting component can
    clear its own entry.
    """

    entries: dict = field(default_factory=dict)

    def note(self, source, message, now_ms=None):
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        previous = self.entries.get(source) or {}
        self.entries[source] = {
            'source': source, 'message': str(message)[:400], 'at': now,
            'count': int(previous.get('count', 0)) + 1,
            'first_at': int(previous.get('first_at') or now),
        }
        return self.entries[source]

    def ok(self, source):
        """Clear one source's failure. Never touches another source's entry."""
        self.entries.pop(source, None)

    @property
    def latest(self):
        if not self.entries:
            return None
        newest = max(self.entries.values(), key=lambda item: item['at'])
        return '%s: %s' % (newest['source'], newest['message'])

    def snapshot(self, now_ms=None):
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        items = sorted(self.entries.values(), key=lambda item: item['at'], reverse=True)
        return {'latest': self.latest, 'count': len(items),
                'entries': [{**item, 'age_ms': max(0, now - int(item['at']))} for item in items]}


@dataclass
class Runtime:
    """Everything the HTTP handlers, pumps and decision loop need to share."""

    settings: Settings
    store: Store
    api: BinancePublic
    session: SimulationSession
    risk: RiskEngine
    broker: PaperBroker
    models: RealModelRuntime
    decisions: ModelDecision
    guard: MarketGuard = field(default_factory=MarketGuard)
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    # limits is derived from Settings in build(); the default keeps the dataclass
    # constructible in tests that do not care about portfolio caps.
    feeds: Feeds = field(default_factory=Feeds)
    cache: MarketCache = field(default_factory=MarketCache)
    decision_state: DecisionState = field(default_factory=DecisionState)
    pending_plans: dict = field(default_factory=dict)
    shadow: object = None
    model_status: dict = field(default_factory=dict)
    errors: ErrorLog = field(default_factory=ErrorLog)
    event_writer: BatchWriter | None = None
    candle_writer: BatchWriter | None = None
    # The database thread. Retention runs there rather than on the event loop, and it is
    # built in main() *after* Runtime.build(), which is how the retention sweep silently
    # called a missing attribute on every pass for the entire life of the service.
    database_worker: object = None
    # Collects the derivative statistics whose history the venue discards after 30 days.
    derivatives_collector: object = None
    # Result of the startup market-data venue comparison.
    venue: dict = None
    # Feature drift monitor and the supervised retraining scheduler.
    drift: object = None
    retrainer: object = None
    # Per-minute trade-flow and liquidation aggregator, drained by the decision loop.
    flow: object = None
    # Last reconciliation report, produced on a timer by the decision loop.
    reconciliation: dict = None

    @property
    def account(self):
        return self.session.account

    def feature_source(self):
        """The feature builder the decision layer is serving from.

        Exposed here so a component that changes an input -- the mark pump writing a new
        funding publication -- can invalidate the cached series without reaching into the
        decision object's internals.
        """
        source = getattr(self.decisions, 'feature_source', None)
        if source is None:
            from .features.feature_source import FeatureSource

            source = FeatureSource(
                self.store,
                max_basis_age_ms=getattr(self.settings, 'max_basis_age_ms', None))
            self.decisions.feature_source = source
        return source

    @property
    def last_error(self):
        """Most recent failure across all components, for the dashboard payload."""
        return self.errors.latest

    @classmethod
    def build(cls, settings: Settings, api, store, session, risk, broker, models, decisions,
              limits=None):
        runtime = cls(settings=settings, store=store, api=api, session=session, risk=risk,
                      broker=broker, models=models, decisions=decisions)
        # A caller that has already resolved the active risk profile passes the matching
        # caps in. Otherwise the limits come from .env, which is a second, silently
        # different answer to the same question the risk engine just answered.
        runtime.limits = limits if limits is not None else portfolio_limits(settings)
        runtime.guard = MarketGuard(max_age_ms=settings.market_guard_max_age_ms)
        runtime.decision_state.processed_bars = store.get_runtime('processed_bars', {})
        runtime.decision_state.per_symbol = store.get_runtime('decision_state', {})
        runtime.decision_state.stored_trades = len(session.account.trades)
        runtime.model_status = cls._load_model_status()
        return runtime

    def restore_orders(self):
        """Reload persisted orders so resting orders survive a restart."""
        from .core.domain import PaperOrder

        for saved in self.store.orders():
            try:
                self.broker.track_order(PaperOrder(**saved))
                if order_state.is_open(saved.get('status')) and saved.get('plan'):
                    self.pending_plans[saved['order_id']] = saved['plan']
            except (TypeError, KeyError):
                continue

    @staticmethod
    def _load_model_status():
        import json

        path = Path(__file__).resolve().parent.parent / 'data' / 'model_baseline.json'
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'status': 'invalid'}
