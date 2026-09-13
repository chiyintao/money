from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

def _float(name, default):
    return float(os.getenv(name, default))

def _flag(name, default='1'):
    return os.getenv(name, default) not in ('0','false','False','no','off')

# Bar length in milliseconds, the unit every "bars held" and "N-bar horizon" calculation
# must use. It lives here rather than in ingest.py because Settings needs it, and
# ingest.py already imports Settings -- putting it in ingest would be a cycle.
#
# This field exists because its absence was silent: decision_loop read
# getattr(settings, 'interval_ms', 300000), which always fell back to 300000, so a 1m
# session timed its exits as if a bar were five minutes long. The default in .env
# happens to be 5m, which is exactly why nobody noticed.
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
DEFAULT_INTERVAL_MS = 60_000


def interval_to_ms(interval):
    """Milliseconds per bar for a Binance interval name.

    Unknown values fall back to the 1m default instead of raising: a typo in INTERVAL
    should not stop the service, and every consumer needs a usable number.
    """
    return INTERVAL_MS.get(str(interval or '').strip().lower(), DEFAULT_INTERVAL_MS)

@dataclass(frozen=True)
class Settings:
    mode: str = os.getenv("TRADING_MODE", "paper")
    base_url: str = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
    ws_url: str = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com")
    # Proxy for the market-data websocket only; empty means a direct connection.
    # It is separate from any REST proxy because the two hosts can be reachable by
    # different routes, and because aiohttp reads proxies from the environment -- a
    # Windows system proxy is not in the environment, so it has to be named here.
    ws_proxy: str = os.getenv("BINANCE_WS_PROXY", "").strip()
    symbols: tuple[str, ...] = tuple(x.strip().upper() for x in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if x.strip())
    interval: str = os.getenv("INTERVAL", "1m")
    interval_ms: int = interval_to_ms(os.getenv("INTERVAL", "1m"))
    lookback: int = int(os.getenv("LOOKBACK", "250"))
    poll_seconds: float = float(os.getenv("POLL_SECONDS", "0.5"))
    market_refresh_seconds: float = float(os.getenv("MARKET_REFRESH_SECONDS", "5"))
    strategy_min_interval_ms: int = int(os.getenv("STRATEGY_MIN_INTERVAL_MS", "100"))
    # Longest a symbol may go without an audit row while its decision is unchanged.
    # The dashboard reads the live cache, so this only bounds the audit trail's growth.
    strategy_audit_interval_ms: int = int(os.getenv("STRATEGY_AUDIT_INTERVAL_MS", "60000"))
    tick_strategy_enabled: bool = os.getenv("TICK_STRATEGY_ENABLED", "1") not in ('0','false','False')
    market_event_audit_ms: int = int(os.getenv("MARKET_EVENT_AUDIT_MS", "1000"))
    # Audit retention, by ROW COUNT rather than age. The events table held 1.07M rows
    # covering only 0.92 days, dominated by 758k strategy_decision rows that collapsed to
    # 32 distinct decisions per 5000 -- a 156x redundancy. An age window therefore cannot
    # bound it: the volume scales with tick rate, not with time. The hot table keeps the
    # newest N rows and everything older moves to events_archive.
    event_retention_rows: int = int(os.getenv("EVENT_RETENTION_ROWS", "50000"))
    # Types kept in the hot table regardless of age, because a session's outcome is
    # explained from them and they number in the hundreds.
    event_retention_durable: bool = os.getenv("EVENT_RETENTION_DURABLE", "1") not in ("0", "false", "")
    # How often the retention sweep runs; 0 disables it.
    event_retention_interval_seconds: float = _float("EVENT_RETENTION_INTERVAL_SECONDS", 900)
    starting_equity: float = _float("STARTING_EQUITY", 10000)
    # Sizing. max_risk_per_trade is what one stopped-out entry may cost and
    # max_portfolio_risk is what every open position together may cost. target_exposure
    # is the notional one entry aims for as a multiple of equity, so a wide stop no
    # longer shrinks a position to nothing while the balance sits idle.
    max_risk_per_trade: float = _float("MAX_RISK_PER_TRADE", .005)
    max_portfolio_risk: float = _float("MAX_PORTFOLIO_RISK", .02)
    target_exposure: float = _float("TARGET_EXPOSURE", 1.0)
    # The edge, in bps, at which a signal earns the full per-trade risk budget. Below it
    # the budget scales linearly to zero at break-even, so a marginal signal that barely
    # clears the cost floor no longer trades the same size as the best one. Set to 0 to
    # size every accepted signal identically, which is what the service did before.
    sizing_reference_edge_bps: float = _float("SIZING_REFERENCE_EDGE_BPS", 25.0)
    max_daily_loss: float = _float("MAX_DAILY_LOSS", .20)
    # Which risk profile the service starts with. A profile sets every sizing parameter
    # below it; naming one here means a restart no longer silently reverts to whatever
    # the .env happened to hold. Overridable per session through the API.
    risk_profile: str = os.getenv("RISK_PROFILE", "balanced")
    # Which exit policy the service starts with: stop width, reward:risk, trailing and
    # the time stop. Also switchable at runtime through /api/exit/policy.
    exit_policy: str = os.getenv("EXIT_POLICY", "balanced")
    # Portfolio caps. Both are multiples of equity: as absolute amounts a single number
    # could not mean the same thing on a 100 account and a 10000 one, and they also
    # contradicted the leverage chosen for the session.
    max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
    max_symbol_leverage: float = _float("MAX_SYMBOL_LEVERAGE", 3.0)
    max_gross_leverage: float = _float("MAX_GROSS_LEVERAGE", 5.0)
    # Execution hygiene. A plan is computed from one closed bar, so it describes a
    # price that existed for a moment. An order that never expires fills hours later at
    # whatever the market happens to be and still carries the stop and target from the
    # original bar, which is how a target ends up behind the entry price.
    order_ttl_ms: int = int(os.getenv("ORDER_TTL_MS", "90000"))
    # How the entry meets the book. Every entry used to be a market order, so every entry
    # paid the taker rate plus the half spread -- 12 bp round trip against a model whose
    # measured edge is a few basis points, which is why nothing ever cleared the gate. A
    # resting limit pays the maker rate and no spread, but it only fills when the market
    # trades to it, so it is a choice with a cost on both sides and has to be explicit.
    # 'market' preserves the previous behaviour exactly.
    entry_order_type: str = os.getenv("ENTRY_ORDER_TYPE", "market").strip().lower()
    # Where the passive entry rests. 0 joins the touch; a positive value improves on it by
    # this many basis points, which raises the maker fee but lowers the fill rate.
    entry_passive_offset_bps: float = _float("ENTRY_PASSIVE_OFFSET_BPS", 0.0)
    # Abandon an entry once the market has travelled this fraction of the way to the
    # stop (adverse) or to the target (in our favour). Both are measured against their
    # own bracket so one threshold scales across a 4%-ATR altcoin and BTC alike. The
    # favourable bound exists because a plan's edge belongs to the bar it was computed
    # on: once the move has largely happened, entering is chasing it.
    max_entry_adverse_fraction: float = _float("MAX_ENTRY_ADVERSE_FRACTION", .5)
    max_entry_chase_fraction: float = _float("MAX_ENTRY_CHASE_FRACTION", .35)
    entry_cooldown_minutes: float = _float("ENTRY_COOLDOWN_MINUTES", 5)
    # How long a rejected entry is remembered for its symbol. The tick path re-evaluates
    # on every book event, so a symbol refused for a structural reason (max_positions,
    # minimum notional, no free margin) used to be retried at 10 Hz, writing a
    # risk_decision plus an order_rejected pair each time -- roughly 100 rows/second
    # across a handful of symbols, which alone could flush the whole retention budget.
    # 0 means "one bar", which is the natural re-evaluation period of the decision.
    entry_reject_retry_ms: int = int(os.getenv("ENTRY_REJECT_RETRY_MS", "0"))
    # What to do when a modelled feature could not be filled from its real source and a
    # placeholder was used instead. "block" (the default) refuses the trade: the model
    # would otherwise be asked to extrapolate from a constant it was never fit on, and the
    # out-of-range gate cannot see this because the placeholder sits inside the training
    # bounds. "warn" keeps trading and records the reason code.
    feature_degraded_policy: str = os.getenv("FEATURE_DEGRADED_POLICY", "block")
    # Drawdown from the high-water mark that stops the account until a human resets it.
    # 0 disables it; the daily breaker remains either way, but it cannot express a
    # cumulative loss because it resets at the date boundary.
    max_drawdown_from_peak: float = _float("MAX_DRAWDOWN_FROM_PEAK", 0.25)
    # Consecutive losing closes and the fraction of the risk budget kept while they last.
    loss_streak_limit: int = int(os.getenv("LOSS_STREAK_LIMIT", "4"))
    loss_streak_scale: float = _float("LOSS_STREAK_SCALE", 0.5)
    # The matching cost model. It was hardcoded in the broker constructor, so the live
    # paper account always traded the defaults and no run could be explained by the costs
    # it assumed. maker_fee_rate is the venue standard-tier maker rate against a 0.0400%
    # taker rate; impact_bps is the market-impact coefficient at full participation, and 0
    # leaves the price size-independent, which is the previous behaviour.
    taker_fee_rate: float = _float("TAKER_FEE_RATE", 0.0004)
    maker_fee_rate: float = _float("MAKER_FEE_RATE", 0.0002)
    spread_bps: float = _float("SPREAD_BPS", 2.0)
    impact_bps: float = _float("IMPACT_BPS", 0.0)
    # Order round trip to the venue, in milliseconds, and how much of the market's move
    # during it goes against the fill. Zero latency fills at the signal price, which is
    # the assumption that makes a backtest look best and reality look worst; the default
    # is the honest one for a retail HTTP client, and setting it to 0 restores the old
    # behaviour exactly.
    entry_latency_ms: float = _float("ENTRY_LATENCY_MS", 350.0)
    latency_adverse_share: float = _float("LATENCY_ADVERSE_SHARE", 1.0)
    # Entries per UTC day, account-wide. A backstop against a runaway loop rather than a
    # strategy constraint: every other cap in the risk engine bounds how much one trade may
    # lose, and none of them bounds how many times a broken entry gate may fire. The gate
    # has produced 37 round trips in under ten seconds on a single symbol. Set generously
    # and lower it only with evidence; 0 disables the cap.
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "60"))
    # Volatility targeting: the per-bar return standard deviation the risk budget is sized
    # for, and the leverage cap on notional held in correlated symbols as a multiple of
    # equity. Both are off at 0, which is the previous behaviour.
    target_bar_volatility: float = _float("TARGET_BAR_VOLATILITY", 0.0)
    max_correlated_leverage: float = _float("MAX_CORRELATED_LEVERAGE", 0.0)
    correlation_threshold: float = _float("CORRELATION_THRESHOLD", 0.6)
    correlation_window_bars: int = int(os.getenv("CORRELATION_WINDOW_BARS", "120"))
    # Open interest, long/short ratios and taker volume ratios. The venue keeps these for
    # thirty days and archives nothing, so collection is not optional: what is not fetched
    # today cannot be fetched later at any price.
    # The archive had no retention and no reader, so it grew without bound while the rows
    # in it were unreachable. Default is ten times the hot window: large enough to answer
    # an audit question about a past session, small enough to stay a bounded cost.
    event_archive_rows: int = int(os.getenv("EVENT_ARCHIVE_ROWS", "1000000"))
    # Startup venue consistency check. "warn" records the finding and starts anyway;
    # "block" refuses to start on a suspect venue; "off" skips it.
    # Feature drift monitoring. The detector existed as thirteen unreferenced lines; this
    # is what actually runs it against the live feature stream.
    # Trade-flow and liquidation aggregation. The socket carries both; before this they
    # were used to trigger a tick and then discarded.
    # Periodic reconciliation between the account, the order book and the ledger.
    reconcile_enabled: bool = _flag("RECONCILE_ENABLED", "1")
    reconcile_interval_seconds: float = _float("RECONCILE_INTERVAL_SECONDS", 300.0)
    flow_collect_enabled: bool = _flag("FLOW_COLLECT_ENABLED", "1")
    flow_archive_rows: int = int(os.getenv("FLOW_ARCHIVE_ROWS", "2000000"))
    flow_drain_seconds: float = _float("FLOW_DRAIN_SECONDS", 30.0)
    drift_check_enabled: bool = _flag("DRIFT_CHECK_ENABLED", "1")
    drift_window: int = int(os.getenv("DRIFT_WINDOW", "2000"))
    drift_interval_seconds: float = _float("DRIFT_INTERVAL_SECONDS", 3600.0)
    drift_psi_threshold: float = _float("DRIFT_PSI_THRESHOLD", 0.2)
    drift_policy: str = os.getenv("DRIFT_POLICY", "warn")
    # Refuse entries unless a promoted production model is loaded, and refuse once it is
    # older than MODEL_MAX_AGE_HOURS. Both default to permissive so an operator opts in;
    # the alternative -- silently trading on weights that failed the promotion gate -- is
    # what the service did for its entire history.
    model_require_promoted: bool = _flag("MODEL_REQUIRE_PROMOTED", "0")
    model_max_age_hours: float = _float("MODEL_MAX_AGE_HOURS", 0.0)
    # Whether unvetted candidate weights may stand in for a promoted model that is present
    # but unusable (expired, unreadable, or missing its calibrator). Off by default: the
    # fallback exists for an empty registry, and applying it to a stale one silently
    # replaces an approved model with the weights the promotion gate refused. The service
    # then refuses to trade, which is the point.
    model_allow_candidate_fallback: bool = _flag("MODEL_ALLOW_CANDIDATE_FALLBACK", "0")
    # Supervised retraining. Off by default: training is expensive and never auto-promoted.
    retrain_enabled: bool = _flag("RETRAIN_ENABLED", "0")
    retrain_interval_seconds: float = _float("RETRAIN_INTERVAL_SECONDS", 21600.0)
    venue_check_policy: str = os.getenv("VENUE_CHECK_POLICY", "warn")
    venue_check_seconds: float = _float("VENUE_CHECK_SECONDS", 6.0)
    derivatives_collect_enabled: bool = _flag("DERIVATIVES_COLLECT_ENABLED", "1")
    derivatives_collect_period: str = os.getenv("DERIVATIVES_COLLECT_PERIOD", "5m")
    derivatives_collect_interval_seconds: float = _float("DERIVATIVES_COLLECT_INTERVAL_SECONDS", 300.0)
    derivatives_collect_symbols: int = int(os.getenv("DERIVATIVES_COLLECT_SYMBOLS", "12"))
    # How quiet a symbol's live feed may get before entries are refused.
    market_guard_max_age_ms: int = int(os.getenv("MARKET_GUARD_MAX_AGE_MS", "120000"))
    # How old a mark price may be before mark_basis is treated as unfillable. Funding
    # settles every eight hours and the mark price needed for a basis is collected every
    # five minutes; this bound sits between the two cadences.
    max_basis_age_ms: int = int(os.getenv("MAX_BASIS_AGE_MS", "900000"))
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_device: str = os.getenv("MODEL_DEVICE", "cpu")
    model_chronos_enabled: bool = _flag("MODEL_CHRONOS_ENABLED", "1")
    model_chronos_horizon: int = int(os.getenv("MODEL_CHRONOS_HORIZON", "15"))
    model_max_chronos_symbols: int = int(os.getenv("MODEL_MAX_CHRONOS_SYMBOLS", "8"))
    model_chronos_timeout_ms: int = int(os.getenv("MODEL_CHRONOS_TIMEOUT_MS", "3000"))
    model_min_edge_bps: float = _float("MODEL_MIN_EDGE_BPS", 0.5)
    model_min_agreement: float = _float("MODEL_MIN_AGREEMENT", 0.6)
    # How many times the round-trip cost the target must clear before a signal is allowed.
    # At 1.0 a plan qualified for merely exceeding the cost of trading it, which on a
    # near-coin-flip signal admits almost everything. Measured edge over 5-15 minute
    # horizons was comparable to the 12bps round trip, so the bar has to be higher.
    min_edge_multiple: float = _float("MIN_EDGE_MULTIPLE", 2.5)
    # Per-symbol meta-labeling gate. See app/symbol_edge.py. The gate decides whether a
    # symbol's own cost-adjusted record justifies acting on the model's proposal for it.
    symbol_edge_enabled: bool = _flag("SYMBOL_EDGE_ENABLED", "1")
    symbol_edge_horizon_bars: int = int(os.getenv("SYMBOL_EDGE_HORIZON_BARS", "3"))
    symbol_edge_window: int = int(os.getenv("SYMBOL_EDGE_WINDOW", "200"))
    symbol_edge_min_samples: int = int(os.getenv("SYMBOL_EDGE_MIN_SAMPLES", "30"))
    symbol_edge_min_net_bps: float = _float("SYMBOL_EDGE_MIN_NET_BPS", 0.0)
    # Standard errors the per-symbol mean net edge must clear. 0 disables the requirement,
    # which is what the gate effectively did before: it tested the sign of a 30-sample mean.
    symbol_edge_min_t: float = _float("SYMBOL_EDGE_MIN_T", 2.0)
    # Benjamini-Hochberg false discovery rate across symbols. 0 disables the multiplicity
    # correction but leaves the significance requirement in place.
    symbol_edge_fdr: float = _float("SYMBOL_EDGE_FDR", 0.0)
    # Deny until proven. Safe because observations are recorded whether or not a symbol is
    # tradable, so a blocked symbol still accumulates the evidence to earn its way back.
    symbol_edge_require_samples: bool = _flag("SYMBOL_EDGE_REQUIRE_SAMPLES", "1")
    # Align the exit policy's time stop with the horizon where the model's edge is actually
    # measured to persist. The audit found edge peaking around 30 minutes and turning
    # negative after that, while the target geometry aimed at multi-hour holds: a
    # reward:risk ratio is meaningless if the edge it waits for has already decayed.
    exit_align_to_measured_horizon: bool = _flag("EXIT_ALIGN_TO_MEASURED_HORIZON", "1")
    # Slots reserved for symbols without enough evidence yet. Signals are only computed for
    # symbols the session holds, so a selector that took only proven symbols would never
    # observe a new one and the gate would freeze on what it happened to learn first.
    symbol_edge_explore_slots: int = int(os.getenv("SYMBOL_EDGE_EXPLORE_SLOTS", "1"))
    model_rule_fallback: bool = _flag("MODEL_RULE_FALLBACK", "0")
    model_ood_policy: str = os.getenv("MODEL_OOD_POLICY", "block")
