"""Regressions for the fidelity round: units, funding settlement, error attribution.

Every test here pins a defect that was invisible in production because the wrong value
happened to look plausible. They are written against the failure mode rather than the
implementation, so a future refactor that reintroduces the bug fails the test.
"""
import datetime
import time

import pytest

from app.core.config import Settings, interval_to_ms
from app.trading.guards import PortfolioLimits
from app.trading.risk_profiles import PRESETS
from app.runtime_context import ErrorLog
from app.trading.simulation import PaperAccount, Position
from app.backtest.simulation_session import SimulationSession


# --------------------------------------------------------------- bar length units
def test_settings_expose_the_bar_length_they_were_configured_with():
    # decision_loop read getattr(settings, 'interval_ms', 300000). The attribute did not
    # exist, so a 1m session sized every 'bars held' calculation as if a bar were five
    # minutes: the measured-horizon time stop was five times too long.
    settings = Settings()
    assert settings.interval_ms == interval_to_ms(settings.interval)


def test_interval_to_ms_covers_every_supported_bar_and_defaults_safely():
    assert interval_to_ms('1m') == 60_000
    assert interval_to_ms('5m') == 300_000
    assert interval_to_ms('1h') == 3_600_000
    # A typo must not stop the service, and must not silently become five minutes either.
    assert interval_to_ms('nonsense') == 60_000
    assert interval_to_ms(None) == 60_000


def test_ingest_and_config_share_one_interval_table():
    from app.market import ingest

    assert ingest.INTERVAL_MS is not None
    assert ingest.interval_ms('4h') == 14_400_000


# ------------------------------------------------------------------ funding settle
def _account_with_position(symbol='BTCUSDT'):
    account = PaperAccount(cash=10_000.0)
    account.positions[symbol] = Position(symbol, 'LONG', 1.0, 100.0, 90.0, 110.0)
    account.marks[symbol] = 100.0
    return account


def test_funding_charges_once_per_window():
    account = _account_with_position()
    now = 1_700_000_000_000
    first = account.apply_funding({'BTCUSDT': 0.0001}, now)
    second = account.apply_funding({'BTCUSDT': 0.0001}, now + 1000)
    assert first == pytest.approx(0.01)
    assert second == 0.0


def test_a_missing_rate_map_does_not_consume_the_window():
    # The window key used to be advanced before the loop, so one failed market snapshot
    # consumed the settlement and its cost was never charged at all.
    account = _account_with_position()
    now = 1_700_000_000_000
    assert account.apply_funding({}, now) == 0.0
    assert account.funding_missed == 1
    charged = account.apply_funding({'BTCUSDT': 0.0001}, now + 1000)
    assert charged == pytest.approx(0.01)
    assert account.funding_settlements[-1]['symbol'] == 'BTCUSDT'


def test_funding_uses_a_separate_window_per_symbol():
    # 8h funding is the norm but the venue runs 4h and 1h on some contracts; one
    # account-wide window charged those half as often as the real thing.
    account = PaperAccount(cash=10_000.0)
    account.positions['BTCUSDT'] = Position('BTCUSDT', 'LONG', 1.0, 100.0, 90.0, 110.0)
    account.positions['XYZUSDT'] = Position('XYZUSDT', 'LONG', 1.0, 100.0, 90.0, 110.0)
    account.marks.update({'BTCUSDT': 100.0, 'XYZUSDT': 100.0})
    base = 1_700_000_000_000
    hour = 3_600_000
    # Align to the top of an eight-hour window so the arithmetic below is exact.
    base -= base % (8 * hour)
    windows = {'BTCUSDT': 8 * hour, 'XYZUSDT': 1 * hour}
    paid = account.apply_funding({'BTCUSDT': 0.0001, 'XYZUSDT': 0.0001}, base, interval_ms=windows)
    assert paid == pytest.approx(0.02)
    # One hour later only the hourly contract settles again.
    account.apply_funding({'BTCUSDT': 0.0001, 'XYZUSDT': 0.0001}, base + hour, interval_ms=windows)
    assert len(account.funding_settlements) == 3
    assert [s['symbol'] for s in account.funding_settlements] == ['BTCUSDT', 'XYZUSDT', 'XYZUSDT']


def test_a_symbol_without_a_published_rate_is_reported_not_charged_zero():
    account = PaperAccount(cash=10_000.0)
    account.positions['AUSDT'] = Position('AUSDT', 'LONG', 1.0, 100.0, 90.0, 110.0)
    account.positions['BUSDT'] = Position('BUSDT', 'LONG', 1.0, 100.0, 90.0, 110.0)
    account.marks.update({'AUSDT': 100.0, 'BUSDT': 100.0})
    account.apply_funding({'AUSDT': 0.0001}, 1_700_000_000_000)
    assert account.funding_missing == ['BUSDT']
    # The window stays open for the missing symbol, so it is charged when the rate
    # finally arrives rather than being written off as a zero-cost settlement.
    account.apply_funding({'AUSDT': 0.0001, 'BUSDT': 0.0001}, 1_700_000_001_000)
    assert account.funding_settlements[-1]['symbol'] == 'BUSDT'


def test_funding_state_survives_a_snapshot_round_trip():
    account = _account_with_position()
    account.apply_funding({'BTCUSDT': 0.0001}, 1_700_000_000_000)
    restored = PaperAccount.restore(account.snapshot())
    assert restored.funding_buckets == account.funding_buckets
    assert restored.apply_funding({'BTCUSDT': 0.0001}, 1_700_000_001_000) == 0.0


# ------------------------------------------------------------- error attribution
def test_one_component_cannot_clear_another_components_failure():
    # runtime.last_error was reset at the top of every decision-loop iteration, so a
    # permanently broken mark-price pump showed a clean dashboard whenever the bar loop
    # happened to succeed.
    errors = ErrorLog()
    errors.note('mark_pump', 'timeout')
    errors.ok('decision_loop')
    assert errors.latest is not None and 'mark_pump' in errors.latest
    errors.note('decision_loop', 'boom')
    errors.ok('mark_pump')
    assert 'decision_loop' in errors.latest


def test_error_log_counts_repeats_and_reports_age():
    errors = ErrorLog()
    errors.note('retention', 'database_worker_not_attached', now_ms=1000)
    errors.note('retention', 'database_worker_not_attached', now_ms=5000)
    snapshot = errors.snapshot(now_ms=9000)
    entry = snapshot['entries'][0]
    assert entry['count'] == 2
    assert entry['first_at'] == 1000
    assert entry['age_ms'] == 4000


def test_error_log_is_empty_when_nothing_has_failed():
    errors = ErrorLog()
    assert errors.latest is None
    assert errors.snapshot()['count'] == 0
    errors.note('x', 'y')
    errors.ok('x')
    assert errors.latest is None


# --------------------------------------------------------- portfolio caps coherent
def test_portfolio_limits_from_a_profile_match_the_profile():
    # Startup applied the persisted profile to the risk engine but rebuilt the portfolio
    # caps from .env, so the panel and the order path disagreed about the same session.
    for name, profile in PRESETS.items():
        limits = PortfolioLimits.from_profile(profile)
        assert limits.max_positions == profile.max_positions
        assert limits.max_gross_leverage == profile.max_gross_leverage
        assert limits.max_symbol_leverage <= limits.max_gross_leverage


def test_runtime_build_accepts_limits_that_override_settings():
    from app.runtime_context import Runtime

    profile = PRESETS['conservative']
    limits = PortfolioLimits.from_profile(profile)
    assert limits.max_gross_leverage == 3.0
    # The default path still derives them from Settings for callers that do not care.
    assert 'limits' in Runtime.build.__code__.co_varnames


# ------------------------------------------------------------- session selection
class _Tier:
    def __init__(self, ranks):
        self.ranks = ranks

    def __call__(self, symbol):
        return self.ranks.get(symbol, 1)


def test_proven_losing_symbols_only_fill_slots_after_unproven_ones():
    # Rejected symbols are a last resort, not a first choice. This pins the ordering the
    # tiering was added for: an unproven symbol is worth a slot ahead of one already
    # measured as a loser, and a loser is still taken when there is nothing else at all
    # (an empty slot yields no signals, and signals are what the gate learns from).
    session = SimulationSession.__new__(SimulationSession)
    session.source = 'gainers'
    session.symbol_count = 4
    rows = [{'symbol': s, 'price': 1.0} for s in ('AUSDT', 'BUSDT', 'CUSDT', 'DUSDT')]
    tier = _Tier({'AUSDT': 0, 'BUSDT': 1, 'CUSDT': 0, 'DUSDT': 0})
    assert session._rank(rows, 2, tier, explore_slots=0) == ['BUSDT', 'AUSDT']
    assert session._rank(rows, 2, _Tier({}), explore_slots=0) == ['AUSDT', 'BUSDT']


def test_unproven_symbols_still_get_the_exploration_slots():
    session = SimulationSession.__new__(SimulationSession)
    session.source = 'gainers'
    session.symbol_count = 4
    rows = [{'symbol': s, 'price': 1.0} for s in ('AUSDT', 'BUSDT', 'CUSDT', 'DUSDT')]
    tier = _Tier({'AUSDT': 2, 'BUSDT': 1})
    picked = session._rank(rows, 4, tier, explore_slots=1)
    assert picked[0] == 'AUSDT'
    assert 'BUSDT' in picked
