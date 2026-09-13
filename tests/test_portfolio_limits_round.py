"""Portfolio-level limits: the risks that only exist between positions.

Every cap in the system was per-entry. The daily breaker reset at midnight; a basket of
correlated symbols was measured as if it were a basket of independent ones; a symbol whose
range tripled took three times the risk at the same stop distance. Each of these is
invisible to a per-trade check and each was reachable in the stored sessions.
"""
import math
import random

import pytest

from app.trading.concentration import concentration_report, correlated_notional, correlation, correlation_matrix, realized_volatility, returns, volatility_scale
from app.trading.guards import PortfolioLimits
from app.trading.risk import RiskEngine


def engine(**kwargs):
    fields = {'max_risk': .005, 'max_daily_loss': .20, 'max_gross_leverage': 5.0,
              'day_start_equity': 10000}
    fields.update(kwargs)
    return RiskEngine(**fields)


def plan(entry=100.0, stop=95.0, side='LONG', take_profit=115.0):
    return {'side': side, 'entry': entry, 'stop': stop, 'take_profit': take_profit}


def _at(day_key):
    from datetime import datetime, timezone
    parsed = datetime.strptime(day_key, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _random_walk(count, sigma, seed=4):
    rng = random.Random(seed)
    price = 100.0
    out = [price]
    for _ in range(count):
        price *= math.exp(rng.gauss(0, sigma))
        out.append(price)
    return out


def _compound(series, rng, noise):
    price = 100.0
    out = [price]
    for step in series:
        price *= (1 + step + rng.gauss(0, noise))
        out.append(price)
    return out


# --------------------------------------------------------- drawdown breaker
def test_the_daily_breaker_alone_lets_an_account_bleed_one_day_at_a_time():
    # Twenty days each losing just under the daily limit. The breaker fires and resets
    # every day, and the account ends at a fraction of its capital with nothing having
    # refused: a daily limit bounds how fast a loss arrives, not how large it gets.
    risk = engine(max_daily_loss=.05, drawdown_limit=0.0)
    equity = 10000.0
    risk.reset_for_session(equity, day_key='2026-01-01')
    for day in range(1, 21):
        risk.approve(plan(), equity, event_time_ms=_at('2026-01-%02d' % min(day + 1, 28)))
        equity *= 0.96
        risk._roll_day(equity)
    assert equity < 4500, 'the fixture must actually destroy capital'
    assert risk.halted is False, 'the daily breaker has forgiven every day of it'


def test_the_high_water_mark_breaker_stops_the_account():
    risk = engine(drawdown_limit=.25)
    risk.reset_for_session(10000.0, day_key='2026-01-01')
    risk.observe_equity(12000.0)
    # 24.2% from the peak: uncomfortable, and still inside the limit.
    assert risk.approve(plan(), 9100.0)['approved'] is True
    result = risk.approve(plan(), 8900.0)
    assert result['approved'] is False
    assert result['reason'] == 'max_drawdown_circuit_breaker'
    assert result['scope'] == 'high_water_mark'
    assert result['drawdown'] == pytest.approx(1 - 8900 / 12000, rel=1e-6)


def test_a_drawdown_halt_survives_the_date_boundary():
    # The daily breaker is deliberately forgiving at midnight. A drawdown halt that also
    # forgave it would be a daily speed bump rather than a breaker.
    risk = engine(drawdown_limit=.2)
    risk.reset_for_session(10000.0, day_key='2026-01-01')
    risk.approve(plan(), 7000.0)
    assert risk.halted is True
    risk._roll_day(7000.0)
    assert risk.halted is True
    assert risk.approve(plan(), 7000.0)['reason'] == 'max_drawdown_circuit_breaker'


def test_clearing_the_flag_alone_is_not_enough_to_resume():
    # Releasing the halt without changing anything else re-trips on the next call, which is
    # the correct behaviour: the condition that stopped the account has not gone away. A
    # resume means accepting the new capital base, and that has to be explicit.
    risk = engine(drawdown_limit=.2)
    risk.reset_for_session(10000.0, day_key='2026-01-01')
    risk.approve(plan(), 7000.0)
    risk.halted = False
    risk.halt_scope = ''
    assert risk.approve(plan(), 7000.0)['reason'] == 'max_drawdown_circuit_breaker'

    # Rebasing the high-water mark is what a deliberate resume looks like.
    risk.peak_equity = 7000.0
    risk.halted = False
    risk.halt_scope = ''
    assert risk.approve(plan(), 7000.0)['approved'] is True


def test_the_breaker_is_off_when_the_limit_is_zero():
    risk = engine(drawdown_limit=0.0)
    risk.reset_for_session(10000.0, day_key='2026-01-01')
    assert risk.approve(plan(), 5000.0)['approved'] is True


def test_the_high_water_mark_round_trips_through_a_snapshot():
    risk = engine(drawdown_limit=.3, target_volatility=.01)
    risk.reset_for_session(10000.0)
    risk.observe_equity(15000.0)
    risk.record_close(-5)
    restored = RiskEngine.restore(risk.snapshot())
    assert restored.peak_equity == 15000.0
    assert restored.drawdown_limit == .3
    assert restored.loss_streak == 1
    assert restored.target_volatility == .01
    assert restored.snapshot() == risk.snapshot()


# ------------------------------------------------------------ losing streak
def test_a_losing_streak_shrinks_size_rather_than_stopping_trading():
    risk = engine(loss_streak_limit=4, loss_streak_scale=0.5)
    normal = risk.size(plan(), 10000.0)['quantity']
    for _ in range(4):
        risk.record_close(-1.0)
    assert risk.streak_scale() == 0.5
    assert risk.size(plan(), 10000.0)['quantity'] == pytest.approx(normal * 0.5, rel=1e-9)


def test_a_win_clears_the_streak():
    risk = engine(loss_streak_limit=3, loss_streak_scale=0.5)
    for _ in range(3):
        risk.record_close(-1.0)
    assert risk.streak_scale() == 0.5
    risk.record_close(2.0)
    assert risk.loss_streak == 0 and risk.streak_scale() == 1.0


def test_the_streak_counter_ignores_unusable_results():
    risk = engine(loss_streak_limit=3)
    risk.record_close(None)
    risk.record_close('nonsense')
    risk.record_close(float('nan'))
    assert risk.loss_streak == 0


# --------------------------------------------------------- volatility target
def test_volatility_scaling_shrinks_a_wild_symbol():
    closes = _random_walk(300, sigma=0.02)
    realized = realized_volatility(closes, 120)
    assert realized > 0.01
    assert volatility_scale(realized, 0.005) < 0.6


def test_volatility_scaling_never_levers_up_a_quiet_symbol():
    # Target volatility is a cap on risk, not a mandate to borrow: the scale is bounded at
    # one, which is the difference between targeting risk and writing a short-vol position.
    assert volatility_scale(0.0001, 0.01) == 1.0


def test_volatility_scaling_has_a_floor():
    assert volatility_scale(10.0, 0.005, floor=0.25) == 0.25


def test_volatility_scaling_leaves_an_unmeasurable_symbol_alone():
    assert volatility_scale(0.0, 0.01) == 1.0
    assert volatility_scale(float('nan'), 0.01) == 1.0
    assert volatility_scale(0.01, 0.0) == 1.0


def test_the_engine_applies_the_volatility_scale_to_the_order_it_returns():
    calm = engine(target_volatility=.01)
    wild = engine(target_volatility=.01)
    closes = _random_walk(300, sigma=0.03)
    small = wild.approve(plan(), 10000.0, realized_volatility=realized_volatility(closes, 120))
    big = calm.approve(plan(), 10000.0, realized_volatility=0.01)
    assert small['volatility_scale'] < 1.0
    assert big['volatility_scale'] == pytest.approx(1.0)
    assert small['quantity'] < big['quantity']


# -------------------------------------------------------------- correlation
def test_returns_skips_unusable_pairs():
    # A non-positive price on either side of a pair makes the ratio meaningless, and the
    # pair is dropped rather than contributing an infinity to the variance.
    assert returns([100, 50, 0, 110]) == [pytest.approx(-0.5)]
    assert returns([100, 110, 121]) == [pytest.approx(0.1), pytest.approx(0.1)]


def test_a_shared_factor_is_detected_as_correlation():
    rng = random.Random(9)
    factor = [rng.gauss(0, 0.01) for _ in range(200)]
    left = _compound(factor, rng, 0.001)
    right = _compound(factor, rng, 0.001)
    noise = _compound([rng.gauss(0, 0.002) for _ in range(200)], rng, 0.0)
    value = correlation(left, right)
    assert value is not None and value > 0.9
    assert abs(correlation(left, noise) or 0.0) < 0.5


def test_correlation_is_none_rather_than_zero_without_history():
    # Zero would mean "independent", which is the one thing an unmeasurable pair is not.
    assert correlation([1, 2, 3], [1, 2, 3]) is None
    assert correlation([1.0] * 100, [1.0] * 100) is None


def test_the_matrix_is_symmetric():
    rng = random.Random(2)
    series = {}
    for name in ('A', 'B', 'C'):
        price = 100.0
        rows = []
        for _ in range(60):
            price *= (1 + rng.gauss(0, .01))
            rows.append(price)
        series[name] = rows
    matrix = correlation_matrix(series)
    assert matrix[('A', 'B')] == matrix[('B', 'A')]
    assert len(matrix) == 6


def test_correlated_notional_ignores_uncorrelated_and_unmeasured_symbols():
    matrix = {('BTCUSDT', 'ETHUSDT'): 0.9, ('ETHUSDT', 'BTCUSDT'): 0.9,
              ('BTCUSDT', 'GOLDUSDT'): 0.05, ('GOLDUSDT', 'BTCUSDT'): 0.05}
    book = {'ETHUSDT': 1000.0, 'GOLDUSDT': 5000.0, 'NEWUSDT': 9000.0}
    # Only ETH is correlated. GOLD is measured and independent. NEW has no history, so it
    # is treated as unknown rather than as safe.
    assert correlated_notional('BTCUSDT', book, matrix) == 1000.0


def test_the_concentration_cap_refuses_a_second_bet_on_the_same_factor():
    limits = PortfolioLimits(max_positions=5, max_symbol_leverage=3.0,
                             max_gross_leverage=10.0, max_correlated_leverage=1.0)
    matrix = {('A', 'B'): 0.95, ('B', 'A'): 0.95}
    # 9800 already held in a symbol that moves with B, plus 500 more, against a cap of one
    # times equity for correlated exposure.
    book = {'A': 9800.0}
    ok, reason = limits.approve('B', 500.0, book, 10000.0, correlations=matrix)
    assert ok is False
    assert reason == 'max_correlated_notional'
    # And a genuinely independent symbol in the same book is still allowed.
    ok, reason = limits.approve('Z', 500.0, book, 10000.0, correlations=matrix)
    assert ok is True and reason == 'portfolio_ok'


def test_the_concentration_cap_is_off_unless_it_is_configured():
    limits = PortfolioLimits(max_positions=5, max_symbol_leverage=3.0, max_gross_leverage=10.0)
    matrix = {('A', 'B'): 0.99, ('B', 'A'): 0.99}
    ok, reason = limits.approve('B', 500.0, {'A': 9000.0}, 10000.0, correlations=matrix)
    assert ok is True and reason == 'portfolio_ok'


def test_a_profile_without_the_cap_does_not_reset_a_configured_one():
    class Profile:
        max_positions = 4
        max_symbol_leverage = 2.0
        max_gross_leverage = 4.0
    limits = PortfolioLimits(max_correlated_leverage=1.5)
    limits.apply_profile(Profile())
    assert limits.max_correlated_leverage == 1.5
    assert limits.max_positions == 4


def test_the_report_explains_the_concentration_it_measured():
    matrix = {('A', 'B'): 0.8, ('B', 'A'): 0.8}
    report = concentration_report('B', {'A': 6000.0, 'C': 4000.0}, matrix, 10000.0)
    assert report['correlated_notional'] == 6000.0
    assert report['correlated_share'] == pytest.approx(0.6)
    assert report['measured_pairs'] == 1


# ------------------------------------------------- working-order reservations
def test_a_resting_order_holds_budget_until_it_stops_working():
    # risk.approve and limits.approve ran once, at submit, against the book as it stood.
    # Several symbols could each be sized against an empty book, rest, and fill in turn,
    # taking the portfolio to a multiple of its risk budget with nothing recording it.
    from app.trading.execution import committed_exposure, release_entry, reserve_entry

    class Order:
        order_id = 'o1'
        symbol = 'A'
        side = 'BUY'
        quantity = 2.0

    runtime = type('R', (), {})()
    reserve_entry(runtime, Order(), plan(entry=100.0, stop=95.0))
    notional, risk_cash = committed_exposure(runtime)
    assert notional == {'A': 200.0}
    assert risk_cash == pytest.approx(10.0)
    release_entry(runtime, 'o1')
    assert committed_exposure(runtime) == ({}, 0.0)


def test_the_second_fill_is_refused_when_the_budget_is_already_spent():
    from app.trading.execution import fill_is_affordable, reserve_entry

    class Order:
        order_id = 'o1'
        symbol = 'A'
        side = 'BUY'
        quantity = 5.0

    class Fill:
        order_id = 'o2'
        symbol = 'B'
        side = 'BUY'
        quantity = 5.0
        price = 100.0

    class Account:
        positions = {}
        equity = 1000.0

    runtime = type('R', (), {})()
    runtime.account = Account()
    runtime.risk = engine(max_risk=.01, max_portfolio_risk=.02)
    runtime.limits = PortfolioLimits(max_positions=5, max_symbol_leverage=10.0,
                                     max_gross_leverage=20.0)
    runtime.reservations = {}
    # One working order already risks 25 USDT of a 20 USDT portfolio budget.
    reserve_entry(runtime, Order(), plan(entry=100.0, stop=95.0))
    target = dict(plan(entry=100.0, stop=95.0), order_id='o2')
    allowed, reason = fill_is_affordable(runtime, Fill(), target)
    assert allowed is False
    assert reason == 'portfolio_risk_budget_exceeded'


def test_a_fill_for_a_symbol_that_already_has_a_position_is_not_a_new_slot():
    from app.trading.execution import fill_is_affordable

    class Position:
        entry = 100.0
        qty = 1.0
        stop = 95.0

    class Fill:
        order_id = 'o1'
        symbol = 'A'
        side = 'BUY'
        quantity = 1.0
        price = 100.0

    class Account:
        equity = 10000.0
        positions = {'A': Position()}

    runtime = type('R', (), {})()
    runtime.account = Account()
    runtime.limits = PortfolioLimits(max_positions=1, max_symbol_leverage=10.0,
                                     max_gross_leverage=20.0)
    runtime.reservations = {}
    allowed, reason = fill_is_affordable(runtime, Fill(), dict(plan(), order_id='o1'))
    assert allowed is True and reason == 'ok'
