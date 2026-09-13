"""The backtest must describe the service, not its own configuration.

The offline runners used to build a RiskEngine with hardcoded values -- including a 2%
daily loss against the service's 20% -- and to fill entries without the re-anchoring
that settlement applies. A result from them could not predict the paper account.
"""
import pytest

from app.backtest.backtest import fill_entry, live_limits, live_risk, run_backtest
from app.trading.broker import PaperBroker
from app.core.config import Settings
from app.trading.guards import PortfolioLimits
from app.trading.risk import RiskEngine
from app.trading.simulation import PaperAccount


def always_long(symbol, history):
    """A deterministic signal, so the sizing under test is the only variable."""
    price = float(history[-1]['close'])
    return {'symbol': symbol, 'side': 'LONG', 'entry': price, 'stop': price * .99,
            'take_profit': price * 1.02, 'features': {'price': price}, 'confidence': 1.0,
            'reason_codes': ['test'], 'scores': {}}


def rising_rows(count=120, offset=0.0):
    rows = []
    for i in range(count):
        close = 100 + offset + i * .2
        rows.append({'open_time': i, 'close_time': i + 1, 'open': close, 'high': close + 1,
                     'low': close - 1, 'close': close, 'volume': 10, 'is_closed': 1})
    return rows


def test_backtest_sizing_matches_the_live_settings():
    settings = Settings()
    risk = live_risk(10000, settings)
    assert risk.max_risk == settings.max_risk_per_trade
    assert risk.max_daily_loss == settings.max_daily_loss
    assert risk.target_exposure == settings.target_exposure
    assert risk.max_portfolio_risk == settings.max_portfolio_risk
    assert risk.max_symbol_leverage == settings.max_symbol_leverage
    assert risk.max_gross_leverage == settings.max_gross_leverage


def test_backtest_limits_match_the_live_settings():
    settings = Settings()
    limits = live_limits(settings)
    assert limits.max_positions == settings.max_positions
    assert limits.max_gross_leverage == settings.max_gross_leverage


def test_fill_entry_reanchors_the_stop_to_the_traded_price():
    account = PaperAccount(10000, .0004, 0.0, initial_cash=10000)
    broker = PaperBroker(fee_rate=.0004, slippage_bps=0)
    plan_row = {'symbol': 'X', 'side': 'LONG', 'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0,
                'stop_distance': 10.0, 'target_distance': 20.0}

    fill = fill_entry(account, broker, live_risk(10000), live_limits(), 'X', plan_row, 102.0, 1000)

    assert fill is not None
    position = account.positions['X']
    assert position.entry == pytest.approx(102.0)
    assert position.stop == pytest.approx(92.0)
    assert position.target == pytest.approx(122.0)


def test_fill_entry_refuses_when_the_caps_leave_no_room():
    account = PaperAccount(10000, .0004, 0.0, initial_cash=10000)
    broker = PaperBroker(slippage_bps=0)
    plan_row = {'symbol': 'X', 'side': 'LONG', 'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0}
    risk = RiskEngine(max_risk=.005, max_gross_leverage=0.0, day_start_equity=10000, max_symbol_leverage=0.0)

    assert fill_entry(account, broker, risk, live_limits(), 'X', plan_row, 100.0, 1000) is None
    assert account.positions == {}


def test_run_backtest_stops_trading_when_the_caps_bind():
    rows = rising_rows()
    risk = RiskEngine(max_risk=.005, max_gross_leverage=0.0, day_start_equity=10000, max_symbol_leverage=0.0)

    result = run_backtest(rows, strategy=always_long, risk=risk,
                          limits=PortfolioLimits(max_symbol_leverage=0.0, max_gross_leverage=0.0))

    assert result['trades'] == []


def test_the_same_signal_does_trade_once_the_caps_allow_it():
    # The control for the test above: with the same signal and the live caps, the entries
    # are taken, so it is the caps that stop the trading rather than a missing signal.
    rows = rising_rows()
    result = run_backtest(rows, strategy=always_long, risk=live_risk(10000), limits=live_limits())
    assert result['metrics']['trades'] > 0


def test_backtest_deploys_a_meaningful_share_of_equity():
    # The behaviour the sizing model exists to produce: an entry that commits real
    # capital instead of the ~14% of equity the old risk-only formula allowed.
    rows = rising_rows()
    result = run_backtest(rows, strategy=always_long, risk=live_risk(10000), limits=live_limits())
    notional = max(trade['entry'] * trade['qty'] for trade in result['trades'])
    assert notional > 10000 * .25
