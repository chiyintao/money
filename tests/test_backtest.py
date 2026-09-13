from app.backtest.backtest import run_backtest, run_portfolio_backtest


def always_long(symbol, history):
    """A deterministic signal, so these tests exercise sizing and fills rather than the
    rule baseline.

    The synthetic series below closes monotonically higher, which pins RSI at 100 and
    the volume ratio at 1.0, so the rule baseline returns FLAT on every bar. The trade
    assertions here used to pass vacuously against zero trades.
    """
    price = float(history[-1]['close'])
    return {'symbol': symbol, 'side': 'LONG', 'entry': price, 'stop': price * .99,
            'take_profit': price * 1.02, 'features': {'price': price}, 'confidence': 1.0,
            'reason_codes': ['test'], 'scores': {}}


def make_rows(offset=0.0):
    return [{'open_time': i, 'close_time': i + 1, 'open': 100 + offset + i * .2,
             'high': 101 + offset + i * .2, 'low': 99 + offset + i * .2,
             'close': 100 + offset + i * .2, 'volume': 10, 'is_closed': 1} for i in range(120)]


def test_portfolio_backtest_shares_one_account():
    result = run_portfolio_backtest({'A': make_rows(0), 'B': make_rows(20)}, strategy=always_long)
    assert result['symbols'] == ['A', 'B']
    assert result['metrics']['start_equity'] == 10000
    assert result['metrics']['trades'] > 0
    assert all('pnl' in trade for trade in result['trades'])


def test_portfolio_backtest_actually_trades_every_symbol():
    result = run_portfolio_backtest({'A': make_rows(0), 'B': make_rows(20)}, strategy=always_long)
    assert {trade['symbol'] for trade in result['trades']} == {'A', 'B'}


def test_backtest_is_deterministic():
    rows = make_rows()
    a = run_backtest(rows, strategy=always_long)
    b = run_backtest(rows, strategy=always_long)
    assert a['metrics']['trades'] > 0
    assert a['metrics'] == b['metrics']
    assert [t['entry'] for t in a['trades']] == [t['entry'] for t in b['trades']]


def test_backtest_rule_baseline_still_runs():
    # The rule baseline is the A/B control; it must keep working even when it declines
    # to trade the synthetic series.
    result = run_backtest(make_rows())
    assert 'metrics' in result and 'trades' in result
