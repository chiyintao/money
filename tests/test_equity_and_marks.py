"""Two accounting facts that were wrong in ways nothing could observe.

1. The equity curve was sampled on every `mark()` call. `mark()` runs from a 2-second
   mark pump, from every feed event, and from the bar loop, so the median gap between
   consecutive samples was 1 millisecond while `metrics.summary` annualised with a
   bars-per-year factor. The dashboard reported 220% annualised volatility over the last
   180 points (41 minutes) and 45,495% over the full stored curve; both were arithmetic
   on a series that was not a time series. Only a bar boundary records a sample now.

2. Funding and liquidation were priced off `marks`, a single dict written by four
   different sources -- mark-price events, trade prints, book mid, and bar close. Binance
   settles funding on the mark price. Whenever the last write happened to be a trade
   price, the charge was computed on the wrong number.
"""
import pytest

from app.core.metrics import summary
from app.trading.simulation import PaperAccount


def _account():
    return PaperAccount(10000, slippage_bps=2)


def _open(account, symbol='X', entry=100, stop=98, target=104, qty=1):
    plan = {'symbol': symbol, 'side': 'LONG', 'entry': entry, 'stop': stop,
            'take_profit': target}
    assert account.open({'approved': True, 'quantity': qty}, plan)
    return plan


def test_a_tick_does_not_record_an_equity_sample():
    """The fix: only the bar boundary appends."""
    account = _account()
    _open(account)
    before = len(account.equity_curve)
    for step in range(1, 50):
        account.mark({'X': 100 + step * 0.01}, record=False)
    assert len(account.equity_curve) == before, (
        'a reprice is not a sample; 50 ticks appended %d points'
        % (len(account.equity_curve) - before))
    account.mark({'X': 101})
    assert len(account.equity_curve) == before + 1, 'a bar boundary records exactly one'


def test_record_defaults_to_true_so_a_bare_call_still_samples():
    """Backtest callers pass no `record` at all and must keep working."""
    account = _account()
    _open(account)
    before = len(account.equity_curve)
    account.mark({'X': 101})
    assert len(account.equity_curve) == before + 1


def test_equity_metrics_are_stable_under_tick_density():
    """The observable consequence: identical prices, identical metrics.

    Before the fix, marking the same 20-bar path with 1 tick per bar and with 200 ticks
    per bar produced completely different volatility and Sharpe, because the extra ticks
    were extra observations of a series treated as if it were evenly spaced.
    """
    prices = [100 + (index % 5) * 0.1 for index in range(20)]
    results = []
    for ticks in (1, 200):
        account = _account()
        _open(account)
        for price in prices:
            for _ in range(ticks):
                account.mark({'X': price}, record=(_ == ticks - 1))
        results.append(summary(account.equity_curve, account.trades,
                               initial_equity=10000, interval_ms=300000))
    assert results[0]['annualized_volatility_pct'] == \
        pytest.approx(results[1]['annualized_volatility_pct'], rel=1e-9), (
            'volatility must depend on the price path, not on how often it was polled')
    assert results[0]['sharpe'] == pytest.approx(results[1]['sharpe'], rel=1e-9)
    assert len(results[0]['periods_per_year'] and [1]) == 1


def test_funding_uses_the_mark_price_not_the_last_price():
    """The venue settles on the mark; a trade price is a different number."""
    account = _account()
    _open(account)
    # last price 100, mark price 200: funding must be charged on 200
    account.mark({'X': 100}, mark_prices={'X': 200}, record=False)
    paid = account.apply_funding({'X': 0.001}, event_time=1)
    assert paid == pytest.approx(200 * 1 * 0.001), (
        'funding was charged on the last price, not the mark')


def test_mark_price_falls_back_to_the_last_price():
    """A symbol with no mark yet must not silently value the position at zero."""
    account = _account()
    _open(account)
    account.mark({'X': 100}, record=False)
    assert account.mark_price('X') == 100
    account.mark({'X': 100}, mark_prices={'X': 101}, record=False)
    assert account.mark_price('X') == 101
    assert account.mark_price('UNSEEN', 42) == 42


def test_reported_equity_uses_the_mark_price():
    """Unrealised pnl is what the venue would call the account worth."""
    account = _account()
    _open(account)
    account.mark({'X': 100}, mark_prices={'X': 110}, record=False)
    position = account.positions['X']
    # Priced at the mark (110), not the last price (100), and net of the fee already
    # paid at entry. Pricing at the last price here would report 10000 - fee instead.
    assert position.entry > 100, 'entry carries slippage'
    assert account.equity == pytest.approx(10000 + (110 - position.entry) - position.entry_fee)
    assert account.unrealized_pnl() == pytest.approx(110 - position.entry)


def test_the_mark_price_survives_a_snapshot_round_trip():
    """Losing it on restart would silently revert funding to the last price."""
    account = _account()
    _open(account)
    account.mark({'X': 100}, mark_prices={'X': 123}, record=False)
    restored = PaperAccount.restore(account.snapshot())
    assert restored.mark_price('X') == 123


def test_closing_a_position_clears_both_prices():
    """A stale mark outliving its position would price the next one wrongly."""
    account = _account()
    _open(account)
    account.mark({'X': 100}, mark_prices={'X': 123}, record=False)
    account.close('X', 105)
    assert 'X' not in account.marks
    assert 'X' not in account.mark_prices
