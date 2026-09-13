"""Which event is allowed to close a position.

The audit found two mutually exclusive exit paths reaching the same account from
different callers:

    pumps.mark_pump       every 2 seconds   mark price, no bar   price comparison
    decision_loop         every bar         close + OHLC bar     evaluate_bar_exit
    feeds_handlers        event driven      price, no bar        price comparison

Whichever arrived first decided the exit, and the 2-second pump almost always won. With a
40bp stop against 5.6-20bp of typical 5m movement, the measured median holding time was
0.93 bars: positions were being closed by an instantaneous tick rather than by a bar that
closed through the level. The backtest sees only bars, so it could not reproduce a single
one of those exits -- the replay and the live system were simulating different strategies.

`exit_on_tick` now names the choice, defaulting to bar-confirmed (the freqtrade rule, and
the only setting under which the backtest is evidence about the live system).
"""
import pytest

from app.trading.simulation import PaperAccount, Position


def _long(account, stop=95, target=120):
    account.positions['X'] = Position('X', 'LONG', 1, 100, stop, target)
    account.marks['X'] = 100


def test_a_tick_does_not_close_a_position_by_default():
    """The default: a tick through the stop is a wick, not a stop-out."""
    account = PaperAccount(10_000, fee_rate=0)
    _long(account)
    account.mark({'X': 94})            # through the stop, no bar
    assert 'X' in account.positions, 'a bare tick closed a position'
    assert not account.trades


def test_a_closed_bar_through_the_stop_does_close_it():
    """The bar path is the one that counts, and it is unchanged."""
    account = PaperAccount(10_000, fee_rate=0)
    _long(account)
    account.mark({'X': 96}, bars={'X': {'open': 100, 'high': 101, 'low': 94, 'close': 96}})
    assert 'X' not in account.positions
    assert account.trades[-1]['reason'] == 'stop_loss'
    assert account.trades[-1]['reason_detail']['source'] == 'bar'


def test_a_target_needs_a_bar_too():
    """Symmetry: the same rule applies to a take-profit."""
    account = PaperAccount(10_000, fee_rate=0)
    _long(account)
    account.mark({'X': 125})           # through the target, no bar
    assert 'X' in account.positions
    account.mark({'X': 125}, bars={'X': {'open': 100, 'high': 126, 'low': 100, 'close': 125}})
    assert 'X' not in account.positions
    assert account.trades[-1]['reason'] == 'take_profit'
    assert account.trades[-1]['reason_detail']['source'] == 'bar'


def test_tick_exits_are_still_available_and_are_labelled():
    """Opting in must be possible, and the audit trail must say which rule fired."""
    account = PaperAccount(10_000, fee_rate=0, exit_on_tick=True)
    _long(account)
    account.mark({'X': 94})
    assert 'X' not in account.positions
    detail = account.trades[-1]['reason_detail']
    assert detail['source'] == 'tick', 'the source must be recorded, not implied'


def test_a_liquidation_still_fires_without_a_bar():
    """Solvency is not a strategy choice: maintenance is checked on every reprice.

    Suppressing tick exits must not suppress the liquidation check, which is the one
    thing that has to react between bars.
    """
    from app.trading.margin import MaintenanceSchedule

    account = PaperAccount(1_000, fee_rate=0)
    account.positions['X'] = Position('X', 'LONG', 10, 100, 95, 120)
    account.marks['X'] = 100
    account.mark_prices['X'] = 100
    # A flat 200% maintenance rate makes any equity insufficient, which is how the
    # existing margin tests force a breach without depending on the tier boundaries.
    account.margin.schedule = MaintenanceSchedule.flat(2.0)
    account.mark({'X': 99}, mark_prices={'X': 99})
    assert 'X' not in account.positions, 'a liquidation must still happen intrabar'
    assert account.liquidations, 'the liquidation was not recorded'


def test_every_exit_records_its_source():
    """The audit could not tell a stop from a target on the existing 94 trades."""
    for tick in (False, True):
        account = PaperAccount(10_000, fee_rate=0, exit_on_tick=tick)
        _long(account)
        account.mark({'X': 94}, bars={'X': {'open': 100, 'high': 101, 'low': 93, 'close': 94}})
        assert account.trades[-1]['reason_detail'].get('source') in ('bar', 'tick')
        account2 = PaperAccount(10_000, fee_rate=0, exit_on_tick=tick)
        _long(account2)
        account2.mark({'X': 94})
        for trade in account2.trades:
            assert trade['reason_detail'].get('source') in ('bar', 'tick')
