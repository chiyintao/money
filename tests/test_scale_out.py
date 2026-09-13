"""Scale-out exits, and the reconciliation that the 0.93-bar holding time was hiding.

Two things the audit found on the accounting side, both of which cost more than they look:

1. `close()` popped the whole position, so a partial exit could not be expressed. That is
   not a venue limitation and it is exactly the shape a triple-barrier label needs -- the
   label is *defined* by scaling out at the upper barrier and letting the rest run. A
   position now carries the fills that built it, and closing consumes them FIFO.

2. `position_quantities` read `position.quantity` while the account stores `qty`, so it
   returned 0 for every symbol. The check could therefore never fail, and a position whose
   entry order only partly filled and then expired was invisible: risk sized from the
   position while the order side believed it had filled less.
"""
from dataclasses import replace

import pytest

from app.backtest.account_reconcile import (order_position_mismatches, position_quantities,
                                            reconcile)
from app.core.domain import OrderIntent
from app.trading.broker import PaperBroker
from app.trading.simulation import PaperAccount, Lot, Position


def _long(qty=3.0, entry=100.0, stop=95.0, target=120.0):
    # No fee and no slippage: these tests are about lot arithmetic, so the exit price
    # should be the price asked for. Cost handling has its own tests.
    account = PaperAccount(10_000, fee_rate=0.0, slippage_bps=0.0)
    account.positions['X'] = Position('X', 'LONG', qty, entry, stop, target)
    account.positions['X'].lots.append(Lot(entry, qty, 0.0, 'o1', 1))
    account.marks['X'] = entry
    return account


# ------------------------------------------------------------------ lots
def test_a_position_records_the_fills_that_built_it():
    """The average price becomes a consequence of the lots, not a parallel calculation."""
    account = PaperAccount(10_000, fee_rate=0.0)
    plan = {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 95, 'take_profit': 120}
    assert account.open({'approved': True, 'quantity': 1}, plan)
    position = account.positions['X']
    assert len(position.lots) == 1
    assert position.lots[0].qty == 1


def test_partial_fills_keep_the_weighted_average_and_every_lot():
    """The documented case: 0.6@100 then 0.9@100.5 -> qty 1.5, entry 100.30."""
    account = PaperAccount(10_000, fee_rate=0.0)
    plan = {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 95, 'take_profit': 120}
    account.open({'approved': True, 'quantity': 0.6}, {**plan, 'execution_price': 100.0,
                                                         'fee': 0.0, 'order_id': 'o1'})
    from app.core.domain import Fill
    account.open_fill(Fill('o1', 'X', 'BUY', 0.9, 100.5, 0.0, event_time=2), plan)
    position = account.positions['X']
    assert position.qty == pytest.approx(1.5)
    assert position.entry == pytest.approx(100.30)
    assert [lot.qty for lot in position.lots] == pytest.approx([0.6, 0.9])


def test_a_partial_close_leaves_the_position_open():
    """The capability that did not exist: scaling out."""
    account = _long(qty=3.0)
    realised = account.close('X', 110.0, 'take_profit', quantity=1.0)
    assert 'X' in account.positions, 'a partial exit must not remove the position'
    assert account.positions['X'].qty == pytest.approx(2.0)
    assert realised == pytest.approx(10.0), 'one unit closed at +10'
    assert account.trades[-1]['partial'] is True
    assert account.trades[-1]['remaining_qty'] == pytest.approx(2.0)
    assert account.trades[-1]['lots_consumed'] == 1


def test_closing_everything_still_removes_the_position():
    """The old behaviour has to survive, since every existing caller relies on it."""
    account = _long(qty=3.0)
    account.close('X', 110.0, 'manual')
    assert 'X' not in account.positions
    assert account.trades[-1]['partial'] is False
    assert account.trades[-1]['qty'] == pytest.approx(3.0)


def test_scaling_out_totals_the_same_as_one_exit():
    """Arithmetic invariant: splitting an exit must not create or destroy pnl.

    This is the property that makes scale-out usable in a backtest at all -- if three
    partial exits summed to something different from one full exit at the same price,
    every result would depend on the exit schedule rather than on the strategy.
    """
    price = 110.0
    whole = _long(qty=3.0)
    whole.close('X', price, 'manual')
    split = _long(qty=3.0)
    for _ in range(3):
        split.close('X', price, 'manual', quantity=1.0)
    assert sum(t['pnl'] for t in split.trades) == pytest.approx(whole.trades[0]['pnl'])
    assert sum(t['qty'] for t in split.trades) == pytest.approx(whole.trades[0]['qty'])


def test_fifo_consumes_the_oldest_lot_first():
    """Which lot leaves decides the realised pnl, so the order has to be defined."""
    account = PaperAccount(10_000, fee_rate=0.0, slippage_bps=0.0)
    position = Position('X', 'LONG', 2.0, 100.0, 95, 120)
    position.lots = [Lot(90.0, 1.0, 0.0, 'a', 1), Lot(110.0, 1.0, 0.0, 'b', 2)]
    account.positions['X'] = position
    account.marks['X'] = 100.0
    # closing one unit at 100 must take the 90 lot: +10, not the 110 lot: -10
    assert account.close('X', 100.0, 'manual', quantity=1.0) == pytest.approx(10.0)
    assert account.positions['X'].lots[0].price == 110.0


def test_funding_is_apportioned_across_a_partial_exit():
    """Funding was charged on the whole position; only the departing share is realised."""
    account = _long(qty=2.0)
    account.positions['X'].funding_paid = 4.0
    account.close('X', 100.0, 'manual', quantity=1.0)
    assert account.trades[-1]['funding'] == pytest.approx(2.0)
    assert account.positions['X'].funding_paid == pytest.approx(2.0), \
        'the remaining half keeps its own half of the funding'


def test_a_zero_quantity_close_is_a_no_op():
    """Not a way to reset a position by accident."""
    account = _long(qty=2.0)
    assert account.close('X', 100.0, 'manual', quantity=0.0) == 0.0
    assert account.positions['X'].qty == pytest.approx(2.0)
    assert not account.trades


def test_a_restored_position_without_lots_can_still_be_closed():
    """Snapshots written before lots existed must not become unclosable."""
    account = _long(qty=2.0)
    account.positions['X'].lots = []          # simulate an old snapshot
    assert account.close('X', 110.0, 'manual', quantity=1.0) == pytest.approx(10.0)
    assert account.positions['X'].qty == pytest.approx(1.0)


# ------------------------------------------------------- reconciliation
def test_position_quantities_reads_the_quantity_the_account_actually_stores():
    """It read `.quantity`; Position stores `.qty`. Every symbol reported 0."""
    account = _long(qty=2.5)
    assert position_quantities(account) == {'X': 2.5}


def test_order_position_mismatch_is_reported():
    """A partly filled entry that expired leaves a position no order completed."""
    account = _long(qty=1.0)
    broker = PaperBroker(fee_rate=.0004, slippage_bps=2)
    order, _ = broker.submit(OrderIntent('X', 'BUY', 1.0, order_id='o1'))
    broker.orders['o1'] = replace(order, filled_quantity=1.0, status='FILLED')
    assert order_position_mismatches(account, list(broker.orders.values())) == []
    broker.orders['o1'] = replace(order, filled_quantity=0.4, status='EXPIRED')
    mismatches = order_position_mismatches(account, list(broker.orders.values()))
    assert len(mismatches) == 1
    assert mismatches[0]['symbol'] == 'X'
    assert mismatches[0]['from_orders'] == pytest.approx(0.4)
    assert mismatches[0]['held'] == pytest.approx(1.0)
    assert mismatches[0]['difference'] == pytest.approx(0.6)


def test_an_order_without_a_position_is_not_a_mismatch():
    """That is the entry-not-yet-accepted case, covered by the open-order check."""
    account = PaperAccount(10_000)
    broker = PaperBroker(fee_rate=.0004, slippage_bps=2)
    order, _ = broker.submit(OrderIntent('X', 'BUY', 1.0, order_id='o1'))
    broker.orders['o1'] = replace(order, filled_quantity=0.4, status='PARTIALLY_FILLED')
    assert order_position_mismatches(account, list(broker.orders.values())) == []


def test_the_reconciliation_report_names_the_finding():
    """The finding string changed, because the comparison changed."""
    class FakeBroker:
        def __init__(self):
            self.orders = {}
            self.transition_refusals = {}

        def open_orders(self):
            return []

        def order_counts(self):
            return {}

    class FakeStore:
        def orders(self, active_only=True):
            return []

    class FakeRuntime:
        pass

    account = _long(qty=1.0)
    broker = FakeBroker()
    order, _ = PaperBroker(fee_rate=.0004).submit(OrderIntent('X', 'BUY', 1.0, order_id='o1'))
    broker.orders['o1'] = replace(order, filled_quantity=0.25, status='EXPIRED')
    runtime = FakeRuntime()
    runtime.account, runtime.broker, runtime.store = account, broker, FakeStore()
    report = reconcile(runtime)
    assert 'position_order_mismatch' in report['findings']
    assert report['positions']['held'] == {'X': 1.0}
