import pytest
from app.trading.broker import PaperBroker
from app.core.domain import OrderIntent
from app.trading.simulation import PaperAccount


def test_partial_fills_accumulate_in_account_before_confirmation():
    broker = PaperBroker(slippage_bps=0)
    account = PaperAccount()
    plan = {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 80, 'take_profit': 130}
    order, _ = broker.submit(OrderIntent('X', 'BUY', 2, order_id='order'), plan=plan)
    accept = lambda fill: account.open_fill(fill, plan)
    fills, status = broker.process(order.order_id, 100, available_qty=1, accept_fill=accept, timestamp=1000)
    assert status == 'PARTIALLY_FILLED'
    fills, status = broker.process(order.order_id, 110, available_qty=1, accept_fill=accept, timestamp=2000)
    assert status == 'FILLED'
    position = account.positions['X']
    assert position.qty == 2
    assert position.entry == 105
    assert position.entry_fee == pytest.approx(.084)
    assert account.cash == pytest.approx(10000-.084)
    assert position.opened_at == 1000


def test_rejected_account_does_not_advance_order():
    broker = PaperBroker()
    account = PaperAccount(cash=1)
    plan = {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 80, 'take_profit': 130}
    order, _ = broker.submit(OrderIntent('X', 'BUY', 1, order_id='order'), plan=plan)
    fills, reason = broker.process(order.order_id, 100, accept_fill=lambda f: account.open_fill(f, plan))
    assert reason == 'account_rejected_fill'
    assert not fills and not account.positions
    assert broker.orders[order.order_id].filled_quantity == 0
    assert broker.orders[order.order_id].status == 'OPEN'
