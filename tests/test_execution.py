from app.trading.broker import ContractSpec, PaperBroker
from app.core.domain import OrderIntent
from app.trading.guards import MarketGuard, PortfolioLimits

def test_broker_rounds_and_costs():
    broker=PaperBroker(slippage_bps=2,specs={'X':ContractSpec('X',tick_size=.1,step_size=.1,min_notional=5)})
    fill,reason=broker.execute(OrderIntent('X','BUY',1.23),100); assert reason=='filled'; assert fill.quantity==1.2 and fill.fee>0

def test_account_uses_broker_fill_without_reslipping():
    broker=PaperBroker(slippage_bps=10)
    fill,reason=broker.execute(OrderIntent('X','BUY',1),100)
    assert reason=='filled'
    from app.trading.simulation import PaperAccount
    account=PaperAccount(cash=1000, slippage_bps=10)
    assert account.open({'approved':True,'quantity':fill.quantity}, {'symbol':'X','side':'LONG','entry':100,'execution_price':fill.price,'fee':fill.fee,'stop':90,'take_profit':120})
    assert account.positions['X'].entry == fill.price
    assert account.cash == 1000 - fill.fee

def test_fill_and_position_share_order_id():
    from app.trading.simulation import PaperAccount
    broker=PaperBroker(slippage_bps=0)
    intent=OrderIntent('X','BUY',1,order_id='order-1')
    fill,reason=broker.execute(intent,100)
    account=PaperAccount(cash=1000,slippage_bps=0)
    assert reason=='filled'
    assert fill.order_id=='order-1'
    assert account.open({'approved':True,'quantity':fill.quantity},{'symbol':'X','side':'LONG','entry':100,'execution_price':fill.price,'fee':fill.fee,'stop':90,'take_profit':120,'order_id':fill.order_id})
    assert account.positions['X'].order_id=='order-1'
    account.close('X',110)
    assert account.trades[0]['order_id']=='order-1'

def test_limit_order_partial_fill_and_cancel():
    broker=PaperBroker(slippage_bps=100)
    order,reason=broker.submit(OrderIntent('X','BUY',2,order_type='LIMIT',limit_price=100,order_id='limit-1'))
    assert reason=='accepted' and order.status=='OPEN'
    fills,status=broker.process('limit-1',101,ask=101,available_qty=1)
    assert fills==[] and status=='not_marketable'
    fills,status=broker.process('limit-1',100,ask=100,available_qty=1)
    assert status=='PARTIALLY_FILLED' and len(fills)==1 and fills[0].price==100
    fills,status=broker.process('limit-1',99,ask=99,available_qty=1)
    assert status=='FILLED' and len(fills)==1 and fills[0].price==99
    canceled,status=broker.cancel('limit-1')
    assert status=='order_not_open' and canceled.status=='FILLED'

def test_depth_execution_uses_weighted_average_and_rejects_short_depth():
    broker=PaperBroker(fee_rate=.001)
    fill,reason=broker.execute_depth(OrderIntent('X','BUY',3,order_id='depth-1'),asks=[(100,1),(102,2)])
    assert reason=='filled'
    assert fill.quantity==3 and round(fill.price,2)==101.33
    missing,reason=broker.execute_depth(OrderIntent('X','BUY',4),asks=[(100,1),(102,2)])
    assert missing is None and reason=='insufficient_depth'

def test_order_expiry_expires_open_order():
    broker=PaperBroker()
    order,status=broker.submit(OrderIntent('X','BUY',1,order_type='LIMIT',limit_price=100,order_id='expire-1'),timestamp=100,plan={'expires_at':200})
    assert status=='accepted'
    expired=broker.expire_orders(200)
    # EXPIRED, not CANCELED. A deadline passing and somebody deciding to stop are different
    # events; reporting both as CANCELED left the audit trail unable to say which happened.
    assert len(expired)==1 and expired[0].status=='EXPIRED'
    fills,status=broker.process('expire-1',99,ask=99)
    assert fills==[] and status=='order_not_open'

def test_cancelled_order_cannot_fill():
    broker=PaperBroker(slippage_bps=0)
    broker.submit(OrderIntent('X','BUY',1,order_type='LIMIT',limit_price=100,order_id='limit-2'))
    _,reason=broker.cancel('limit-2')
    assert reason=='canceled'
    fills,status=broker.process('limit-2',99,ask=99)
    assert fills==[] and status=='order_not_open'

def test_cancel_api_state_is_terminal():
    broker=PaperBroker()
    order,_=broker.submit(OrderIntent('X','BUY',1,order_type='LIMIT',limit_price=100,order_id='api-cancel'))
    canceled,reason=broker.cancel(order.order_id)
    assert reason=='canceled' and canceled.status=='CANCELED'

def test_stale_market_rejects_entry():
    guard=MarketGuard(100); guard.observe('X',1000); assert guard.approve_new_entry('X',1050)[0]; assert not guard.approve_new_entry('X',1200)[0]

def test_portfolio_limits():
    limits=PortfolioLimits(max_positions=1,max_symbol_leverage=1.0,max_gross_leverage=1.0)
    assert limits.approve('X',50,{},equity=100)[0]
    assert not limits.approve('Y',50,{'X':50},equity=100)[0]
