from app.trading.margin import DEFAULT_TIERS, MaintenanceSchedule, MaintenanceTier, MarginModel
from app.trading.simulation import PaperAccount, Position

def test_margin_and_liquidation():
    model=MarginModel(leverage=2,maintenance_rate=.005); p=Position('X','LONG',1,100,90,120); assert model.initial_margin(100)==50; assert model.liquidation_price(p)<100; assert not model.should_liquidate(1,{'X':p},{'X':100})[0]

def test_account_liquidation_and_restore():
    a=PaperAccount(200,fee_rate=0); p={'symbol':'X','side':'LONG','entry':100,'stop':.01,'take_profit':1000}; assert a.open({'approved':True,'quantity':2},p)
    # Force the breach through the schedule, which is what actually decides now. A flat
    # 200% maintenance rate makes any equity insufficient.
    a.margin.schedule=MaintenanceSchedule.flat(2.0)
    a.mark({'X':99}); assert a.liquidations; restored=PaperAccount.restore(a.snapshot()); assert restored.liquidations==a.liquidations


# --------------------------------------------------------------- risk tiers
def test_maintenance_rises_with_notional_rather_than_staying_flat():
    schedule=MaintenanceSchedule()
    small=schedule.maintenance_margin(10_000)/10_000
    large=schedule.maintenance_margin(2_000_000)/2_000_000
    # A flat rate understates the requirement of exactly the positions that matter most.
    assert small<large
    assert small==0.004


def test_brackets_are_continuous_at_every_boundary():
    schedule=MaintenanceSchedule()
    for lower,upper in zip(DEFAULT_TIERS,DEFAULT_TIERS[1:]):
        edge=lower.notional_cap
        below=schedule.maintenance_margin(edge-1e-6)
        above=schedule.maintenance_margin(edge+1e-6)
    # The cumulative amount exists precisely so the requirement does not jump.
    assert abs(below-above)<1e-3


def test_a_per_symbol_bracket_overrides_the_defaults():
    schedule=MaintenanceSchedule(brackets={'XUSDT':(MaintenanceTier(float('inf'),0.10,0.0),)})
    assert schedule.maintenance_margin(1_000,'XUSDT')==100.0
    assert schedule.maintenance_margin(1_000,'YUSDT')==4.0


def test_liquidation_price_uses_the_positions_own_bracket():
    model=MarginModel(leverage=10)
    small=Position('X','LONG',1,100,90,120)
    large=Position('X','LONG',1_000_000,100,90,120)
    # Leverage is not the only thing that moves the level: the bracket does. A larger
    # position sits in a higher maintenance bracket, so its buffer is thinner and it is
    # liquidated closer to its entry -- the opposite of what a flat 0.5% rate implies.
    assert model.liquidation_price(large)>model.liquidation_price(small)
    assert model.liquidation_price(small)==90.4


def test_a_short_liquidates_above_its_entry():
    model=MarginModel(leverage=5)
    assert model.liquidation_price(Position('X','SHORT',1,100,110,90))>100


def test_liquidation_charges_the_liquidation_fee_and_never_leaves_negative_cash():
    account=PaperAccount(100,fee_rate=0)
    account.margin=MarginModel(leverage=20,schedule=MaintenanceSchedule.flat(0.01))
    account.positions['X']=Position('X','LONG',20,100,1,1000)
    account.marks['X']=100
    # A gap far through the maintenance level: the loss exceeds the whole deposit.
    record=account.settle_liquidation.__self__ and None
    account.marks['X']=80
    account.mark({'X':80})
    assert account.positions=={}
    assert account.cash>=0, 'an account balance below zero does not exist'
    assert account.bankruptcies==1
    trade=account.trades[-1]
    assert trade['reason']=='liquidation'
    assert trade['liquidation_fee']>0


def test_bar_exits_use_the_high_and_low_not_the_close():
    # A bar that pierced the stop and closed back above it used to leave the position
    # open. The real one would have been stopped out.
    account=PaperAccount(10_000,fee_rate=0)
    account.positions['X']=Position('X','LONG',1,100,95,120)
    account.marks['X']=100
    bar={'open':100,'high':101,'low':94,'close':100.5}
    account.mark({'X':100.5},bars={'X':bar})
    assert account.positions=={}
    assert account.trades[-1]['reason']=='stop_loss'
    assert account.trades[-1]['reason_detail']['source']=='bar'


def test_a_stop_and_a_target_are_no_longer_the_same_reason():
    # A tick exit is the point of this test: it checks that a stop and a target are told
    # apart. Whether a tick may close a position at all is a separate question, covered
    # in test_exit_source.py.
    account=PaperAccount(10_000,fee_rate=0,exit_on_tick=True)
    account.positions['X']=Position('X','LONG',1,100,95,120)
    account.marks['X']=100
    account.mark({'X':121})
    assert account.positions=={}
    assert account.trades[-1]['reason']=='take_profit'
    assert account.trades[-1]['reason_detail']['source']=='tick'


def test_ambiguous_bars_take_the_stop():
    account=PaperAccount(10_000,fee_rate=0)
    account.positions['X']=Position('X','LONG',1,100,95,110)
    account.marks['X']=100
    account.mark({'X':100},bars={'X':{'open':100,'high':115,'low':90,'close':100}})
    assert account.trades[-1]['reason']=='stop_loss'
    assert account.trades[-1]['reason_detail']['ambiguous'] is True
