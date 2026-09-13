from app.features.features import ema, rsi, snapshot
from app.trading.risk import RiskEngine
from app.storage.storage import Store
from app.trading.simulation import PaperAccount

def rows(n=80):
    return [{"open":100+i*.2,"high":101+i*.2,"low":99+i*.2,"close":100+i*.2,"volume":10,"open_time":i,"close_time":i} for i in range(n)]

def test_features():
    f=snapshot(rows()); assert f["ema20"]>f["ema50"]; assert 0<=f["rsi"]<=100; assert len(ema([1,2,3],2))==3

def test_risk_caps_position():
    p={"entry":100,"stop":98}; d=RiskEngine(.005,.02,2,10000).approve(p,10000,0); assert d["approved"]; assert d["risk_cash"]<=50.01

def test_daily_breaker():
    p={"side":"LONG","entry":100,"stop":98}; engine=RiskEngine(.005,.02,2,10000); assert engine.approve(p,10000,0)["approved"]; engine.day_start_equity=10000; d=engine.approve(p,9700,0); assert not d["approved"]

def test_closed_bar_storage_is_idempotent(tmp_path):
    store=Store(str(tmp_path)); row={"open_time":1,"close_time":2,"open":1,"high":2,"low":1,"close":1.5,"volume":3,"is_closed":1}; store.upsert_candles('X','1m',[row]); store.upsert_candles('X','1m',[row]); assert len(store.candles('X','1m'))==1; store.close()

def test_paper_costs_reduce_profit():
    # exit_on_tick: this test is about fees and slippage, and a tick exit is the shortest
    # path to a closed trade. The bar-confirmed default is covered in test_exit_source.py.
    account=PaperAccount(10000,slippage_bps=2,exit_on_tick=True); plan={"symbol":"X","side":"LONG","entry":100,"stop":98,"take_profit":104}; assert account.open({"approved":True,"quantity":1},plan); account.mark({"X":104}); assert account.trades[0]["pnl"] < 4
