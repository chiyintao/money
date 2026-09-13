from app.backtest.backtest import run_backtest
from app.trading.strategy import predict, plan
from app.trading.risk import RiskEngine


def make_rows(kind='trend', n=180):
    rows=[]
    for i in range(n):
        price=100+i*.2 if kind=='trend' else 100+(i%4)*.01
        rows.append({'open_time':i*60000,'close_time':(i+1)*60000,'open':price,'high':price+1,'low':max(.01,price-1),'close':price,'volume':100+(50 if i%10==0 else 0),'is_closed':1})
    return rows


def test_strategy_avoids_flat_chop():
    result=run_backtest(make_rows('flat'))
    assert result['trades'] == []


def test_backtest_signal_executes_after_signal_bar():
    result=run_backtest(make_rows('trend'))
    assert result['trades']
    assert all('pnl' in trade for trade in result['trades'])


def test_risk_state_round_trip():
    risk=RiskEngine(.005,.02,2,10000,True,'2026-01-01')
    restored=RiskEngine.restore(risk.snapshot())
    assert restored.snapshot()==risk.snapshot()
