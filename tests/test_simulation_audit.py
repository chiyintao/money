from app.trading.simulation import PaperAccount
from app.storage.storage import Store
from app.backtest.simulation_session import SimulationSession


def test_closed_trade_pnl_includes_both_fees():
    a = PaperAccount(cash=1000, fee_rate=.001, slippage_bps=0)
    assert a.open({'approved': True, 'quantity': 1}, {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 90, 'take_profit': 120})
    a.close('X', 110)
    trade = a.trades[0]
    assert round(trade['pnl'], 6) == round(10 - 0.1 - 0.11, 6)


def test_account_audit_detects_missing_mark():
    account=PaperAccount(cash=1000)
    account.open({'approved':True,'quantity':1},{'symbol':'X','side':'LONG','entry':100,'stop':90,'take_profit':120})
    account.marks.pop('X')
    audit=account.audit()
    assert not audit['ok'] and 'missing_mark:X' in audit['errors']

def test_session_snapshot_restores_every_trade(tmp_path):
    store = Store(str(tmp_path))
    session = SimulationSession(store)
    session.start(1000, 3, 'gainers', 2)
    session.account.open({'approved': True, 'quantity': 1}, {'symbol': 'X', 'side': 'LONG', 'entry': 100, 'stop': 90, 'take_profit': 120})
    session.account.close('X', 110)
    session._persist()
    restored = SimulationSession.restore(store, store.get_runtime('simulation_session'))
    assert len(restored.account.trades) == 1
    assert restored.session_id == session.session_id


def test_closed_trade_records_detailed_execution_breakdown():
    account = PaperAccount(cash=1000, fee_rate=.001, slippage_bps=0)
    plan = {'symbol': 'BTCUSDT', 'side': 'LONG', 'entry': 100, 'stop': 95, 'take_profit': 120, 'order_id': 'order-42'}
    assert account.open({'approved': True, 'quantity': 2}, plan, timestamp=1000)
    account.close('BTCUSDT', 110, 'take_profit', timestamp=61000)
    trade = account.trades[0]
    assert trade['order_id'] == 'order-42'
    assert trade['entry_time'] == 1000 and trade['exit_time'] == 61000
    assert trade['holding_ms'] == 60000
    assert trade['entry_notional'] == 200
    assert trade['gross_pnl'] == 20
    assert trade['entry_fee'] == .2 and trade['exit_fee'] == .22
    assert round(trade['fees'], 6) == .42
    assert trade['stop_price'] == 95 and trade['take_profit_price'] == 120
    assert trade['pnl_pct'] > 0


def test_store_expands_trade_payload_and_supports_legacy_fields(tmp_path):
    store = Store(str(tmp_path))
    store.record_trade({'trade_id': 'legacy-1', 'symbol': 'ETHUSDT', 'side': 'SHORT', 'entry': 200, 'exit': 190, 'qty': 1, 'pnl': 9.84, 'fees': .16, 'reason': 'manual'})
    trade = store.recent_trades(1)[0]
    assert trade['trade_id'] == 'legacy-1'
    assert trade['entry_price'] == 200 and trade['exit_price'] == 190
    assert trade['entry_notional'] == 200
    assert trade['gross_pnl'] == 10
    assert 'payload' not in trade
