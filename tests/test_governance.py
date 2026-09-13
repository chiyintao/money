from app.models.drift import drift_report
from app.models.model_registry import ModelRegistry
from app.ops.prometheus import collect, render
from app.models.calibration import fit_isotonic
from app.features.feature_spec import FEATURE_VERSION


def calibrator():
    """A calibrator artifact, because the manifest flag alone no longer promotes a model."""
    return fit_isotonic([.1, .2, .3, .4, .5, .6], [0, 0, 1, 0, 1, 1])

def test_drift_and_registry(tmp_path):
    reference = {'x': list(range(500))}
    shifted = {'x': [value + 400 for value in range(500)]}
    assert drift_report(reference, shifted)['status'] == 'drift'
    assert drift_report(reference, {'x': list(range(500))})['status'] == 'stable'
    reg=ModelRegistry(str(tmp_path/'models')); m=reg.register({'coef':[1],'feature_version':FEATURE_VERSION,'split':{'ready':True,'label_intervals_verified':True}}, {'test':{'rows':100,'directional_accuracy_pct':55},'portfolio_oos':{'costs_included':True,'trades':40,'net_return':.02,'max_drawdown':.05},'calibration':{'fitted':True}}, {'rows':300}, 'v1', calibrator=calibrator()); assert m['sha256']; assert m['feature_version'] == FEATURE_VERSION; assert reg.promote('v1')=='v1'; assert reg.current()['version']=='v1'

def test_metrics_text():
    # The exported series come from the dashboard state, so a figure can only appear here
    # if it also appears there. The previous Metrics class was a counter store nothing
    # called while /metrics served one hand-written line.
    state = {'equity': 100.5, 'account': {'cash': 50, 'positions': [{'symbol': 'BTCUSDT'}]},
             'metrics': {'trades': 3, 'sharpe': 1.25, 'gross_profit': 200.0,
                         'total_fees': 20.0, 'total_funding': -5.0},
             'risk_state': {'halted': False, 'peak_equity': 120.0, 'loss_streak': 2,
                            'streak_scale': 0.5},
             'decision_counts': {'total': 9, 'orders': 4, 'rejections': {'stale_market_data': 2}},
             'simulation': {}, 'health': {}, 'open_orders': [1, 2]}
    body = render(collect(state))
    assert 'paper_equity 100.5' in body
    assert 'paper_open_positions 1' in body
    assert 'paper_trades_total 3' in body
    assert 'paper_rejections_total{reason="stale_market_data"} 2' in body
    assert 'paper_cost_share_pct 12.5' in body
    assert '# TYPE paper_equity gauge' in body
    assert '# TYPE paper_trades_total counter' in body
    # Drawdown from the high-water mark, not from the starting balance.
    assert 'paper_risk_drawdown 0.1625' in body


def test_registry_expires_and_verifies_checksum(tmp_path):
    reg=ModelRegistry(str(tmp_path/'models'))
    reg.register({'coef':[1],'feature_version':FEATURE_VERSION,'split':{'ready':True,'label_intervals_verified':True}}, {'test':{'rows':100,'directional_accuracy_pct':55},'portfolio_oos':{'costs_included':True,'trades':40,'net_return':.02,'max_drawdown':.05},'calibration':{'fitted':True}}, {'rows':300}, 'v1', calibrator=calibrator())
    reg.promote('v1', max_age_ms=60000)
    assert reg.current(now_ms=10**15)['status'] == 'expired'
    with __import__('pytest').raises(ValueError, match='production_model_expired'):
        reg.load_production(now_ms=10**15)


def test_risk_reset_for_new_simulation():
    from app.trading.risk import RiskEngine
    risk=RiskEngine(max_daily_loss=.02,day_start_equity=10000,halted=True,day_key='old-day')
    risk.reset_for_session(2500,day_key='new-day')
    assert risk.day_start_equity == 2500
    assert risk.halted is False
    assert risk.day_key == 'new-day'
    assert risk.approve({'side':'LONG','entry':100,'stop':90},2500)['approved'] is True
    blocked=risk.approve({'side':'LONG','entry':100,'stop':90},2400)
    assert blocked['reason']=='daily_loss_circuit_breaker'
    assert risk.snapshot()['halt_threshold']==2450
    risk.reset_for_session(5000,day_key='new-session')
    assert risk.snapshot()['halted'] is False
