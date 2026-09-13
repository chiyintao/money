from app.core.metrics import summary

def test_summary_uses_explicit_initial_equity():
    result=summary([10100,10000],[{'pnl':10}],initial_equity=10000)
    assert result['start_equity']==10000
    assert abs(result['return_pct']) < 1e-9
    assert result['max_drawdown_pct'] > 0

def test_summary_reports_distribution_metrics():
    result=summary([10000,10100,10200],[{'pnl':100},{'pnl':-50}])
    assert result['trades']==2
    assert result['profit_factor']==2
    assert result['expectancy']==25
    assert 'sharpe' in result and 'sortino' in result
    assert result['total_fees']==0
    assert result['total_funding']==0
