"""A run that loses money in the replay must not be promoted.

The walk-forward test scores every prediction on its own. It cannot see two positions held
at once sharing one balance, a drawdown accumulating, or a handful of winners hiding a
long tail of losers -- and the run that produced the current candidates measured -10.33 bp
per prediction while the account trading it finished 620 trades at -17.01% with a 17.59%
drawdown. The promotion decision read the per-prediction numbers only.

These tests pin the rule: the replay is part of the decision, and a missing replay is not a
pass.
"""
from app.models.training_job import passes, verdict_for

OPTIONS = {'min_active_samples': 10, 'min_prob_positive': 0.95}

WALK_FORWARD = {
    'backend': 'lightgbm',
    'aggregate': {'active_samples': 1434, 'net_edge_bps': 12.0, 'folds_positive_edge': 4,
                  'folds_negative_edge': 1},
    'bootstrap': {'prob_positive': 0.99, 'low_bps': 2.0, 'high_bps': 30.0},
}


def replay(**overrides):
    evidence = {'backend': 'lightgbm', 'status': 'ok', 'costs_included': True, 'trades': 620,
                'net_return': 0.05, 'max_drawdown': 0.03}
    evidence.update(overrides)
    return evidence


def test_a_profitable_replay_alongside_a_passing_walk_forward_promotes():
    assert passes(WALK_FORWARD, OPTIONS, replay()) is True


def test_a_replay_that_lost_money_refuses_promotion():
    losing = replay(net_return=-0.1701, max_drawdown=0.1759)
    assert passes(WALK_FORWARD, OPTIONS, losing) is False
    assert any('portfolio replay' in line or 'net_return' in line
               for line in verdict_for(WALK_FORWARD, OPTIONS, losing))


def test_a_drawdown_beyond_the_limit_refuses_promotion():
    assert passes(WALK_FORWARD, OPTIONS, replay(max_drawdown=0.60)) is False


def test_a_missing_replay_is_not_a_pass():
    assert passes(WALK_FORWARD, OPTIONS, {'status': 'failed', 'error': 'no_data'}) is False
    assert passes(WALK_FORWARD, OPTIONS, None) is True  # explicit isolation only
    assert verdict_for(WALK_FORWARD, OPTIONS, {'status': 'failed'}) != []


def test_a_failing_walk_forward_is_never_rescued_by_the_replay():
    weak = {**WALK_FORWARD, 'bootstrap': {'prob_positive': 0.10, 'low_bps': -5.0,
                                          'high_bps': 1.0}}
    assert passes(weak, OPTIONS, replay()) is False
