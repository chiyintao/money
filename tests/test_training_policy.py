from app.models.training_policy import retrain_required, validate_split_sizes


def test_split_policy_rejects_empty_or_small_evaluation_windows():
    assert validate_split_sizes([{}] * 10, .7, .15)['ready'] is False
    split = validate_split_sizes([{}] * 60, .7, .15)
    assert split['ready'] and split['train'] == 42 and split['validation'] == 9 and split['test'] == 9


def test_retraining_is_triggered_by_age_dataset_or_features():
    base = {'created_at': 1000, 'dataset': {'sha256': 'a'}, 'feature_version': 'v1'}
    assert retrain_required(base, now_ms=1001, max_age_ms=100)[0] is False
    assert retrain_required(base, now_ms=1100, max_age_ms=100)[1] == 'model_expired'
    assert retrain_required(base, now_ms=1001, current_dataset_hash='b')[1] == 'dataset_changed'
    assert retrain_required(base, now_ms=1001, current_feature_version='v2')[1] == 'features_changed'
