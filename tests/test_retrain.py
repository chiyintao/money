import asyncio

from app.models.retrain import RetrainScheduler


def test_scheduler_skips_fresh_model():
    calls = []
    scheduler = RetrainScheduler(lambda: calls.append(1), lambda: {'created_at': 1000, 'dataset': {'sha256': 'a'}, 'feature_version': 'v1'}, interval_ms=100, feature_version='v1')
    result = asyncio.run(scheduler.run_once(now_ms=1001, dataset_hash='a'))
    assert result['status'] == 'skipped' and calls == []


def test_scheduler_runs_stale_model_without_promoting():
    scheduler = RetrainScheduler(lambda: {'status': 'candidate'}, lambda: {'created_at': 1000, 'dataset': {'sha256': 'a'}, 'feature_version': 'v1'}, interval_ms=100, feature_version='v1')
    result = asyncio.run(scheduler.run_once(now_ms=1100, dataset_hash='a'))
    assert result['status'] == 'trained' and result['result']['status'] == 'candidate'
