from app.models.dataset_split import chronological_split
from app.features.feature_spec import FEATURES
from app.models.fit_model import fit
from app.models.walkforward import walk_forward
from helpers import rows as build_rows


def rows():
    return build_rows(100, symbols=('BTC', 'ETH'), future_return=None)


def test_grouped_split_purges_forward_labels():
    data = [{**r, 'label_end_time': r['timestamp']+5} for r in rows()]
    train, valid, test, meta = chronological_split(data)
    assert meta['ready'] and meta['purged'] == 20
    assert max(r['label_end_time'] for r in train) < min(r['timestamp'] for r in valid)
    assert max(r['label_end_time'] for r in valid) < min(r['timestamp'] for r in test)
    for part in (train, valid, test):
        assert all(sum(r['timestamp']==t for r in part)==2 for t in {r['timestamp'] for r in part})


def test_future_features_do_not_change_fit():
    original = rows()
    changed = [{**r, FEATURES[0]: 1e9} if r['timestamp'] >= 70 else r for r in original]
    first, second = fit(original, output=None), fit(changed, output=None)
    assert first['normalization'] == second['normalization']
    assert first['coefficients'] == second['coefficients']


def test_bad_dataset_is_not_trained():
    data = rows()
    assert not fit(data+[data[0]], output=None)['split']['ready']
    data[0][FEATURES[-1]] = float('nan')
    assert not fit(data, output=None)['split']['ready']


def test_walkforward_models_are_actually_fitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = walk_forward(rows(), 40, 10, 10)
    assert result['windows'] == 6
    assert all(r['model']['status']=='ok' for r in result['reports'])
    assert not (tmp_path/'data').exists()


def test_boosted_inference_matches_offline_and_short_costs():
    import pytest
    from app.models.advanced_model import fit_advanced, predict_advanced
    for target, side in ((.01, 'LONG'), (-.01, 'SHORT')):
        data = [{**r, 'future_return': target} for r in rows()]
        model = fit_advanced(data, output=None, rounds=2, cost_bps=4)
        prediction = predict_advanced(model, data[-1])
        assert prediction['side'] == side
        assert prediction['expected_return'] == pytest.approx(target)
        assert prediction['expected_net_return'] == pytest.approx(.0096)
        assert model['test']['mean_net_return'] == pytest.approx(.0096)
        assert model['test']['mae'] == pytest.approx(0)
    with pytest.raises(ValueError, match='legacy_model'):
        predict_advanced({'stumps': []}, {})

