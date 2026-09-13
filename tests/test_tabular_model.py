import pytest
from app.models.tabular_model import train_tabular, TabularPredictor, evaluate
from helpers import row


def dataset():
    return [row(i, label_end_time=i+1, future_return=(.005 if i%2 else -.005))
            for i in range(150)]


@pytest.mark.parametrize('backend', ['lightgbm', 'catboost'])
def test_native_training_and_loading(tmp_path, backend):
    pytest.importorskip(backend)
    result = train_tabular(dataset(), backend, str(tmp_path), rounds=10)
    predictor = TabularPredictor(result['path'])
    assert predictor.predict(dataset()[0])['mode'] == 'shadow'
    assert result['metrics']['test']['rows'] > 0
    assert result['split']['purged'] > 0
    from pathlib import Path
    path = Path(result['path'])/result['model_file']
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        TabularPredictor(result['path'])


def test_cost_is_deducted_on_both_sides():
    assert evaluate([.01, -.01], [.01, -.01], 4)['mean_net_return'] == pytest.approx(.0096)


def test_unverified_labels_rejected(tmp_path):
    rows = dataset()
    del rows[0]['label_end_time']
    with pytest.raises(ValueError, match='unverified'):
        train_tabular(rows, output_root=str(tmp_path))
