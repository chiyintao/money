import pytest
from app.models.tabular_model import train_tabular, TabularPredictor, evaluate
from helpers import row


def dataset():
    return [row(i, label_end_time=i+1, future_return=(.005 if i%2 else -.005))
            for i in range(150)]


@pytest.mark.parametrize('backend', ['lightgbm', 'catboost'])
def test_native_training_and_loading(tmp_path, backend):
    pytest.importorskip(backend)
    # trials is explicit: it is the search size behind the candidate, and the deflated
    # Sharpe is meaningless without it. Passing 1 here means "one configuration was tried",
    # which is true of this test and is not true of a real tuning run.
    result = train_tabular(dataset(), backend, str(tmp_path), rounds=10, trials=1)
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
        train_tabular(rows, output_root=str(tmp_path), trials=1)


def test_a_missing_trial_count_is_refused(tmp_path):
    """The regression that made every candidate's Sharpe undeflated.

    trials used to default to 1, and 1 is not neutral: expected_max_sharpe returns 0 for a
    single trial, so the benchmark that the deflated Sharpe subtracts is zero and no
    correction is applied. A candidate then reports its raw Sharpe under a deflated name.
    A caller that does not know its search size cannot know whether its Sharpe is real,
    and must be told so rather than handed the most flattering case.
    """
    with pytest.raises(ValueError, match='trials_required'):
        train_tabular(dataset(), output_root=str(tmp_path), rounds=10)


@pytest.mark.parametrize('bad', [0, -1, 'many', 2.5])
def test_a_nonsensical_trial_count_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        train_tabular(dataset(), output_root=str(tmp_path), rounds=10, trials=bad)


def test_the_search_size_is_recorded_in_the_manifest(tmp_path):
    """An unrecorded search cannot be audited afterwards, which is how this started."""
    result = train_tabular(dataset(), 'lightgbm', str(tmp_path), rounds=10, trials=250,
                           search_note='horizon x edge_multiple x stop_width')
    assert result['search']['trials'] == 250
    assert 'horizon' in result['search']['note']
    assert result['validation']['trials'] == 250
