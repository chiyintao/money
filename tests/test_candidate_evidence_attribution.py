"""Evidence must belong to the artifact it describes.

The run computes one walk-forward result -- always with the LightGBM backend -- and then
attached it to every candidate it wrote, including the CatBoost one. A reader selecting
between candidates, or a promotion gate later, therefore saw a CatBoost manifest whose
\`walk_forward.backend\` said lightgbm: another model's validation standing in for this
one's. The portfolio replay has the same problem, since it consumes those same
predictions.

The rule these tests pin: an artifact is only ever labelled with evidence produced from
its own predictions. Evidence that belongs to a different backend is recorded as such --
attributed and marked as not applying -- rather than copied in as if it were this
artifact's result.
"""
import json
from pathlib import Path

from app.models.training_job import TrainingRunner


def manifest(tmp_path, backend):
    folder = tmp_path / ('%s-abc' % backend)
    folder.mkdir()
    (folder / 'manifest.json').write_text(json.dumps({
        'backend': backend, 'feature_version': 'features-v3',
        'metrics': {'test': {'rows': 10}}}), encoding='utf-8')
    return folder


def read(folder):
    return json.loads((Path(folder) / 'manifest.json').read_text(encoding='utf-8'))


def evidence(backend='lightgbm'):
    return {'backend': backend, 'status': 'ok', 'trades': 620, 'net_return': -0.1701,
            'max_drawdown': 0.1759, 'costs_included': True}


def walk_forward(backend='lightgbm'):
    return {'backend': backend, 'folds': 4, 'aggregate': {'net_edge_bps': -10.33,
                                                          'active_samples': 1434},
            'bootstrap': {'prob_positive': 0.22}}


def test_a_candidate_keeps_evidence_computed_from_its_own_predictions(tmp_path):
    lightgbm = manifest(tmp_path, 'lightgbm')
    TrainingRunner._attach_portfolio_evidence(str(lightgbm), evidence('lightgbm'),
                                             walk_forward('lightgbm'), ['passed'])
    stored = read(lightgbm)
    assert stored['metrics']['walk_forward']['backend'] == 'lightgbm'
    assert stored['metrics']['portfolio_oos']['net_return'] == -0.1701


def test_another_models_evidence_is_not_written_as_this_artifacts_result(tmp_path):
    catboost = manifest(tmp_path, 'catboost')
    TrainingRunner._attach_portfolio_evidence(str(catboost), evidence('lightgbm'),
                                             walk_forward('lightgbm'), ['passed'])
    stored = read(catboost)['metrics']
    assert 'walk_forward' not in stored, 'a lightgbm walk-forward is not catboost evidence'
    assert 'portfolio_oos' not in stored
    other = read(catboost).get('external_evidence') or {}
    assert other['walk_forward']['backend'] == 'lightgbm'
    assert other['portfolio_oos']['backend'] == 'lightgbm'
    assert 'not' in other['note'].lower()


def test_evidence_with_no_stated_backend_is_treated_as_foreign(tmp_path):
    """An unattributed result cannot be proven to be this artifact's, so it is not used."""
    catboost = manifest(tmp_path, 'catboost')
    TrainingRunner._attach_portfolio_evidence(str(catboost), {'status': 'ok', 'trades': 5},
                                             {'folds': 1, 'aggregate': {}}, None)
    stored = read(catboost)['metrics']
    assert 'walk_forward' not in stored
    assert 'portfolio_oos' not in stored
