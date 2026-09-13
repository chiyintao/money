import json

from app.models.model_registry import ModelRegistry
from app.models.model_runtime import load_calibrated_model
from app.features.feature_spec import FEATURE_VERSION


def calibrator():
    """A minimal isotonic artifact that passes the validator the loader applies."""
    from app.models.calibration import fit_isotonic
    return fit_isotonic([.1, .2, .3, .4, .5, .6], [0, 0, 1, 0, 1, 1])


def test_runtime_requires_calibration_artifact(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'models'))
    registry.register({'status': 'ok', 'model_version': 'v1', 'feature_version':FEATURE_VERSION, 'split':{'ready':True,'label_intervals_verified':True}}, {'test': {'rows': 100, 'directional_accuracy_pct': 55}, 'portfolio_oos': {'costs_included': True, 'trades':40, 'net_return':.02, 'max_drawdown':.05}}, {'sha256': 'a'}, 'v1')
    # Promotion itself refuses an uncalibrated model now; this asserts the loader also
    # refuses one, so a manifest that reached production by another route is still caught.
    registry.promote('v1', allow_calibrator_missing=True)
    assert load_calibrated_model(registry)['reason'] == 'calibrator_missing'


def test_runtime_loads_checked_calibration_artifact(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'models'))
    # The artifact is supplied at register time, which is the only order in which the gate
    # and the loader can be looking at the same file. Writing it after promoting, which is
    # what this test used to do, is exactly the sequence the gate exists to prevent.
    registry.register({'status': 'ok', 'model_version': 'v1', 'feature_version':FEATURE_VERSION, 'split':{'ready':True,'label_intervals_verified':True}}, {'test': {'rows': 100, 'directional_accuracy_pct': 55}, 'portfolio_oos': {'costs_included': True, 'trades':40, 'net_return':.02, 'max_drawdown':.05}, 'calibration': {'fitted': True}}, {'sha256': 'a'}, 'v1', calibrator=calibrator())
    registry.promote('v1')
    assert load_calibrated_model(registry)['status'] == 'active'


def test_the_loader_refuses_an_artifact_the_gate_accepted_only_as_a_flag(tmp_path):
    # Promoted through the escape hatch, so the manifest claims a calibrator and no file
    # exists. The loader must refuse rather than trust the manifest.
    registry = ModelRegistry(str(tmp_path / 'models'))
    registry.register({'status': 'ok', 'model_version': 'v1', 'feature_version':FEATURE_VERSION, 'split':{'ready':True,'label_intervals_verified':True}}, {'test': {'rows': 100, 'directional_accuracy_pct': 55}, 'portfolio_oos': {'costs_included': True, 'trades':40, 'net_return':.02, 'max_drawdown':.05}, 'calibration': {'fitted': True}}, {'sha256': 'a'}, 'v1', calibrator=calibrator())
    registry.promote('v1')
    (tmp_path / 'models' / 'v1' / 'calibration.json').unlink()
    assert load_calibrated_model(registry)['reason'] == 'calibrator_missing'


def test_a_corrupt_artifact_is_refused_rather_than_applied(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'models'))
    registry.register({'status': 'ok', 'model_version': 'v1', 'feature_version':FEATURE_VERSION, 'split':{'ready':True,'label_intervals_verified':True}}, {'test': {'rows': 100, 'directional_accuracy_pct': 55}, 'portfolio_oos': {'costs_included': True, 'trades':40, 'net_return':.02, 'max_drawdown':.05}, 'calibration': {'fitted': True}}, {'sha256': 'a'}, 'v1', calibrator=calibrator())
    registry.promote('v1', allow_calibrator_missing=True)
    (tmp_path / 'models' / 'v1' / 'calibration.json').write_text(json.dumps({'method': 'isotonic', 'blocks': []}))
    assert load_calibrated_model(registry)['reason'] == 'calibrator_invalid'
