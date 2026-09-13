import pytest
from app.models.model_registry import ModelRegistry
from app.features.feature_spec import FEATURE_VERSION


def test_accuracy_alone_cannot_promote(tmp_path):
    registry=ModelRegistry(tmp_path)
    model={'feature_version':FEATURE_VERSION,'split':{'ready':True,'label_intervals_verified':True}}
    registry.register(model,{'test':{'rows':1000,'directional_accuracy_pct':99}}, {}, 'v1')
    with pytest.raises(ValueError,match='portfolio_evidence'):
        registry.promote('v1')
    assert registry.current() is None


def test_unverified_labels_cannot_promote(tmp_path):
    registry=ModelRegistry(tmp_path)
    registry.register({}, {'test':{'rows':1000,'directional_accuracy_pct':99}}, {}, 'v1')
    with pytest.raises(ValueError,match='label_intervals'):
        registry.promote('v1')
