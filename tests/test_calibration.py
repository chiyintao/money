import pytest

from app.models.calibration import brier_score, calibrate, expected_calibration_error, fit_isotonic


def test_isotonic_calibration_is_monotonic_and_bounded():
    calibrator = fit_isotonic([.1, .2, .3, .4], [1, 0, 1, 1])
    values = [calibrate(x, calibrator) for x in [.1, .2, .3, .4]]
    assert values == sorted(values)
    assert all(0 < value < 1 for value in values)


def test_calibration_metrics_are_deterministic():
    probabilities = [.1, .8, .9, .2]
    outcomes = [0, 1, 1, 0]
    assert brier_score(probabilities, outcomes) == pytest.approx(.025)
    assert expected_calibration_error(probabilities, outcomes, bins=2) == pytest.approx(.15)


def test_calibrator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match='calibration_data_too_small'):
        fit_isotonic([.5], [1])
    with pytest.raises(ValueError, match='outcomes_must_be_binary'):
        fit_isotonic([.2, .3], [0, 2])
