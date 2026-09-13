"""The gates that decide whether a model or a symbol is allowed to trade.

Both were structurally incapable of refusing anything: the promotion gate demanded a
feature version the code no longer produced, and the per-symbol edge gate tested the sign
of a 30-sample mean with no dispersion term. Neither failure produced an error, a warning
or an anomaly in any report -- just trades.
"""
import math
import random

import pytest

from app.models.calibration import fit_isotonic
from app.features.feature_spec import FEATURE_VERSION
from app.models.model_registry import ModelRegistry
from app.strategy.symbol_edge import SymbolEdge, horizon_analysis


# --------------------------------------------------------------- promotion gate
def valid_calibrator():
    """A calibrator artifact the loader would accept.

    Registering a manifest whose calibration flag is set while no artifact exists is the
    state these tests used to assert was promotable. It was not: the flag said the model
    was calibrated and the version folder held no calibrator, so the promotion succeeded
    and the runtime fell back at load time with nothing reporting why. A real artifact is
    now required, and this is the smallest one that passes the same validator the loader
    uses.
    """
    return fit_isotonic([.1, .2, .3, .4, .5, .6], [0, 0, 1, 0, 1, 1])


def _register(registry, metrics=None, version='v1', feature_version=FEATURE_VERSION,
              calibrator=None):
    model = {'coef': [1], 'feature_version': feature_version,
             'split': {'ready': True, 'label_intervals_verified': True}}
    base = {'test': {'rows': 1000, 'directional_accuracy_pct': 55},
            'portfolio_oos': {'costs_included': True, 'trades': 40,
                              'net_return': .02, 'max_drawdown': .05},
            'calibration': {'fitted': True}}
    base.update(metrics or {})
    supplied = valid_calibrator() if calibrator is None else calibrator
    if (base.get('calibration') or {}).get('fitted') and supplied is not None:
        registry.register(model, base, {'rows': 300}, version, calibrator=supplied)
    else:
        registry.register(model, base, {'rows': 300}, version)
    return registry


def test_a_model_built_from_the_current_features_can_be_promoted(tmp_path):
    # The gate compared against the literal 'features-v1' while feature_spec produced
    # 'features-v3', so it could never pass. data/models was empty and the service ran
    # unpromoted candidate weights for its entire life.
    registry = _register(ModelRegistry(str(tmp_path / 'models')))
    assert registry.promote('v1') == 'v1'
    assert registry.current()['status'] == 'active'


def test_a_model_from_a_stale_feature_version_is_still_refused(tmp_path):
    registry = _register(ModelRegistry(str(tmp_path / 'models')), feature_version='features-v1')
    with pytest.raises(ValueError, match='incompatible_feature_version'):
        registry.promote('v1')


def test_an_uncalibrated_model_cannot_be_promoted(tmp_path):
    # Probabilities were produced by 0.5 + expected_return * 10, a linear rescaling with
    # no defined interpretation, because nothing ever fit or shipped a calibrator.
    registry = _register(ModelRegistry(str(tmp_path / 'models')),
                         metrics={'calibration': {'fitted': False}}, calibrator=None)
    with pytest.raises(ValueError, match='calibrator_missing'):
        registry.promote('v1')
    assert registry.promote('v1', allow_calibrator_missing=True) == 'v1'


def test_a_calibration_flag_without_an_artifact_is_refused_at_register(tmp_path):
    # The other half of the same requirement. The gate reads a flag in the manifest and
    # the loader reads a file in the version folder; nothing checked that they agreed.
    registry = ModelRegistry(str(tmp_path / 'models'))
    with pytest.raises(ValueError, match='calibration_artifact_missing'):
        registry.register({'feature_version': FEATURE_VERSION},
                          {'calibration': {'fitted': True}}, {'rows': 1}, 'v1',
                          calibrator=None)


def test_deleting_the_artifact_after_registering_stops_the_promotion(tmp_path):
    from app.models.calibration import calibration_path

    registry = _register(ModelRegistry(str(tmp_path / 'models')))
    calibration_path(registry.root, 'v1').unlink()
    with pytest.raises(ValueError, match='calibrator_artifact_missing'):
        registry.promote('v1')


def test_an_invalid_artifact_is_refused_at_register(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'models'))
    with pytest.raises(ValueError, match='calibration_artifact_invalid'):
        registry.register({'feature_version': FEATURE_VERSION}, {'test': {'rows': 1}},
                          {'rows': 1}, 'v1',
                          calibrator={'method': 'platt', 'blocks': []})


def test_evidence_gates_still_apply_in_order(tmp_path):
    registry = _register(ModelRegistry(str(tmp_path / 'models')),
                         metrics={'test': {'rows': 10, 'directional_accuracy_pct': 55}})
    with pytest.raises(ValueError, match='insufficient_test_rows'):
        registry.promote('v1')
    # And without the portfolio replay the gate is unreachable for a different reason.
    registry = _register(ModelRegistry(str(tmp_path / 'models2')),
                         metrics={'portfolio_oos': {'status': 'unavailable'}})
    with pytest.raises(ValueError, match='insufficient_portfolio_evidence'):
        registry.promote('v1')


# ------------------------------------------------------------- edge significance
def _tracker(values, **kwargs):
    tracker = SymbolEdge(min_samples=30, enabled=True, round_trip_cost=0.0012, **kwargs)
    tracker.scored['XUSDT'] = list(values)
    return tracker


def test_a_noisy_positive_mean_does_not_earn_the_right_to_trade():
    # Thirty draws from a zero-mean distribution with a realistic spread: the old gate
    # asked only whether the mean was above zero, which is a coin flip.
    rng = random.Random(7)
    values = [rng.gauss(2.0, 40.0) for _ in range(30)]
    mean = sum(values) / len(values)
    assert mean > 0, 'the fixture must actually be a positive sample'
    assert _tracker(values).allows('XUSDT')[0] is False
    assert _tracker(values).allows('XUSDT')[1] == 'symbol_edge_not_significant'


def test_a_large_consistent_edge_does_earn_it():
    rng = random.Random(11)
    values = [rng.gauss(25.0, 10.0) for _ in range(120)]
    allowed, reason = _tracker(values).allows('XUSDT')
    assert allowed is True and reason == 'symbol_edge_ok'


def test_the_floor_still_applies_before_significance():
    rng = random.Random(3)
    values = [rng.gauss(-25.0, 10.0) for _ in range(120)]
    allowed, reason = _tracker(values, min_net_bps=0.0).allows('XUSDT')
    assert allowed is False and reason == 'symbol_edge_below_floor'


def test_statistics_report_the_dispersion_the_old_gate_ignored():
    stats = _tracker([10.0, 20.0, 30.0]).statistics('XUSDT')
    assert stats['count'] == 3
    assert stats['mean'] == pytest.approx(20.0)
    assert stats['stderr'] == pytest.approx(10.0 / math.sqrt(3))


def test_a_flat_series_has_no_defined_t_statistic_rather_than_an_infinite_one():
    stats = _tracker([5.0] * 40).statistics('XUSDT')
    assert stats['stderr'] == 0.0
    assert stats['t'] is None


def test_the_fdr_correction_removes_the_lucky_ones():
    # Two hundred symbols, all pure noise with a small positive drift. With one test per
    # symbol and no correction, a handful always pass. With it, essentially none should.
    rng = random.Random(5)
    tracker = SymbolEdge(min_samples=30, enabled=True, round_trip_cost=0.0, min_t=0.0, fdr=0.10)
    for index in range(200):
        tracker.scored['S%03dUSDT' % index] = [rng.gauss(0.5, 20.0) for _ in range(40)]
    without = SymbolEdge(min_samples=30, enabled=True, round_trip_cost=0.0, min_t=0.0, fdr=0.0)
    without.scored = {symbol: list(series) for symbol, series in tracker.scored.items()}
    naive = sum(1 for symbol in tracker.scored if without.allows(symbol)[0])
    corrected = sum(1 for symbol in tracker.scored if tracker.allows(symbol)[0])
    assert naive > 20, 'the uncorrected gate is expected to promote a crowd of noise'
    assert corrected < naive / 4


def test_q_values_are_monotone_in_the_ranking():
    rng = random.Random(13)
    tracker = SymbolEdge(min_samples=10, enabled=True, round_trip_cost=0.0)
    for index in range(30):
        # A real spread per symbol: a constant series has no standard error and so has no
        # p-value to correct.
        tracker.scored['S%02dUSDT' % index] = [10.0 + index * 0.5 + rng.gauss(0, 5) for _ in range(12)]
    q = tracker.q_values()
    assert q and all(0.0 <= value <= 1.0 for value in q.values())
    strongest = max(tracker.scored, key=lambda s: tracker.statistics(s)['mean'])
    assert q[strongest] == min(q.values())


# ----------------------------------------------------------- horizon selection
class _Row(dict):
    pass


class _Store:
    """Minimal store surface: horizon_analysis only needs decisions and candles."""

    def __init__(self, decisions, bars):
        self._decisions = decisions
        self._bars = bars

    def db(self):
        return self

    def execute(self, sql, params=()):
        return self

    def fetchall(self):
        return []


def _noise_bars(count=400, seed=1):
    rng = random.Random(seed)
    price = 100.0
    bars = []
    for index in range(count):
        price *= (1 + rng.gauss(0, 0.004))
        bars.append({'open_time': index * 300_000, 'close_time': index * 300_000 + 299_999,
                     'open': price, 'high': price * 1.001, 'low': price * 0.999,
                     'close': price, 'volume': 10.0, 'is_closed': True})
    return bars


def test_horizon_analysis_reports_whether_anything_cleared_its_error_bar(tmp_path):
    # Several overlapping horizons with argmax over them: the maximum of several noisy
    # means is positive even when none of them is real. The selected horizon must
    # therefore be justified by a one-sided lower bound, not by the point estimate.
    from app.core.domain import Event
    from app.storage.storage import Store

    bars = _noise_bars()
    store = Store(tmp_path)
    try:
        store.upsert_candles('BTCUSDT', '5m', bars)
        for index in range(60, len(bars) - 10):
            bar = bars[index]
            store.record_event(Event('strategy_decision', {
                'symbol': 'BTCUSDT',
                'decision': {'side': 'LONG' if index % 2 else 'SHORT', 'bar_time': bar['open_time']},
                'market': {'price': bar['close'], 'bar_time': bar['open_time']},
            }).json())
        result = horizon_analysis(store, '5m', horizons=(1, 3, 6), cost=0.0012, min_samples=10)
    finally:
        store.close()
    assert set(result) >= {'pooled', 'best_horizon', 'selection', 'positive_but_unproven'}
    for horizon, stats in result['pooled'].items():
        if stats['n'] > 1:
            assert 'lower_bound_bps' in stats and 'stderr_bps' in stats
            assert stats['lower_bound_bps'] <= stats['mean_net_bps']
    if result['best_horizon'] is not None:
        chosen = result['pooled'][result['best_horizon']]
        assert chosen['lower_bound_bps'] > 0



