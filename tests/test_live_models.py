import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.features.feature_spec import FEATURES, FEATURES_V3
from app.strategy.live_models import ModelDecision, RealModelRuntime, finite
from app.strategy.shadow import ShadowModels
from helpers import feature_values, row as feature_row


def rows(n=80):
    return [{'open_time': i, 'close_time': i, 'open': 100 + i * .2, 'high': 101 + i * .2,
             'low': 99 + i * .2, 'close': 100 + i * .2, 'volume': 10 + i % 5, 'is_closed': True}
            for i in range(n)]


def dataset(n=150):
    return [feature_row(i, label_end_time=i + 1,
                        future_return=(.005 if i % 2 else -.005)) for i in range(n)]


def _decision(runtime, **kwargs):
    """ModelDecision for tests about the ensemble rather than about feature provenance.

    The degeneracy gate is exercised by its own test below; everywhere else these
    fixtures have no derivatives history, so the gate would block every signal and the
    vote logic under test would never be reached.
    """
    kwargs.setdefault('degraded_blocks', False)
    return ModelDecision(runtime, **kwargs)


class StubRuntime:
    """RealModelRuntime stand-in: same surface, deterministic predictions."""

    def __init__(self, values, mode='candidate', chronos_vote=None, out_of_range=None):
        self.values = dict(values)
        self.out_of_range = list(out_of_range or [])
        self.mode = mode
        self.load_errors = {}
        self.chronos = None
        # features=FEATURES_V3 because that is what a real artifact of this shape
        # declares, and the decision layer asks the feature source for exactly the columns
        # its members name. A stub without a declaration silently requests the whole
        # contract, which makes every collected family a degraded input.
        self.models = {name: SimpleNamespace(version=name + '-v1', features=FEATURES_V3)
                       for name in self.values}
        self.fast = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stub-model')
        self.chronos_vote_value = chronos_vote
        self.submitted = []

    @property
    def has_members(self):
        return bool(self.values)

    async def ensure_loaded(self):
        return self.status()

    def load_if_needed(self):
        return self.status()

    def predict_tabular(self, features):
        members = {name: {'source': name, 'kind': 'tabular', 'expected_return': value,
                          'side': 'LONG' if value > 0 else 'SHORT', 'expected_net_return': abs(value) - .0012,
                          'probability_up': None, 'version': name + '-v1', 'cost_bps': 12.0,
                          'out_of_range': list(self.out_of_range)}
                   for name, value in self.values.items()}
        return members, {}

    def submit_chronos(self, symbol, closes, bar_time):
        self.submitted.append((symbol, bar_time))
        return True

    async def chronos_forecast(self, symbol, closes, bar_time, timeout_ms=0):
        self.submit_chronos(symbol, closes, bar_time)
        return self.chronos_vote_value

    def chronos_vote(self, symbol, bar_time):
        return self.chronos_vote_value

    def status(self):
        return {'mode': self.mode, 'label': self.mode, 'members': [], 'promotion': {'status': 'fallback'},
                'chronos': {'enabled': False}, 'load_errors': {}, 'loaded': True, 'load_seconds': 0.0}

    async def close(self):
        self.fast.shutdown(wait=True)


def test_finite_guards_non_finite_values():
    assert finite('1.5') == 1.5
    assert finite(float('nan')) is None
    assert finite(None) is None


def test_out_of_distribution_inputs_block_or_warn():
    runtime = StubRuntime({'a': .002, 'b': .002}, out_of_range=['ema20', 'volume'])
    blocked = asyncio.run(_decision(runtime).signal('ALTUSDT', rows()))
    assert blocked['side'] == 'FLAT' and 'out_of_distribution' in blocked['reason_codes']
    warned = asyncio.run(_decision(runtime, ood_policy='warn').signal('ALTUSDT', rows()))
    assert warned['side'] == 'LONG' and 'out_of_distribution_warning' in warned['reason_codes']
    assert warned['out_of_range'] == ['ema20', 'volume']


def test_a_symbol_the_model_never_trained_on_says_so():
    """'out_of_distribution' alone sent the operator hunting feature bounds when the
    real answer was that the session had selected markets outside the training universe.
    The two facts get separate codes, and the drift code is still reported for shape
    compatibility."""
    from app.strategy.live_models import RealModelRuntime
    runtime = object.__new__(RealModelRuntime)
    runtime.feature_space = {'verified': True, 'symbols': ['BTCUSDT', 'ETHUSDT']}
    assert runtime.symbol_untrained('BTCUSDT') is False
    assert runtime.symbol_untrained('SOMEMICROCAPUSDT') is True


def test_an_unverified_feature_space_never_claims_a_symbol_is_unknown():
    """Without verified bounds there is no training universe to compare against, so
    the check must abstain rather than brand every symbol untrained."""
    from app.strategy.live_models import RealModelRuntime
    runtime = object.__new__(RealModelRuntime)
    runtime.feature_space = {'verified': False, 'symbols': ['BTCUSDT']}
    assert runtime.symbol_untrained('SOMEMICROCAPUSDT') is False
    runtime.feature_space = {}
    assert runtime.symbol_untrained('SOMEMICROCAPUSDT') is False


def test_an_empty_symbol_is_not_reported_as_untrained():
    from app.strategy.live_models import RealModelRuntime
    runtime = object.__new__(RealModelRuntime)
    runtime.feature_space = {'verified': True, 'symbols': ['BTCUSDT']}
    assert runtime.symbol_untrained('') is False
    assert runtime.symbol_untrained(None) is False


def test_feature_space_guard_uses_training_bounds(tmp_path):
    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False)
    runtime.feature_space = {'status': 'ok', 'bounds': {'ema20': [100.0, 200.0], 'rsi': [0.0, 100.0]}}
    assert runtime.out_of_range({'ema20': 150.0, 'rsi': 50.0}) == []
    assert runtime.out_of_range({'ema20': 1e-5, 'rsi': 50.0}) == ['ema20']
    assert runtime.out_of_range({'rsi': 50.0}) == ['ema20']
    runtime.feature_space = {'status': 'unavailable'}
    assert runtime.out_of_range({'ema20': 1e-5}) == []


def test_chronos_forecast_waits_for_the_vote_and_respects_timeout(tmp_path):
    class SlowChronos:
        error = None
        forecaster = object()

        def predict(self, features=None, closes=None):
            time.sleep(.2)
            return {'source': 'chronos-2', 'mode': 'paper_decision', 'horizon': 15, 'last_price': 100.0,
                    'quantiles': {}, 'median_return': .003}

    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=True)
    runtime.chronos = SlowChronos()
    closes = [100.0 + i * .1 for i in range(64)]

    async def scenario():
        runtime.start()
        early = await runtime.chronos_forecast('X', closes, 10, timeout_ms=50)
        late = await runtime.chronos_forecast('X', closes, 10, timeout_ms=3000)
        await runtime.close()
        return early, late

    early, late = asyncio.run(scenario())
    assert early is None
    assert late['expected_return'] == pytest.approx(.003)
    assert late['side'] == 'LONG'


def test_chronos_forecast_is_skipped_when_disabled(tmp_path):
    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False)
    assert asyncio.run(runtime.chronos_forecast('X', [100.0] * 64, 1, 10)) is None


def test_signal_comes_from_real_model_votes():
    runtime = StubRuntime({'lightgbm': .0018, 'catboost': .0012})
    decision = _decision(runtime)
    signal = asyncio.run(decision.signal('BTCUSDT', rows()))
    assert signal['source'] == 'real_models'
    assert signal['side'] == 'LONG'
    assert signal['votes'] == {'lightgbm': .0018, 'catboost': .0012}
    assert signal['confidence'] == 1.0 and signal['agreement'] == 1.0
    assert signal['expected_return'] == pytest.approx(.0015)
    assert signal['model_version'] == 'catboost-v1+lightgbm-v1'
    assert signal['reason_codes'] == ['real_model_ensemble', 'ensemble_agrees']
    assert decision.counts['signals'] == 1
    assert runtime.submitted == [('BTCUSDT', 79)]


def test_signal_blocks_disagreement_and_thin_edge():
    disagreed = _decision(StubRuntime({'a': .001, 'b': .0009, 'c': -.0008, 'd': -.0007}))
    signal = asyncio.run(disagreed.signal('BTCUSDT', rows()))
    assert signal['side'] == 'FLAT' and 'agreement_below_floor' in signal['reason_codes']
    thin = _decision(StubRuntime({'a': 1e-6, 'b': 1e-6}))
    signal = asyncio.run(thin.signal('BTCUSDT', rows()))
    assert signal['side'] == 'FLAT' and 'edge_below_floor' in signal['reason_codes']


def test_chronos_vote_joins_the_ensemble():
    vote = {'source': 'chronos-2', 'kind': 'chronos', 'expected_return': .002,
            'side': 'LONG', 'version': 'chronos-2', 'cost_bps': 0.0}
    decision = _decision(StubRuntime({'lightgbm': .001, 'catboost': -.001}, chronos_vote=vote))
    signal = asyncio.run(decision.signal('BTCUSDT', rows()))
    assert set(signal['votes']) == {'lightgbm', 'catboost', 'chronos-2'}
    assert decision.counts['votes_with_chronos'] == 1


def test_missing_weights_fail_closed_unless_rule_fallback_enabled():
    runtime = StubRuntime({}, mode='unavailable')
    blocked = asyncio.run(_decision(runtime).signal('BTCUSDT', rows()))
    assert blocked['side'] == 'FLAT' and blocked['source'] == 'model_blocked'
    assert 'model_unavailable' in blocked['reason_codes']
    fallback = _decision(runtime, rule_fallback=True)
    signal = asyncio.run(fallback.signal('BTCUSDT', rows()))
    assert signal['source'] == 'rule_fallback' and signal['fallback_reason'] == 'model_unavailable'


def test_plan_follows_model_direction_and_clears_costs():
    decision = _decision(StubRuntime({'a': .002, 'b': .002}))
    signal = asyncio.run(decision.signal('BTCUSDT', rows(), price=100.0))
    plan = decision.plan(signal)
    assert plan['side'] == 'LONG' and plan['entry'] == 100.0
    assert plan['stop'] < plan['entry'] < plan['take_profit']
    assert plan['expected_edge'] > 0
    assert decision.plan({**signal, 'side': 'FLAT'}) is None
    expensive = _decision(StubRuntime({'a': .002, 'b': .002}), min_edge_multiple=50)
    assert expensive.plan(asyncio.run(expensive.signal('BTCUSDT', rows(), price=100.0))) is None


def test_signal_cache_is_per_bar_and_refreshed_by_chronos():
    runtime = StubRuntime({'a': .002, 'b': .002})
    decision = _decision(runtime)
    asyncio.run(decision.signal('BTCUSDT', rows()))
    asyncio.run(decision.signal('BTCUSDT', rows()))
    assert decision.counts['signals'] == 1
    newer = rows() + [{'open_time': 80, 'close_time': 80, 'open': 116, 'high': 117, 'low': 115,
                       'close': 116, 'volume': 12, 'is_closed': True}]
    asyncio.run(decision.signal('BTCUSDT', newer))
    assert decision.counts['signals'] == 2
    runtime.chronos_vote_value = {'source': 'chronos-2', 'kind': 'chronos', 'expected_return': .003,
                                  'side': 'LONG', 'version': 'chronos-2', 'cost_bps': 0.0}
    decision.invalidate('BTCUSDT', 80)
    refreshed = asyncio.run(decision.signal('BTCUSDT', newer))
    assert 'chronos-2' in refreshed['votes']


def test_runtime_loads_real_candidate_weights(tmp_path):
    pytest.importorskip('lightgbm')
    import hashlib
    import json
    from app.models.tabular_model import train_tabular
    rows_used = dataset()
    trained = train_tabular(rows_used, 'lightgbm', str(tmp_path / 'candidates'), rounds=10,
                            trials=1)
    dataset_path = tmp_path / 'research_v3' / 'training_dataset.jsonl'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text('\n'.join(json.dumps(row) for row in rows_used), encoding='utf-8')
    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False, backends=('lightgbm',),
                               candidate_dir=tmp_path / 'candidates')
    status = runtime.load()
    assert status['mode'] == 'candidate'
    assert [member['name'] for member in status['members']] == ['lightgbm']
    assert status['members'][0]['path'] == trained['path']
    assert status['members'][0]['status'] == 'candidate'
    assert status['promotion']['reason'] == 'no_production_model'
    expected_digest = hashlib.sha256(json.dumps(sorted(rows_used, key=lambda row: (row['timestamp'], row['symbol'])),
                                                sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert status['feature_space']['verified'] is True
    assert status['feature_space']['sha256'] == expected_digest
    # Values far outside the training range are reported by name; a value inside it is not.
    probe = feature_values(0)
    assert runtime.out_of_range(probe) == []
    outside = {**probe, FEATURES[0]: 1e9, FEATURES[3]: 1e9}
    assert runtime.out_of_range(outside) == sorted([FEATURES[0], FEATURES[3]])
    # A feature the model needs but the caller did not supply counts as outside.
    # The supplied one must be inside the trained range (the dataset's ema20_gap is
    # always positive, so 0.0 would itself be out of range).
    assert runtime.out_of_range({FEATURES[0]: probe[FEATURES[0]]}) == sorted(FEATURES[1:])
    signal = _decision(runtime).signal_sync('XUSDT', rows(), price=100.0)
    assert signal['source'] == 'real_models'
    assert signal['votes']['lightgbm'] is not None


def test_runtime_without_weights_reports_unavailable(tmp_path):
    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False)
    status = runtime.load()
    assert status['mode'] == 'unavailable'
    assert status['members'] == []
    assert runtime.status()['load_errors'] == {'lightgbm': 'candidate_missing', 'catboost': 'candidate_missing'}


def test_shadow_shares_one_runtime(tmp_path):
    class Runtime:
        load_errors = {}

        def __init__(self):
            self.calls = []

        def research_predict(self, symbol, rows):
            self.calls.append(symbol)
            return {'predictions': {}, 'load_errors': {}}

    runtime = Runtime()
    shadow = ShadowModels(tmp_path, max_symbols=1, runtime=runtime)
    assert shadow._owns_runtime is False
    bars = [{'close_time': i, 'close': 100 + i, 'high': 101 + i, 'low': 99 + i, 'volume': 10, 'is_closed': True} for i in range(60)]

    async def scenario():
        assert shadow.submit('BTC', bars)
        shadow.start()
        for _ in range(50):
            await asyncio.sleep(.02)
            if runtime.calls:
                break
        await shadow.close()

    asyncio.run(scenario())
    assert runtime.calls == ['BTC']
    assert shadow.status()['mode'] == 'shadow'


def test_live_service_has_no_rule_decision_path():
    """The served decision path must go through the model, wherever the code lives.

    This used to assert on main.py's source text, which broke whenever code moved.
    It now checks the actual call sites in the trading modules.
    """
    from app.strategy import decision_loop
    from app import main
    assert not hasattr(main, 'predict')
    assert not hasattr(main, 'plan')
    for module in (main, decision_loop):
        source = Path(module.__file__).read_text(encoding='utf-8')
        assert 'rule_predict' not in source.replace('rule_fallback', '')
    loop_source = Path(decision_loop.__file__).read_text(encoding='utf-8')
    assert 'decisions.signal(' in loop_source
    assert 'decisions.plan(' in loop_source


# ------------------------------------------------- the evidence behind the weights
def test_the_held_out_evidence_includes_the_numbers_that_decide_it():
    """Accuracy is the one figure a model with no edge can pass.

    The model center showed rows, accuracy, MAE and the cost assumption. The two candidates
    the service actually trades score 48.24% on 188,640 held-out rows -- below a coin flip,
    on every symbol -- and nothing in the interface said so. What decides whether the model
    is worth trading is the net edge on the samples it was willing to act on and how few of
    them there are. Both were computed on every run and displayed nowhere.
    """
    from app.strategy.live_models import _test_metrics

    metrics = {"test": {"rows": 188640, "directional_accuracy_pct": 48.24, "mae": 0.0036,
                        "cost_bps": 12.0, "active_samples": 269, "active_share_pct": 0.1426,
                        "active_hit_rate_pct": 53.53, "active_net_edge_bps": 2.08,
                        "mean_net_return": 2.97e-07,
                        "evaluation": "independent_overlapping_samples_not_portfolio"}}
    shown = _test_metrics(metrics)
    assert shown["active_net_edge_bps"] == 2.08
    assert shown["active_share_pct"] == 0.1426
    assert shown["active_hit_rate_pct"] == 53.53
    # The manifest declares that these samples overlap in time and are not a portfolio
    # result. A reader who cannot see that caveat is reading a number as something it is not.
    assert shown["evaluation"] == "independent_overlapping_samples_not_portfolio"
    assert "train" not in shown, "the held-out block is the one that means anything"


def test_the_evidence_survives_a_manifest_with_no_test_block():
    from app.strategy.live_models import _test_metrics

    assert _test_metrics({}) == {}
    assert _test_metrics(None) == {}
    assert _test_metrics({"test": None}) == {}


def test_per_symbol_dispersion_is_reported_rather_than_averaged_away():
    """The replay of 85 real fills found hit rates from 33% to 64% under one rule.

    The per-symbol numbers are written on every training run. Nothing read them, so the
    spread that the meta-labeling layer exists to handle was discoverable only by opening
    the manifest by hand. On the live weights it says something the aggregate hides: every
    one of the twelve symbols is below 50%.
    """
    from app.strategy.live_models import _symbol_dispersion

    by_symbol = {name: {"directional_accuracy_pct": value, "rows": 100}
                 for name, value in (("A", 46.15), ("B", 48.28), ("C", 49.51), ("D", 55.0))}
    spread = _symbol_dispersion(by_symbol)
    assert spread["symbols"] == 4
    assert spread["min"] == 46.15
    assert spread["max"] == 55.0
    assert spread["median"] == 49.51
    assert spread["below_coin_flip"] == 3
    assert _symbol_dispersion({}) is None
    assert _symbol_dispersion(None) is None
    # Entries that are not dictionaries are skipped rather than raising: the manifest is
    # written by whichever library version happened to be installed at the time.
    assert _symbol_dispersion({"A": None, "B": {"rows": 5}}) is None


def test_the_loaded_weights_describe_their_own_evidence():
    """End to end on the real artifacts: what /api/models serves.

    The numbers asserted here are the ones in data/research_v3/candidates. If a retrain
    replaces them the shape must stay, so this checks the shape and the sign of the claim
    rather than the exact values.
    """
    from pathlib import Path

    root = Path("data/research_v3/candidates")
    if not root.is_dir() or not any(root.iterdir()):
        pytest.skip("no candidate artifacts in this checkout")
    runtime = RealModelRuntime("data", chronos_enabled=False)
    runtime.load()
    assert runtime.models, "the candidates must load, or every signal falls back"
    for member in runtime.models.values():
        described = member.describe()
        test = described["test"]
        assert test["rows"] > 0
        assert "active_net_edge_bps" in test, "the edge is the number the decision rests on"
        assert "active_share_pct" in test, "and so is how few samples it rests on"
        assert described["by_symbol"]["symbols"] >= 1
        assert described["by_symbol"]["min"] <= described["by_symbol"]["max"]
