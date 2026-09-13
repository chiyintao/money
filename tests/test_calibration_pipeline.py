"""The calibrator had no producer, and the gate and the loader disagreed about it.

Before this: `fit_isotonic` was called only by its own unit test. `ModelRegistry.register`
wrote model.json and manifest.json and nothing else, `promote` required
`metrics.calibration.fitted`, and `model_runtime.load_calibrated_model` required a
calibration.json in the version folder. So the gate could not be passed, and had it been
passed the loader would have refused the result. Both halves now read the same validator
from the same module.
"""
import math
import random
from types import SimpleNamespace

import pytest

from app.models.calibration import SCORE_SCALE, build_calibration, calibrate, calibration_metrics, decision_score, load_calibrator, valid_calibrator
from app.features.feature_spec import FEATURE_VERSION, FEATURES
from app.models.model_registry import ModelRegistry
from app.models.model_runtime import load_calibrated_model


SYMBOLS = ("BTCUSDT", "ETHUSDT")
STEP = 300_000
START = 1_700_000_000_000


def signal_rows(count=900, seed=11, symbols=SYMBOLS):
    """Dataset rows shaped like build_dataset output, with verified label intervals.

    Two symbols, because a portfolio replay refuses a single one: nine correlated legs are
    still a portfolio, one leg is a position.

    The label carries a learnable component, and is drawn from its own generator. Both
    details are load-bearing. On pure noise the four depth-1 stumps predict values inside
    the 4bp cost band, every out-of-sample decision is FLAT, and the replay has nothing to
    replay -- the portfolio block comes back "too_few_symbols_with_decisions:1" and a test
    about whether the gate's inputs exist fails on the fixture rather than on the code.
    And because the features and the label used to share one random stream, adding a
    feature to the contract silently redrew every label in every test that uses this.
    """
    rng = random.Random(seed)
    labels = random.Random(seed + 1)
    rows = []
    for symbol in symbols:
        for index in range(count):
            stamp = START + index * STEP
            features = {name: rng.gauss(0, 1) for name in FEATURES}
            signal = 0.02 * features["ema20_gap"] + 0.015 * features["return_10"]
            rows.append({**features, "symbol": symbol, "timestamp": stamp,
                         "label_end_time": stamp + 12 * STEP,
                         "future_return": signal + labels.gauss(0, 0.008)})
    return rows


def candles(symbol, steps=400, drift=0.0004, seed=1):
    """A bar series that passes the same OHLCV validation the replay applies."""
    rng = random.Random(seed)
    out = []
    price = 100.0
    for index in range(steps):
        opening = price
        closing = price * (1 + drift)
        out.append({"symbol": symbol, "open_time": START + index * STEP,
                    "close_time": START + index * STEP + STEP - 1, "open": opening,
                    "high": max(opening, closing) * 1.001,
                    "low": min(opening, closing) * 0.999, "close": closing,
                    "volume": 10.0, "is_closed": 1})
        price = closing
    return out


def funding_rows(symbols, steps, rate=0.0001, window=8 * 3600 * 1000):
    """Published funding rates, one per settlement window, from the first settlement on.

    Without these the replay reports costs_included False, which is the correct answer and
    also the one that makes every promotion below impossible: a perpetual position pays
    funding for as long as it is held, and evidence that omits it is not evidence that all
    costs are included.
    """
    return {symbol: [(START + (index + 1) * window - 1, rate)
                     for index in range(max(1, (steps * STEP) // window))]
            for symbol in symbols}


def derivative_rows(bars, rate=0.0001, window=8 * 3600 * 1000):
    """Published funding rows for a store, one per symbol per settlement window.

    A promotion decision on a perpetual strategy without funding is incomplete: the position
    pays it for as long as it is held, and the live loop charges it on every tick. The
    evidence block now says so -- costs_included is False when the rates are absent -- so a
    fixture that wants its evidence accepted has to publish them, exactly as the collector
    does.
    """
    rows = []
    for symbol, values in bars.items():
        end = max(int(bar["open_time"]) for bar in values)
        for index in range(max(1, end // window)):
            stamp = (index + 1) * window - 1
            if stamp > end:
                break
            rows.append({"symbol": symbol, "event_time": stamp, "open_interest": 1000.0,
                         "funding_rate": rate, "mark_price": 100.0})
    return rows


def candles_for(rows, drift=0.0004):
    """Bars for every symbol in the rows, covering the whole sampled window.

    The span is taken from the rows rather than fixed. A replay reads candles over the
    test slice's timestamps, so a fixture whose bars stop before that slice produces an
    empty series, and the evidence comes back "portfolio_needs_multiple_symbols:0" -- a
    reason about the data, for a fixture that simply did not reach far enough.
    """
    symbols = sorted({row["symbol"] for row in rows})
    end = max(int(row["timestamp"]) for row in rows)
    steps = (end - START) // STEP + 2
    return {symbol: candles(symbol, steps, drift, seed=index)
            for index, symbol in enumerate(symbols)}


def winning_evidence(drift=0.003, steps=400, cost_bps=4.0):
    """Out-of-sample decisions the replay turns into a positive, well-sampled result.

    A steadily rising series and a long call on every bar past the warmup: the exit policy
    targets +2%, which a 0.3% per-bar drift reaches in about seven bars, so the account
    accumulates the sample the promotion gate asks for. Constructed on purpose -- this is
    the evidence a promotion is supposed to be based on, not a claim that any real model
    would produce it.
    """
    from app.models.portfolio_oos import portfolio_evidence

    rows = [row for index, symbol in enumerate(SYMBOLS)
            for row in candles(symbol, steps, drift, seed=index)]
    oos = []
    for symbol in SYMBOLS:
        for index in range(60, steps):
            oos.append({"timestamp": START + index * STEP, "symbol": symbol,
                        "predicted_return": 0.01, "fold": 0})
    return portfolio_evidence(rows, oos, cost_bps=cost_bps, interval="5m",
                              funding=funding_rows(SYMBOLS, steps))


# ------------------------------------------------------------------ the score
def test_the_runtime_and_the_fitter_share_one_score_definition():
    # A calibrator fit on one scale and applied on another returns numbers that look
    # calibrated. The conversion used to be written out separately at each call site.
    assert decision_score(0.002) == 0.5 + 0.002 * SCORE_SCALE
    import inspect
    import re
    from app.strategy import live_models
    # One implementation of the conversion, and it is the one the fitter used. The old
    # `signal_model` module spelled it out again at its own call site with its own
    # probability gate; nothing in the service called it, so it was a second answer to the
    # question "should this trade" that no test of the live path would ever notice.
    source = inspect.getsource(live_models)
    assert not re.search(r"calibrate\(\s*0\.5\s*\+", source)
    assert not re.search(r"0\.5\s*\+\s*[^\n]*\*\s*10", source)
    assert "decision_score" in source


# ----------------------------------------------------------------- the fitter
def test_a_calibrator_is_fitted_and_described():
    rng = random.Random(3)
    predictions = [rng.gauss(0, 0.004) for _ in range(4000)]
    actuals = [p + rng.gauss(0, 0.01) for p in predictions]
    calibrator, metrics = build_calibration(predictions, actuals)
    assert valid_calibrator(calibrator)
    assert metrics["fitted"] is True
    assert metrics["rows"] + metrics["held_out_rows"] == 4000
    assert calibrator["score_scale"] == SCORE_SCALE
    # The held-out share is what makes this a fact rather than a restatement of the fit:
    # isotonic interpolation reproduces its own sample exactly.
    assert 0.0 <= metrics["ece_after"] <= 1.0
    assert 0.0 <= metrics["brier_after"] <= 1.0
    assert isinstance(metrics["improved"], bool)


def test_a_calibrator_actually_reduces_the_error_on_a_calibratable_sample():
    # Predictions that are systematically overconfident: the raw score says 0.8 when the
    # direction is right 55% of the time. Isotonic has something to correct here.
    rng = random.Random(5)
    predictions = [0.03 + rng.gauss(0, 0.001) for _ in range(3000)]
    actuals = [rng.gauss(0.0005 if rng.random() < 0.55 else -0.0005, 0.01)
               for _ in range(3000)]
    _calibrator, metrics = build_calibration(predictions, actuals)
    assert metrics["fitted"] is True
    assert metrics["ece_after"] < metrics["ece_before"]
    assert metrics["improved"] is True


def test_too_few_rows_is_reported_rather_than_raised():
    # A training run that cannot fit one must still finish, so the refusal names its reason
    # instead of ending in a traceback. A run that silently produced nothing is how this
    # gate stayed unreachable.
    calibrator, metrics = build_calibration([0.001] * 10, [0.002] * 10)
    assert calibrator is None
    assert metrics["fitted"] is False
    assert "calibration_data_too_small" in metrics["reason"]
    assert calibration_metrics([], [])["fitted"] is False


def test_a_zero_prediction_is_not_a_directional_call():
    calibrator, metrics = build_calibration([0.0] * 200 + [0.01] * 200,
                                            [0.01] * 400)
    assert metrics["fitted"] is True
    assert metrics["rows"] + metrics["held_out_rows"] == 200


def test_the_validator_rejects_what_the_loader_would_refuse():
    assert valid_calibrator({"method": "isotonic", "blocks": [{"x_min": 0, "x_max": 1,
                                                              "value": 0.5}]})
    assert not valid_calibrator(None)
    assert not valid_calibrator({})
    assert not valid_calibrator({"method": "platt", "blocks": [{"x_min": 0, "x_max": 1,
                                                               "value": 0.5}]})
    assert not valid_calibrator({"method": "isotonic", "blocks": []})
    assert not valid_calibrator({"method": "isotonic",
                                 "blocks": [{"x_min": 1, "x_max": 0, "value": 0.5}]})
    assert not valid_calibrator({"method": "isotonic", "blocks": [{"x_min": 0}]})


def test_calibrate_applies_the_path_the_fitter_used():
    rng = random.Random(9)
    predictions = [rng.gauss(0, 0.006) for _ in range(3000)]
    actuals = [p + rng.gauss(0, 0.01) for p in predictions]
    calibrator, metrics = build_calibration(predictions, actuals)
    assert metrics["fitted"] is True
    # A larger predicted move must not calibrate to a lower probability: the whole point of
    # an isotonic fit is that the mapping stays monotone in the score it was fit on.
    low = calibrate(decision_score(-0.02), calibrator)
    high = calibrate(decision_score(0.02), calibrator)
    assert high >= low
    assert 0.0 < low <= 1.0 and 0.0 < high <= 1.0


# ------------------------------------------------- where the number ends up
class _Runtime:
    """A model stack that returns exactly the member dicts a test hands it."""

    mode = 'models'
    chronos = None

    def __init__(self, members):
        self.members = dict(members)
        self.load_errors = {}
        # `_version_label` reads `.version` off each entry, so the stub carries one.
        self.models = {name: SimpleNamespace(version=name + '-v1') for name in self.members}

    @property
    def has_members(self):
        return bool(self.members)

    def predict_tabular(self, features):
        return dict(self.members), {}

    def submit_chronos(self, symbol, closes, bar_time):
        return False

    def chronos_vote(self, symbol, bar_time):
        return None

    def status(self):
        return {}


def _engine(members, **kwargs):
    """A decisions object wired to a fixed model stack, for assembling one signal.

    `degraded_blocks` is off because these fixtures carry no derivatives history: the
    degeneracy gate would block every signal before the vote logic under test is reached.
    """
    from app.features.feature_source import FeatureSource
    from app.strategy.live_models import ModelDecision

    kwargs.setdefault('degraded_blocks', False)
    kwargs.setdefault('min_agreement', 0.0)
    kwargs.setdefault('min_edge_bps', 0.0)
    return ModelDecision(_Runtime(members), feature_source=FeatureSource(store=None), **kwargs)


def test_the_calibrated_probability_reaches_the_signal():
    """It used to be computed and thrown away.

    `ProductionMember.predict` calibrated its score, `predict_tabular` copied the value into
    the member dict, and `_assemble` read only `expected_return` and `out_of_range`. The
    signal carried no probability at all, so a service whose entire calibration layer was
    reporting `fitted: True` was serving decisions in which the number played no part and
    could not be seen.
    """
    from app.strategy.live_models import ProductionMember

    model = {'model_version': 'v1', 'target': 'gross_return', 'base_score': 0.004,
             'parameters': {'learning_rate': 0.05, 'cost_bps': 4}, 'stumps': [],
             'normalization': {'mean': [0.0] * len(FEATURES), 'scale': [1.0] * len(FEATURES)}}
    calibrator = {'method': 'isotonic',
                  'blocks': [{'x_min': 0.5, 'x_max': 0.6, 'value': 0.62}]}
    member = ProductionMember(model, calibrator, {'version': 'v1'})
    engine = _engine({'production': member})
    signal = engine._assemble(
        'BTCUSDT', {name: 0.0 for name in FEATURES},
        {'production': {'source': 'production', 'kind': 'production',
                        'expected_return': 0.004, 'side': 'LONG',
                        'probability_up': 0.62, 'cost_bps': 0.0}},
        {}, bar_time=1_700_000_000_000)
    assert signal['probability_up'] == 0.62
    assert signal['calibrated_members'] == 1
    assert signal['uncalibrated_members'] == 0
    # And it survives into the decision log the API serves.
    assert engine.recent[-1]['probability_up'] == 0.62


def test_an_uncalibrated_member_reports_no_probability_rather_than_a_rescaled_return():
    """The tabular family has no calibrator, so it has no probability.

    Its `predict` returns no `probability_up` key at all, which is the honest answer. The
    failure this guards against is the one that used to be in the codebase: inventing one by
    rescaling the predicted return, which produces a number between zero and one that is not
    a probability of anything.
    """
    from app.strategy.live_models import TabularMember

    class Predictor:
        manifest = {'backend': 'lightgbm', 'cost_bps': 4, 'status': 'candidate'}

        def predict(self, features):
            return {'source': 'lightgbm', 'expected_return': 0.01,
                    'expected_net_return': 0.0096, 'side': 'LONG', 'mode': 'paper_decision',
                    'cost_bps': 4}

    member = TabularMember('lightgbm', Predictor(), 'candidates/lightgbm-x')
    engine = _engine({'lightgbm': member})
    signal = engine._assemble(
        'BTCUSDT', {name: 0.0 for name in FEATURES},
        {'lightgbm': {'source': 'lightgbm', 'kind': 'tabular', 'expected_return': 0.01,
                      'side': 'LONG', 'probability_up': None, 'cost_bps': 4.0}},
        {}, bar_time=1_700_000_000_000)
    assert signal['probability_up'] is None
    assert signal['calibrated_members'] == 0
    assert signal['uncalibrated_members'] == 1
    # The signal is still produced: an absent probability is not a blocked trade.
    assert signal['source'] == 'real_models'


def test_the_probability_does_not_silently_become_a_gate():
    """Reported, not gated, and the difference is deliberate.

    Two members vote opposite directions with the same calibrated probability. The live
    gate is sign agreement plus an edge floor; a probability gate would be a third gate,
    and choosing its threshold is a product decision with a measured curve behind it, not
    a side effect of carrying the number.
    """
    from app.strategy.live_models import ProductionMember

    def member(base):
        model = {'model_version': 'v1', 'target': 'gross_return', 'base_score': base,
                 'parameters': {'learning_rate': 0.05, 'cost_bps': 0},
                 'stumps': [],
                 'normalization': {'mean': [0.0] * len(FEATURES),
                                   'scale': [1.0] * len(FEATURES)}}
        calibrator = {'method': 'isotonic',
                      'blocks': [{'x_min': 0.0, 'x_max': 1.0, 'value': 0.99}]}
        return ProductionMember(model, calibrator, {'version': 'v1'})

    up, down = member(0.004), member(-0.004)
    engine = _engine({'up': up, 'down': down})
    features = {name: 0.0 for name in FEATURES}
    signal = engine._assemble('BTCUSDT', features,
                              {'up': up.predict(features), 'down': down.predict(features)},
                              {}, bar_time=1_700_000_000_000)
    assert signal['votes'] == {'up': 0.004, 'down': -0.004}
    assert signal['probability_up'] == 0.99
    assert signal['calibrated_members'] == 2
    # Both agree on the price of being right and disagree on the direction, so the edge is
    # what decides, not the probability.
    assert signal['side'] == 'FLAT'
    assert 'no_directional_edge' in signal['reason_codes']


# --------------------------------------------------------- the stumps producer
def test_fit_advanced_declares_the_feature_version_the_registry_compares_against():
    # The literal "features-v1" outlived the feature set it named, so every artifact this
    # produced was refused for an incompatible feature version it did not have -- while the
    # registry had already been fixed to compare against the constant.
    from app.models.advanced_model import fit_advanced
    result = fit_advanced(signal_rows(600), rounds=2, cost_bps=4, output=None)
    assert result["status"] == "ok"
    assert result["feature_version"] == FEATURE_VERSION


def test_fit_advanced_reports_the_missing_replay_rather_than_omitting_it():
    from app.models.advanced_model import fit_advanced
    result = fit_advanced(signal_rows(600), rounds=2, cost_bps=4, output=None)
    evidence = result["portfolio_oos"]
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "no_store_passed_for_portfolio_replay"
    # And the gate reads that as a refusal, which is what it is.
    from app.models.portfolio_oos import gate_verdict
    assert gate_verdict(evidence)["passes"] is False


def test_fit_advanced_replays_through_a_store_when_given_one(tmp_path):
    from app.models.advanced_model import fit_advanced
    from app.storage.storage import Store

    rows = signal_rows(600)
    store = Store(tmp_path)
    bars = candles_for(rows)
    try:
        for symbol, values in bars.items():
            store.upsert_candles(symbol, "5m", values)
        store.record_derivatives(derivative_rows(bars))
        result = fit_advanced(rows, rounds=2, cost_bps=1, output=None, store=store,
                              interval="5m")
    finally:
        store.close()
    evidence = result["portfolio_oos"]
    assert evidence["status"] != "unavailable"
    assert evidence["costs_included"] is True
    assert "net_return" in evidence and "max_drawdown" in evidence
    assert evidence.get("reason") != "no_store_passed_for_portfolio_replay"


def test_every_gate_input_has_a_producer(tmp_path):
    """The point of the whole round, stated as one assertion.

    Whatever the gate says about a model fitted on noise, it must never again say that an
    input is missing. Before this it always said exactly that: the calibrator had no
    producer, the portfolio block had no producer, and the feature version declared by the
    artifact was a literal the code had stopped emitting.
    """
    from app.models.advanced_model import fit_advanced
    from app.storage.storage import Store

    rows = signal_rows(900)
    store = Store(tmp_path)
    bars = candles_for(rows)
    try:
        for symbol, values in bars.items():
            store.upsert_candles(symbol, "5m", values)
        store.record_derivatives(derivative_rows(bars))
        result = fit_advanced(rows, rounds=4, cost_bps=4, output=None, store=store,
                              interval="5m")
    finally:
        store.close()

    # Each of these is the reason the gate used to refuse, and each now has a real value.
    assert result["feature_version"] == FEATURE_VERSION
    assert result["calibration"]["fitted"] is True
    evidence = result["portfolio_oos"]
    assert evidence["costs_included"] is True
    # Not "unavailable" any more: the replay ran against real candles and the account
    # produced a number. On a model fitted to noise that number is not a good one, which is
    # a different fact from the number not existing -- and the two used to be
    # indistinguishable, because both surfaced as insufficient_portfolio_evidence.
    assert evidence["status"] in ("ok", "no_trades", "insufficient_trades")
    assert evidence.get("reason") != "no_store_passed_for_portfolio_replay"
    assert isinstance(evidence["trades"], int)
    assert math.isfinite(float(evidence["net_return"]))
    assert math.isfinite(float(evidence["max_drawdown"]))

    registry = ModelRegistry(str(tmp_path / "models"))
    registry.register(result,
                      {"test": result["test"], "calibration": result["calibration"],
                       "portfolio_oos": evidence},
                      {"rows": result["rows"]}, version="gbst-v3",
                      calibrator=result["calibrator"])
    with pytest.raises(ValueError) as refusal:
        registry.promote("gbst-v3", min_test_rows=100,
                         min_directional_accuracy_pct=0.0)
    # The gate still refuses -- a model fitted on noise should not reach production -- but
    # it refuses because the account lost money, not because it could not find the ledger.
    assert str(refusal.value) == "insufficient_portfolio_evidence"
    assert float(evidence["net_return"]) <= 0 or int(evidence["trades"]) < 30


def test_a_real_model_with_sufficient_evidence_is_promoted_and_then_loads(tmp_path):
    """The whole path, which had never once run end to end.

    Every gate input now has a producer: the feature version comes from the constant, the
    calibrator is fit on the test slice and written beside the weights, and the portfolio
    evidence is replayed through the same account the service uses. Before this the
    registry could not be promoted into, so data/models stayed empty and the service ran
    unvetted weights for its whole life.
    """
    from app.models.advanced_model import fit_advanced, predict_advanced
    from app.storage.storage import Store

    rows = signal_rows(900)
    store = Store(tmp_path)
    bars = candles_for(rows)
    try:
        for symbol, values in bars.items():
            store.upsert_candles(symbol, "5m", values)
        store.record_derivatives(derivative_rows(bars))
        result = fit_advanced(rows, rounds=4, cost_bps=4, output=None, store=store,
                              interval="5m")
    finally:
        store.close()
    assert result["calibration"]["fitted"] is True

    # Supply portfolio evidence that actually clears the gate. The model fitted above was
    # fitted on pure noise, so its own replay cannot produce a positive return, and no
    # amount of plumbing makes it. What is under test here is the path: evidence in, a
    # promoted and loadable production model out.
    evidence = winning_evidence()
    assert evidence["status"] == "ok" and evidence["trades"] >= 30

    registry = ModelRegistry(str(tmp_path / "models"))
    registry.register(result,
                      {"test": result["test"], "calibration": result["calibration"],
                       "portfolio_oos": evidence},
                      {"rows": result["rows"]}, version="gbst-v3",
                      calibrator=result["calibrator"])
    assert registry.promote("gbst-v3", min_test_rows=100,
                            min_directional_accuracy_pct=0.0) == "gbst-v3"

    bundle = load_calibrated_model(registry)
    assert bundle["status"] == "active", bundle["reason"]
    assert bundle["calibrator"]["method"] == "isotonic"
    raw = predict_advanced(bundle["model"], {name: 0.1 for name in FEATURES})
    probability = calibrate(decision_score(raw["expected_return"]), bundle["calibrator"])
    assert 0.0 < probability < 1.0


def test_the_loader_and_the_gate_read_the_same_file(tmp_path):
    from app.models.calibration import calibration_path

    registry = ModelRegistry(str(tmp_path / "models"))
    registry.register({"feature_version": FEATURE_VERSION}, {"test": {"rows": 1}},
                      {"rows": 1}, "v1")
    # Nothing was registered, so there is nothing to load and load_calibrator says so.
    assert load_calibrator(registry.root, "v1") is None
    assert not calibration_path(registry.root, "v1").exists()
