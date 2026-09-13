"""Real trained-model decision path for the paper trading service.

Paper trading used to trade a hand-written EMA/RSI rule. This module routes every
simulated decision through the real trained artifacts that already exist in the
workspace:

* a promoted, isotonic-calibrated production model from data/models when one
  passed the registry gates, otherwise
* the newest real LightGBM / CatBoost artifacts under
  data/research_v3/candidates, plus
* an optional Chronos-2 quantile-forecast vote (data/pretrained/chronos-2).

Inference never runs on the asyncio event loop, and nothing here submits orders:
signals carry explicit provenance (artifact path, checksum, promotion state) and
the risk engine, market guards, portfolio limits and the paper broker still gate
every entry. When no real weight can be loaded the decision service fails closed
(FLAT) unless rule fallback is explicitly enabled in the environment.
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core.domain import finite
from ..features.feature_source import FUNDING_FEATURES, FeatureSource
from ..features.features import snapshot
from ..models.dataset_io import cached_profile
from ..models.fit_model import FEATURES
from ..trading.strategy import predict as rule_predict
from ..trading import exit_policy

TABULAR_BACKENDS = ('lightgbm', 'catboost')
MODE_LABELS = {
    'production': '生产模型（已通过注册表门禁）',
    'candidate': '真实候选权重（未通过晋级门禁）',
    'unavailable': '真实模型不可用',
    'loading': '真实模型加载中',
}


def bar_id(row):
    """Canonical identity of a closed bar, shared by every model cache."""
    return int(row.get('close_time') or row.get('open_time') or 0)


def _test_metrics(metrics):
    if not isinstance(metrics, dict):
        return {}
    test = metrics.get('test') or {}
    # The held-out numbers, including the ones that say whether it makes money. Accuracy is
    # the one figure a model with no edge can pass: the two candidates the service actually
    # trades score 48.24% on 188,640 held-out rows, below a coin flip, and nothing in the
    # model center said so. What decides the question is the net edge on the samples the
    # model was willing to act on, and how few of them there are -- 269 and 480 out of
    # 188,640, which is the difference between a strategy and a rounding error.
    keys = ('rows', 'directional_accuracy_pct', 'mae', 'cost_bps', 'active_samples',
            'active_share_pct', 'active_hit_rate_pct', 'active_net_edge_bps',
            'mean_net_return', 'evaluation')
    return {key: test.get(key) for key in keys if key in test}


def _symbol_dispersion(by_symbol):
    """Where per-symbol accuracy actually spread.

    The replay of 85 real fills found that the aggregate hit rate hides the thing that
    matters -- some symbols at 64%, others at 33%, traded by one rule with one size. The
    per-symbol numbers are computed on every training run and shown nowhere, so the spread
    was only discoverable by reading the manifest by hand.
    """
    values = [float(row['directional_accuracy_pct']) for row in (by_symbol or {}).values()
              if isinstance(row, dict) and row.get('directional_accuracy_pct') is not None]
    if not values:
        return None
    values.sort()
    return {'symbols': len(values), 'min': round(values[0], 4),
            'median': round(values[len(values) // 2], 4), 'max': round(values[-1], 4),
            'below_coin_flip': sum(1 for value in values if value < 50.0)}


class ProductionMember:
    """Promoted registry artifact (gradient boosted stumps) with its calibrator."""

    kind = 'production'

    def __init__(self, model, calibrator, manifest, folder=''):
        self.model = dict(model or {})
        self.calibrator = calibrator
        self.manifest = dict(manifest or {})
        self.name = 'production'
        self.version = str(self.manifest.get('version') or self.model.get('model_version') or 'production')
        self.status = str(self.manifest.get('status') or 'production')
        self.sha256 = self.manifest.get('sha256')
        self.path = str(folder or self.manifest.get('path') or '')
        self.cost_bps = float(((self.model.get('parameters') or {}).get('cost_bps')) or 0.0)
        self.created_at = int(self.manifest.get('created_at') or 0)
        self.metrics = self.manifest.get('metrics') or {}
        self.features = tuple(self.model.get('features')
                              or self.manifest.get('features') or ())

    def predict(self, features, closes=None):
        from ..models.advanced_model import predict_advanced
        raw = predict_advanced(self.model, features)
        probability = None
        if self.calibrator:
            try:
                # The same score definition the fitter used. A calibrator fit on one scale
                # and applied on another returns numbers that look calibrated and are not.
                from ..models.calibration import calibrate, decision_score
                probability = calibrate(decision_score(raw['expected_return']), self.calibrator)
            except (ValueError, KeyError, TypeError):
                probability = None
        return {'source': 'production', 'model': self.version, 'side': raw['side'],
                'expected_return': raw['expected_return'], 'expected_net_return': raw['expected_net_return'],
                'probability_up': probability, 'mode': 'paper_decision'}

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'version': self.version, 'status': self.status,
                'path': self.path, 'sha256': self.sha256, 'cost_bps': self.cost_bps,
                'created_at': self.created_at, 'test': _test_metrics(self.metrics)}


class TabularMember:
    """Native LightGBM / CatBoost candidate artifact loaded from disk."""

    kind = 'tabular'

    def __init__(self, name, predictor, folder):
        self.name = name
        self.predictor = predictor
        meta = predictor.manifest
        self.manifest = meta
        self.version = '%s-%s' % (meta.get('backend', name), Path(folder).name.rsplit('-', 1)[-1][:12])
        self.status = str(meta.get('status') or 'candidate')
        self.sha256 = meta.get('sha256')
        self.path = str(folder)
        self.cost_bps = float(meta.get('cost_bps') or 0.0)
        self.created_at = int(meta.get('created_at') or 0)
        self.metrics = meta.get('metrics') or {}
        self.metrics_by_symbol = meta.get('metrics_by_symbol') or {}
        # What this artifact was actually fit on. Serving reads this rather than the
        # contract's current list, so a feature addition does not turn every loaded model
        # into a degraded input.
        self.features = tuple(getattr(predictor, 'features', ()) or ())
        # The walk-forward verdict is why a run did not promote, and it lived only in the
        # training job's in-memory record. Written into the manifest it becomes a property
        # of the artifact, which is where a reader of the model center can see it.
        self.walk_forward = (self.metrics.get('walk_forward')
                             or meta.get('walk_forward') or None)

    def predict(self, features, closes=None):
        return self.predictor.predict(features)

    def describe(self):
        return {'name': self.name, 'kind': self.kind, 'version': self.version, 'status': self.status,
                'path': self.path, 'sha256': self.sha256, 'cost_bps': self.cost_bps,
                'created_at': self.created_at, 'test': _test_metrics(self.metrics),
                'by_symbol': _symbol_dispersion(self.metrics_by_symbol),
                'walk_forward': self.walk_forward}


class ChronosMember:
    """Chronos-2 quantile forecaster; heavy weights are loaded lazily, once."""

    kind = 'chronos'

    def __init__(self, model_path, device='cpu', horizon=15, mode='paper_decision'):
        self.path = str(model_path)
        self.device = device
        self.horizon = int(horizon)
        self.mode = mode
        self.forecaster = None
        self.error = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.forecaster is not None:
                return self.forecaster
            if self.error is not None:
                raise RuntimeError(self.error)
            try:
                from ..models.chronos_model import ChronosForecaster
                self.forecaster = ChronosForecaster(self.path, self.device, mode=self.mode)
            except Exception as exc:  # weights missing or torch/chronos unavailable
                self.error = repr(exc)
                raise
            return self.forecaster

    def predict(self, features=None, closes=None):
        values = [finite(value) for value in (closes or [])]
        prices = [value for value in values if value is not None and value > 0]
        if len(prices) != len(values) or len(prices) < 32:
            raise ValueError('invalid_forecast_window')
        forecaster = self.load()
        with self._lock:
            return forecaster.forecast(prices, self.horizon)


class RealModelRuntime:
    """Owns the real model weights and serializes all inference off the event loop."""

    def __init__(self, data_dir='data', device='cpu', chronos_enabled=True, backends=TABULAR_BACKENDS,
                 candidate_dir=None, registry_root=None, max_chronos_symbols=8, chronos_horizon=15,
                 allow_candidate_fallback=False):
        self.root = Path(data_dir)
        self.device = device
        self.chronos_enabled = bool(chronos_enabled)
        self.backends = tuple(backends)
        self.candidate_dir = Path(candidate_dir) if candidate_dir else self.root / 'research_v3' / 'candidates'
        self.registry_root = Path(registry_root) if registry_root else self.root / 'models'
        self.pretrained_dir = self.root / 'pretrained' / 'chronos-2'
        self.max_chronos_symbols = int(max_chronos_symbols)
        self.chronos_horizon = int(chronos_horizon)
        # Off by default. An operator who wants unvetted weights when a promoted model has
        # gone stale can say so; nobody should get them by accident.
        self.allow_candidate_fallback = bool(allow_candidate_fallback)
        self.models = {}
        self.load_errors = {}
        self.promotion = {'status': 'unknown'}
        self.mode = 'loading'
        self.loaded = False
        self.load_seconds = 0.0
        self.loaded_at = 0
        self.feature_space = {}
        # The research surface computes its own features. It used to call a _features method
        # that lives on ModelDecision, which this class is not, so every shadow prediction
        # raised AttributeError("'RealModelRuntime' object has no attribute '_features'") and
        # the one surface that would have shown the models disagreeing showed five errors
        # instead. The trading path was unaffected -- it goes through ModelDecision, which
        # has the method -- which is exactly why the failure went unnoticed: the dashboard
        # looked broken rather than the runtime.
        self.feature_source = FeatureSource()
        self._required = None
        self.chronos = ChronosMember(self.pretrained_dir, device, self.chronos_horizon) if self.chronos_enabled else None
        self.chronos_results = {}
        self.chronos_pending = {}
        self.chronos_seen = {}
        self.chronos_event = asyncio.Event()
        self.chronos_task = None
        self.on_chronos_result = None
        self.stopping = False
        self.fast = ThreadPoolExecutor(max_workers=2, thread_name_prefix='model-fast')
        self.slow = ThreadPoolExecutor(max_workers=1, thread_name_prefix='model-chronos')
        self._load_lock = asyncio.Lock()

    # ---------------------------------------------------------------- loading
    @property
    def has_members(self):
        return bool(self.models)

    def load(self):
        """Blocking load; runs inside the fast pool."""
        started = time.monotonic()
        self.models = {}
        self.load_errors = {}
        self._load_production()
        promoted = bool(self.models)
        # The candidate fallback exists for one situation: the registry holds no promoted
        # model at all, and unvetted weights are better than no decision. It was applied to
        # every situation. A production model that is present but expired, missing its
        # calibrator, or unreadable took the same path and was silently replaced by the
        # very weights the promotion gate refused -- a downgrade in provenance that looks
        # from the outside exactly like a healthy service. When production exists and
        # cannot be used, nothing is loaded and the decision layer refuses.
        kind = (self.promotion or {}).get('kind')
        if not promoted and kind == 'unusable' and not self.allow_candidate_fallback:
            self.mode = 'unavailable'
            self.load_errors['production'] = 'production_unusable:%s' % (
                (self.promotion or {}).get('reason') or 'unknown')
        else:
            if not promoted:
                self._load_candidates()
            self.mode = ('production' if promoted
                         else 'candidate' if self.models else 'unavailable')
        if self.models and not promoted:
            # Running on an unpromoted candidate was reported only as mode == 'candidate',
            # which reads like a normal state. It is not: no candidate in this repository
            # has ever satisfied the promotion gate, so every number the service produced
            # came from weights that failed review. The reason is surfaced as a warning
            # rather than left for the operator to infer from a mode string.
            reason = (self.promotion or {}).get('reason') or (self.promotion or {}).get('status')
            self.load_errors['promotion'] = 'running_unpromoted_candidate:%s' % reason
        self.feature_space = self._load_feature_space() if self.models else {}
        self.load_seconds = time.monotonic() - started
        self.loaded_at = int(time.time() * 1000)
        self.loaded = True
        return self.status()

    def load_if_needed(self):
        if not self.loaded:
            self.load()
        return self.status()

    def unavailable_reason(self):
        """Why no weights are loaded, named as precisely as the loader knows.

        "model_unavailable" is true of an empty registry, a stale production model and a
        corrupt one alike, and the operator response is different in each case: train one,
        retrain one, or fix a file. The specific code is carried into the blocked signal so
        every FLAT decision in the audit trail says which.
        """
        production = self.load_errors.get('production')
        if production:
            return str(production)
        if self.load_errors:
            name = sorted(self.load_errors)[0]
            return 'model_unavailable:%s' % name
        return 'model_unavailable'

    async def ensure_loaded(self):
        if self.loaded:
            return self.status()
        async with self._load_lock:
            if self.loaded:
                return self.status()
            try:
                await asyncio.get_running_loop().run_in_executor(self.fast, self.load)
            except Exception as exc:
                self.load_errors['runtime'] = repr(exc)
                self.mode = 'unavailable'
                self.loaded = True
        return self.status()

    def _load_production(self):
        from ..models.model_registry import ModelRegistry
        from ..models.model_runtime import load_calibrated_model
        try:
            registry = ModelRegistry(str(self.registry_root))
            bundle = load_calibrated_model(registry)
        except Exception as exc:
            self.promotion = {'status': 'error', 'reason': repr(exc), 'kind': 'unusable'}
            return
        if bundle.get('status') != 'active':
            # The kind travels with the verdict: load() decides from it whether unvetted
            # candidates may stand in, and dropping it here sent every failure down the
            # same path regardless of what had actually gone wrong.
            self.promotion = {'status': bundle.get('status'), 'reason': bundle.get('reason'),
                              'kind': bundle.get('kind') or 'unusable'}
            return
        manifest = bundle.get('manifest') or {}
        folder = str(self.registry_root / str(manifest.get('version') or ''))
        member = ProductionMember(bundle['model'], bundle['calibrator'], manifest, folder)
        self.models['production'] = member
        self.promotion = {'status': 'active', 'kind': 'active', 'version': member.version,
                          'path': member.path, 'sha256': member.sha256,
                          'created_at': member.created_at}

    def _load_candidates(self):
        from ..models.tabular_model import TabularPredictor
        for backend in self.backends:
            folders = sorted(self.candidate_dir.glob(backend + '-*'), key=lambda path: path.stat().st_mtime, reverse=True)
            if not folders:
                self.load_errors[backend] = 'candidate_missing'
                continue
            for folder in folders:
                try:
                    predictor = TabularPredictor(folder, mode='paper_decision')
                except Exception as exc:
                    self.load_errors[backend] = '%s: %r' % (folder.name, exc)
                    continue
                self.models[backend] = TabularMember(backend, predictor, folder)
                break

    def _load_feature_space(self):
        """Training feature ranges, verified against the candidate dataset hash."""
        # The dataset the models were actually trained from, found by digest.
        #
        # This used to be the fixed name training_dataset.jsonl. Training writes one
        # dataset per tier -- training_dataset_mainstream.jsonl and its speculative
        # sibling -- so once a run used a tier dataset the stored digest could never
        # equal the one in the manifest. verified went false, trained_market returned
        # no symbols at all, the session selected an empty universe, and nothing was
        # ever traded. The failure reported itself as 'the model was not trained on
        # this market' while the models were trained on it exactly.
        expected = {member.manifest.get('dataset_sha256') for member in self.models.values()}
        expected.discard(None)
        root = self.root / 'research_v3'
        candidates = sorted(root.glob('training_dataset*.jsonl')) if root.is_dir() else []
        path = None
        profile = None
        for candidate in candidates:
            try:
                found = cached_profile(candidate, FEATURES)
            except (OSError, ValueError, KeyError):
                continue
            if not expected or found['sha256'] in expected:
                path, profile = candidate, found
                break
        if profile is None:
            return {'status': 'unavailable', 'reason': 'training_dataset_missing'}
        try:
            # Streaming with a cache: a pooled multi-symbol dataset is far too large to
            # load, sort and re-serialize in memory on every start, and re-hashing it
            # costs ~30s. The digest value is unchanged either way.
            profile = cached_profile(path, FEATURES)
        except (OSError, ValueError, KeyError) as exc:
            return {'status': 'unavailable', 'reason': repr(exc)}
        bounds = profile['bounds']
        digest = profile['sha256']
        # Whether these bounds can refuse anything. The stored bounds were the observed
        # extremes of already-clipped features, so for four of the ten they equal the clip
        # range and the gate below is unreachable for them. Recorded on the feature space
        # so /health and the model panel can say so instead of reporting a green gate.
        from ..models.dataset_io import bounds_are_informative

        quality = bounds_are_informative(bounds, FEATURES)
        return {'status': 'ok', 'path': str(path), 'rows': profile['rows'], 'sha256': digest,
                'verified': bool(expected) and expected == {digest}, 'bounds': bounds,
                'bounds_method': profile.get('bounds_method'),
                'bounds_quality': quality,
                'symbols': profile['symbols']}

    def trained_market(self, market):
        if not self.feature_space.get('verified'):
            return []
        symbols = set(self.feature_space.get('symbols', []))
        return [row for row in market.get('all', []) if row.get('symbol') in symbols]

    def symbol_untrained(self, symbol):
        """Whether the model was ever trained on this symbol at all.

        A separate question from "are these features unusual", and it has a different
        answer. The gate below refuses a symbol whose features fall outside the training
        range, which is right -- but a symbol that never appeared in training is not
        drifting, it is unknown, and no amount of in-range features would make its
        prediction meaningful. Reporting both under one code left the operator reading
        "out_of_distribution" with a list of ordinary-looking features and no way to see
        that the session had selected markets the model had never seen.

        Observed live: the model trained on 12 majors, the session ran source='gainers',
        and all five selected micro-caps were blocked. ATR of 0.027-0.034 against a
        training ceiling of 0.0148 -- the gate was correct, and its message was not.
        """
        known = set((self.feature_space or {}).get('symbols') or [])
        if not known or not (self.feature_space or {}).get('verified'):
            return False
        return bool(symbol) and symbol not in known

    def out_of_range(self, features):
        """Features outside the verified training range; empty when unverifiable.

        Features whose stored bound is no wider than the clip range are skipped rather than
        tested: they cannot fire, and counting them as passing would make the gate look
        stronger than it is. The skipped set is exposed as bounds_quality so the
        distinction is visible.
        """
        bounds = (self.feature_space or {}).get('bounds')
        if not bounds:
            return []
        quality = (self.feature_space or {}).get('bounds_quality') or {}
        vacuous = set(quality.get('degenerate') or ())
        outside = []
        for name, limits in bounds.items():
            if name in vacuous:
                continue
            value = finite(features.get(name))
            # A feature present in training and absent at serving is itself out of
            # distribution, so it is reported here as well as by the degeneracy gate.
            if value is None or value < limits[0] or value > limits[1]:
                outside.append(name)
        return sorted(outside)

    # ------------------------------------------------------------ inference
    def predict_tabular(self, features):
        """Run every loaded non-Chronos member; returns (members, errors)."""
        members, errors = {}, {}
        outside = self.out_of_range(features)
        for name, member in self.models.items():
            try:
                result = member.predict(features)
                value = finite(result.get('expected_return'))
                if value is None:
                    raise ValueError('non_finite_prediction')
                members[name] = {'source': result.get('source', name), 'kind': member.kind,
                                 'expected_return': value, 'side': result.get('side'),
                                 'expected_net_return': finite(result.get('expected_net_return')),
                                 'probability_up': finite(result.get('probability_up')),
                                 'version': member.version, 'cost_bps': member.cost_bps,
                                 'out_of_range': list(outside)}
            except Exception as exc:
                errors[name] = repr(exc)
        return members, errors

    def _research_features(self, symbol, rows):
        """One feature row for the research surface.

        The same rule the decision path uses: ask the source only for the columns the loaded
        models declare, so a v3 artifact is not refused for the twenty columns it never
        named. Falling back to the raw snapshot keeps a feature failure from taking the
        shadow worker down."""
        closed = [row for row in rows if row.get('is_closed')] or list(rows)
        if not closed:
            return {}, ()
        bar_time = bar_id(closed[-1])
        if self._required is None:
            declared = set()
            for member in self.models.values():
                declared.update(getattr(member, 'features', ()) or ())
            self._required = (tuple(name for name in FEATURES if name in declared)
                              if declared else tuple(FEATURES))
        self.feature_source.feature_names = self._required
        try:
            return self.feature_source.snapshot(closed, symbol, bar_time)
        except Exception:
            return snapshot(closed), tuple(FUNDING_FEATURES)

    def research_predict(self, symbol, rows):
        """Full per-model prediction used by the research (shadow) surface."""
        features, _degraded = self._research_features(symbol, rows)
        members, errors = self.predict_tabular(features)
        bar_time = int(rows[-1].get('close_time') or rows[-1].get('open_time') or 0)
        predictions = dict(members)
        if self.chronos is not None:
            try:
                record = self.chronos_predict(symbol, [row['close'] for row in rows], bar_time)
                median = finite(record.get('median_return'))
                predictions['chronos-2'] = {'source': 'chronos-2', 'kind': 'chronos', 'mode': record.get('mode', 'paper_decision'),
                                            'horizon': record.get('horizon'), 'median_return': median,
                                            'quantiles': record.get('quantiles'), 'expected_return': median,
                                            'side': 'LONG' if (median or 0) > 0 else 'SHORT' if (median or 0) < 0 else 'FLAT',
                                            'latency_ms': record.get('latency_ms'), 'version': 'chronos-2'}
            except Exception as exc:
                errors['chronos-2'] = repr(exc)
        return {'predictions': predictions, 'load_errors': {**self.load_errors, **errors}}

    # -------------------------------------------------------------- chronos
    def chronos_predict(self, symbol, closes, bar_time=None):
        """Run or reuse one Chronos forecast for this symbol/bar in the caller thread."""
        cached = self.chronos_results.get(symbol)
        if cached and cached.get('status') == 'ok' and bar_time is not None and int(cached.get('bar_time') or -1) == int(bar_time):
            return cached
        if self.chronos is None:
            raise ValueError('chronos_disabled')
        started = time.monotonic()
        forecast = self.chronos.predict(closes=closes)
        record = {'symbol': symbol, 'bar_time': bar_time, 'status': 'ok',
                  'completed_at': int(time.time() * 1000),
                  'latency_ms': round((time.monotonic() - started) * 1000, 3), **forecast}
        self.chronos_results[symbol] = record
        if self.on_chronos_result:
            self.on_chronos_result(symbol, bar_time)
        return record

    def submit_chronos(self, symbol, closes, bar_time):
        if self.chronos is None or self.stopping or len(closes) < 32:
            return False
        if int(self.chronos_seen.get(symbol, -1)) == int(bar_time):
            return False
        if symbol not in self.chronos_seen and len(self.chronos_seen) >= self.max_chronos_symbols:
            return False
        self.chronos_seen[symbol] = int(bar_time)
        self.chronos_pending[symbol] = (list(closes[-512:]), int(bar_time))
        try:
            self.chronos_event.set()
        except RuntimeError:
            return False
        return True

    async def chronos_forecast(self, symbol, closes, bar_time, timeout_ms=3000):
        """Submit and briefly wait for this bar's Chronos vote.

        The wait keeps the forecast an actual ensemble member instead of a value that
        always lands after the decision for the bar has already been cached. Missing
        the deadline is harmless: the forecast still lands, invalidates the cached
        signal, and joins the next decision.
        """
        vote = self.chronos_vote(symbol, bar_time)
        if vote is not None:
            return vote
        if self.chronos is None or len(closes) < 32 or timeout_ms <= 0:
            return None
        self.submit_chronos(symbol, closes, bar_time)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_ms / 1000
        while loop.time() < deadline:
            vote = self.chronos_vote(symbol, bar_time)
            if vote is not None:
                return vote
            record = self.chronos_results.get(symbol) or {}
            if int(record.get('bar_time') or -1) == int(bar_time) and record.get('status') == 'error':
                return None
            await asyncio.sleep(.05)
        return None

    def chronos_vote(self, symbol, bar_time):
        record = self.chronos_results.get(symbol)
        if not record or record.get('status') != 'ok' or int(record.get('bar_time') or -1) != int(bar_time):
            return None
        value = finite(record.get('median_return'))
        if value is None:
            return None
        return {'source': 'chronos-2', 'kind': 'chronos', 'expected_return': value,
                'expected_net_return': None, 'probability_up': None,
                'side': 'LONG' if value > 0 else 'SHORT' if value < 0 else 'FLAT',
                'version': 'chronos-2', 'cost_bps': 0.0,
                'latency_ms': record.get('latency_ms'), 'horizon': record.get('horizon')}

    async def _chronos_loop(self):
        while not self.stopping:
            await self.chronos_event.wait()
            while self.chronos_pending and not self.stopping:
                symbol = next(iter(self.chronos_pending))
                closes, bar_time = self.chronos_pending.pop(symbol)
                try:
                    await asyncio.get_running_loop().run_in_executor(self.slow, self.chronos_predict, symbol, closes, bar_time)
                except Exception as exc:
                    self.chronos_results[symbol] = {'symbol': symbol, 'bar_time': bar_time, 'status': 'error',
                                                    'error': repr(exc), 'completed_at': int(time.time() * 1000)}
            self.chronos_event.clear()

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self.chronos is None or self.chronos_task is not None:
            return self.chronos_task
        self.chronos_task = asyncio.create_task(self._chronos_loop())
        return self.chronos_task

    async def close(self):
        self.stopping = True
        self.chronos_event.set()
        if self.chronos_task is not None:
            await asyncio.gather(self.chronos_task, return_exceptions=True)
            self.chronos_task = None
        self.fast.shutdown(wait=True)
        self.slow.shutdown(wait=True)

    def status(self):
        chronos = {'enabled': self.chronos_enabled, 'device': self.device, 'horizon': self.chronos_horizon,
                   'loaded': bool(self.chronos and self.chronos.forecaster is not None),
                   'error': getattr(self.chronos, 'error', None),
                   'forecasts': len(self.chronos_results), 'queued': len(self.chronos_pending)}
        return {'mode': self.mode, 'label': MODE_LABELS.get(self.mode, self.mode), 'loaded': self.loaded,
                'loaded_at': self.loaded_at, 'load_seconds': round(self.load_seconds, 3),
                'promotion': dict(self.promotion), 'members': [member.describe() for member in self.models.values()],
                'feature_space': self.feature_space, 'chronos': chronos, 'load_errors': dict(self.load_errors)}


class ModelDecision:
    """Turns real-model predictions into the signal/plan shape the paper loop consumes."""

    def __init__(self, runtime, fee_rate=.0004, slippage_bps=2.0, min_edge_bps=0.5, min_agreement=0.6,
                 min_edge_multiple=1.0, rule_fallback=False, ood_policy='block', chronos_timeout_ms=3000,
                 maker_fee_rate=0.0002, entry_order_type='market',
                 cache_size=512, recent_size=12, feature_source=None, degraded_blocks=True,
                 require_promoted=False, max_model_age_ms=0):
        self.runtime = runtime
        # Declared feature names this code cannot produce; non-empty means unservable.
        self.unusable_features = []
        # Every feature row the service evaluates comes from here, and so does every row
        # the training job writes. When this was two code paths the funding family was
        # silently constant at serving time.
        self.feature_source = feature_source if feature_source is not None else FeatureSource()
        self._required = None
        self.degraded_blocks = bool(degraded_blocks)
        # Refuse to trade on weights that never passed review, and on weights that are old
        # enough to be describing a market that no longer exists. Falling back to a
        # candidate is the failure mode this guards: it produces signals indistinguishable
        # from a promoted model's, so the only place the difference can be acted on is here.
        self.require_promoted = bool(require_promoted)
        self.max_model_age_ms = int(max_model_age_ms or 0)
        self.ood_policy = str(ood_policy or 'warn')
        self.chronos_timeout_ms = int(chronos_timeout_ms)
        self.fee_rate = float(fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)
        self.slippage_bps = float(slippage_bps)
        # Whether entries rest on the book. It decides the round-trip cost the gate is
        # measured against, so it has to reach this object rather than only the broker.
        self.maker_entry = str(entry_order_type or 'market').strip().lower() == 'limit'
        self.min_edge_bps = float(min_edge_bps)
        self.min_agreement = float(min_agreement)
        self.min_edge_multiple = float(min_edge_multiple)
        self.rule_fallback = bool(rule_fallback)
        # Swappable at runtime by the exit-policy endpoint. Kept on the decisions object
        # because that is what builds the plan, so a change applies to the next signal
        # without rebuilding the model stack.
        self.exit_policy = None
        # Per-symbol meta-labeling gate; installed by main after seeding from history.
        self.symbol_edge = None
        self.cache_size = int(cache_size)
        self.counts = {'signals': 0, 'blocked': 0, 'rule_fallback': 0, 'votes_with_chronos': 0}
        self.recent = []
        self._recent_size = int(recent_size)
        self._cache = {}

    @property
    def round_trip_cost_pct(self):
        """What one round trip actually costs under the configured execution mode.

        A market entry crosses the spread and pays the taker rate on both legs:
        2 x taker + 2 x half-spread. A passive entry rests on the book, is charged the
        maker rate when it fills, and never pays the spread, so the same round trip is
        2 x maker. The gate, the labels and the training cost were all priced from one
        number that assumed the first mode, which is why a 12 bp bar sat above every
        prediction the model makes -- the median is about 1 bp and the 90th percentile
        about 4 bp. Pricing the mode that is actually configured is what makes the
        comparison meaningful; it does not change what the market does."""
        if self.maker_entry:
            return 2 * self.maker_fee_rate
        return 2 * self.fee_rate + 2 * self.slippage_bps / 10000

    # ------------------------------------------------------------- signals
    async def signal(self, symbol, rows, price=None):
        closed = [row for row in rows if row.get('is_closed')] or list(rows)
        if len(closed) < 50:
            return self._materialize(self._blocked(symbol, 'insufficient_history', len(closed)), price)
        await self.runtime.ensure_loaded()
        key = (symbol, bar_id(closed[-1]))
        signal = self._cache.get(key)
        if signal is None:
            features, degraded = self._features(symbol, closed, key[1])
            members, errors = await asyncio.get_running_loop().run_in_executor(self.runtime.fast, self.runtime.predict_tabular, features)
            if not members and not self.runtime.has_members:
                signal = self._fallback(symbol, closed, self._unavailable_reason(), errors)
            else:
                vote = await self.runtime.chronos_forecast(symbol, [row['close'] for row in closed], key[1], self.chronos_timeout_ms)
                if vote is not None:
                    members = {**members, 'chronos-2': vote}
                signal = self._assemble(symbol, features, members, errors, key[1], degraded)
            self._store(key, signal)
        return self._materialize(signal, price)

    def signal_sync(self, symbol, rows, price=None):
        """Blocking variant with identical math, used by offline backtests."""
        closed = [row for row in rows if row.get('is_closed')] or list(rows)
        if len(closed) < 50:
            return self._materialize(self._blocked(symbol, 'insufficient_history', len(closed)), price)
        self.runtime.load_if_needed()
        key = (symbol, bar_id(closed[-1]))
        signal = self._cache.get(key)
        if signal is None:
            features, degraded = self._features(symbol, closed, key[1])
            members, errors = self.runtime.predict_tabular(features)
            if not members and not self.runtime.has_members:
                signal = self._fallback(symbol, closed, self._unavailable_reason(), errors)
            else:
                vote = self.runtime.chronos_vote(symbol, key[1])
                if vote is not None:
                    members = {**members, 'chronos-2': vote}
                signal = self._assemble(symbol, features, members, errors, key[1], degraded)
            self._store(key, signal)
        return self._materialize(signal, price)

    def provenance_problem(self):
        """Why the loaded weights may not be traded, or None when they may.

        Two distinct failures. Unpromoted weights failed the promotion gate -- in this
        repository no candidate has ever passed it, so every historical number came from
        weights that failed review. Expired weights passed review at a time whose market no
        longer exists; the promotion gate has no opinion about that, and nothing else did
        either.
        """
        if not getattr(self.runtime, 'models', None):
            return None
        if self.require_promoted:
            promotion = self.runtime.promotion or {}
            if promotion.get('status') != 'active':
                reason = promotion.get('reason') or promotion.get('status') or 'unknown'
                return 'model_not_promoted:' + str(reason)
        if self.max_model_age_ms > 0:
            loaded_at = int(getattr(self.runtime, 'loaded_at', 0) or 0)
            if loaded_at and int(time.time() * 1000) - loaded_at > self.max_model_age_ms:
                return 'model_expired'
            manifest_created = (self.runtime.promotion or {}).get('created_at')
            if manifest_created:
                try:
                    age = int(time.time() * 1000) - int(manifest_created)
                except (TypeError, ValueError):
                    age = 0
                if age > self.max_model_age_ms:
                    return 'model_expired'
        return None

    def _unavailable_reason(self):
        """The most specific reason the loaded runtime can give for having no weights."""
        resolver = getattr(self.runtime, 'unavailable_reason', None)
        if callable(resolver):
            try:
                return resolver()
            except Exception:
                pass
        return 'model_unavailable'

    def required_features(self):
        """The features the loaded models actually declare, in contract order.

        Asking the source for the contract's whole list instead would mark every column a
        v3 artifact never named as a missing input, and the degeneracy gate would refuse
        every trade. That is the right answer for a model that needs a column nothing can
        fill and the wrong one for a model that never asked for it.
        """
        if self._required is not None:
            return self._required
        declared = set()
        for member in (getattr(self.runtime, 'models', None) or {}).values():
            declared.update(getattr(member, 'features', ()) or ())
        if not declared:
            # A member that declares nothing gets the whole contract: it is a model whose
            # input list is unknown, and asking for everything is the conservative read.
            self._required = tuple(FEATURES)
        else:
            self._required = tuple(name for name in FEATURES if name in declared)
            # A non-empty declaration that intersects the contract in nothing is not a
            # small model, it is an unservable one: the weights were fitted on columns this
            # code cannot name, so every input the model reads would be absent. It used to
            # return an empty tuple, and an empty required list switches OFF the degeneracy
            # check -- the gate whose whole job is to refuse exactly this -- while the
            # service reported a healthy feature layer. Reported here, and the deployment
            # is refused by the degraded-input gate rather than traded.
            if not self._required:
                self.unusable_features = sorted(declared)
                self.runtime.load_errors['features'] = (
                    'declared_features_unproducible:%s' % ','.join(self.unusable_features[:5]))
        return self._required

    def _features(self, symbol, closed, bar_time=None):
        """One feature row plus the modelled features that had to be filled from a default."""
        if bar_time is None:
            bar_time = bar_id(closed[-1]) if closed else None
        # Resolved here rather than in __init__: the runtime may load its weights after the
        # decision object is built.
        self.feature_source.feature_names = self.required_features()
        try:
            return self.feature_source.snapshot(closed, symbol, bar_time)
        except Exception:
            # A feature failure must not take the decision loop down with it; it is
            # reported through the same degraded channel as a missing feed.
            return snapshot(closed), tuple(FUNDING_FEATURES)

    def _assemble(self, symbol, features, members, errors, bar_time, degraded=()):
        values = {name: member['expected_return'] for name, member in members.items() if member.get('expected_return') is not None}
        if not values:
            return self._fallback(symbol, [{'open_time': bar_time, 'close': features.get('price', 0),
                                            'high': features.get('price', 0), 'low': features.get('price', 0)}],
                                  'no_model_prediction', errors)
        expected = sum(values.values()) / len(values)
        # The calibrated probability, carried to where the decision is made. It was
        # computed by ProductionMember, copied into this dict, and then read by nothing --
        # not the gate below, not the signal, not the API. A model that reports how
        # confident it is and a service that discards the answer is the same as not
        # calibrating at all, and the calibrator is only now producible at all.
        #
        # Reported, not gated. The live gate is sign agreement plus an edge floor; turning
        # a probability into a third gate would change which trades are taken, and the
        # threshold for that has to come from a measured curve rather than from here.
        calibrated = [member['probability_up'] for member in members.values()
                      if member.get('expected_return') is not None
                      and member.get('probability_up') is not None]
        probability_up = sum(calibrated) / len(calibrated) if calibrated else None
        direction = 1 if expected > 0 else -1 if expected < 0 else 0
        agreeing = sum(1 for value in values.values() if (value > 0) - (value < 0) == direction) if direction else 0
        agreement = agreeing / len(values)
        # The cost contract, applied identically on both sides of the system. Training
        # decides a row is actionable when the prediction clears the round trip in the
        # predicted DIRECTION (tabular_model.evaluate: side = pred > cost ? LONG :
        # pred < -cost ? SHORT : FLAT). Serving used to gate on abs(expected) against a
        # 0.5bp floor, which dropped both the sign and the cost: on the shipped artifact
        # that marked 99.94% of test rows active instead of 0.14%, at -13.95bp per trade.
        # One definition, so the reported edge and the traded edge cannot diverge again.
        round_trip_cost = self.round_trip_cost_pct
        cost_bps = round_trip_cost * 10000
        net_bps = abs(expected) * 10000 - cost_bps if direction else 0.0
        edge_bps = net_bps
        outside = sorted({name for member in members.values() for name in (member.get('out_of_range') or [])})
        side, codes = 'FLAT', ['real_model_ensemble']
        # Weights that never passed review, and weights nobody has refreshed. Both produce
        # signals indistinguishable from a healthy model's, so this is the only place the
        # difference can be acted on. Checked before the other gates because it is a
        # statement about the model rather than about this particular input.
        provenance = self.provenance_problem()
        if provenance:
            codes.append(provenance)
        # A modelled feature that is a placeholder is not an out-of-range value, so the OOD
        # gate cannot see it: zero sits comfortably inside the training bounds. It is a
        # different failure and it needs its own check, because the model is being asked to
        # extrapolate from a constant it was never fit on.
        # Every gate is evaluated, then the decision is taken. This was an if/elif chain,
        # which meant the first gate to fire hid every gate below it: with
        # `feature_degraded` first, the edge measurement never ran, so there was no
        # record of how many bars the edge floor would also have blocked -- the number
        # needed to tell a data problem apart from a model problem. Blocking is now the
        # union of the reasons rather than the first one encountered.
        blocked_by = []
        if not direction:
            codes.append('no_directional_edge')
            blocked_by.append('no_directional_edge')
        if agreement + 1e-9 < self.min_agreement:
            codes.append('agreement_below_floor')
            blocked_by.append('agreement_below_floor')
        if edge_bps < self.min_edge_bps:
            codes.append('edge_below_floor')
            blocked_by.append('edge_below_floor')
        if outside and self.ood_policy == 'block':
            # Two different facts, reported separately. A symbol the model was never
            # trained on is not drifting; it is unknown, and saying "out_of_distribution"
            # for it sent the operator looking at feature bounds when the real answer was
            # that the session had selected markets outside the training universe.
            untrained = getattr(self.runtime, 'symbol_untrained', None)
            if untrained is not None and untrained(symbol):
                codes.append('symbol_not_in_training')
                blocked_by.append('symbol_not_in_training')
            codes.append('out_of_distribution')
            blocked_by.append('out_of_distribution')
        if degraded and self.degraded_blocks:
            codes.append('feature_degraded')
            codes.extend('feature_degraded:' + name for name in degraded)
            blocked_by.append('feature_degraded')
        if not blocked_by:
            side = 'LONG' if direction > 0 else 'SHORT'
            codes.append('ensemble_agrees')
            if outside:
                codes.append('out_of_distribution_warning')
        # Meta-labeling. The primary model proposed 'side'; this layer decides whether that
        # proposal is worth acting on for this particular symbol. The observation is
        # recorded before the gate is consulted, so a symbol that is currently blocked
        # keeps accumulating the evidence needed to earn its way back.
        proposed = 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'FLAT'
        if self.symbol_edge is not None:
            self.symbol_edge.observe(symbol, proposed, bar_time, features.get('price'),
                                     expected)
            if side in ('LONG', 'SHORT'):
                allowed, edge_reason = self.symbol_edge.allows(symbol)
                codes.append(edge_reason)
                if not allowed:
                    side = 'FLAT'
        if 'chronos-2' in values:
            self.counts['votes_with_chronos'] += 1
        signal = {'symbol': symbol, 'side': side, 'confidence': round(agreement, 4), 'features': dict(features),
                  'degraded_features': list(degraded),
                  'feature_source': self.feature_source.health(),
                  'scores': {'long': sum(1 for value in values.values() if value > 0),
                             'short': sum(1 for value in values.values() if value < 0)},
                  'reason_codes': codes, 'source': 'real_models',
                  'expected_return': round(expected, 8), 'expected_net_return': round(abs(expected) - self.round_trip_cost_pct, 8),
                  'probability_up': None if probability_up is None else round(probability_up, 6),
                  'calibrated_members': len(calibrated),
                  'uncalibrated_members': len(values) - len(calibrated),
                  'agreement': round(agreement, 4), 'edge_bps': round(edge_bps, 4),
                  'votes': {name: round(value, 8) for name, value in values.items()},
                  'member_errors': {**self.runtime.load_errors, **errors},
                  'out_of_range': outside, 'ood_policy': self.ood_policy,
                  'model_mode': self.runtime.mode, 'model_version': self._version_label(),
                  'bar_time': int(bar_time), 'computed_at': int(time.time() * 1000),
                  # The direction the primary model proposed, before this gate acted on it.
                  # 'side' is what the gate decided, so it reads FLAT for every symbol that
                  # is currently blocked -- and a blocked symbol is exactly the one whose
                  # evidence matters. Keeping both means the audit trail can still show
                  # what was proposed, which is what the gate learns from.
                  'proposed_side': proposed}
        self.counts['signals'] += 1
        self._remember({'symbol': symbol, 'side': side, 'source': 'real_models', 'bar_time': int(bar_time),
                        'expected_return': signal['expected_return'], 'edge_bps': signal['edge_bps'],
                        'agreement': signal['agreement'], 'votes': signal['votes'],
                        'probability_up': signal['probability_up'],
                        'calibrated_members': signal['calibrated_members'],
                        'model_mode': self.runtime.mode, 'reason_codes': codes, 'out_of_range': outside})
        return signal

    def _fallback(self, symbol, closed, reason, errors=None):
        if not self.rule_fallback:
            self.counts['blocked'] += 1
            self._remember({'symbol': symbol, 'side': 'FLAT', 'source': 'model_blocked', 'reason': reason})
            blocked = self._blocked(symbol, reason, len(closed))
            blocked['member_errors'] = {**self.runtime.load_errors, **(errors or {})}
            blocked['bar_time'] = bar_id(closed[-1]) if closed else 0
            blocked['features'] = self._features(symbol, closed)[0] if len(closed) >= 50 else {}
            return blocked
        self.counts['rule_fallback'] += 1
        legacy = rule_predict(symbol, closed)
        signal = {**legacy, 'source': 'rule_fallback', 'fallback_reason': reason, 'model_mode': 'unavailable',
                  'expected_return': None, 'expected_net_return': None, 'agreement': None, 'edge_bps': None,
                  'votes': {}, 'member_errors': {**self.runtime.load_errors, **(errors or {})},
                  'model_version': None, 'bar_time': bar_id(closed[-1]),
                  'computed_at': int(time.time() * 1000)}
        self._remember({'symbol': symbol, 'side': signal['side'], 'source': 'rule_fallback', 'reason': reason})
        return signal

    def _blocked(self, symbol, reason, rows=0):
        return {'symbol': symbol, 'side': 'FLAT', 'confidence': 0.0, 'features': {}, 'scores': {'long': 0, 'short': 0},
                'reason_codes': ['model_blocked', reason], 'source': 'model_blocked', 'expected_return': None,
                'expected_net_return': None, 'agreement': None, 'edge_bps': None, 'votes': {},
                'member_errors': dict(self.runtime.load_errors), 'model_mode': self.runtime.mode,
                'model_version': None, 'rows': rows, 'bar_time': 0, 'computed_at': int(time.time() * 1000)}

    def _version_label(self):
        versions = sorted({member.version for member in self.runtime.models.values()})
        if self.runtime.chronos is not None and self.runtime.chronos.forecaster is not None:
            versions.append('chronos-2')
        return '+'.join(versions) if versions else None

    def _materialize(self, signal, price):
        out = dict(signal)
        features = dict(signal.get('features') or {})
        value = finite(price)
        if value is not None and value > 0:
            features['price'] = value
        out['features'] = features
        if features.get('price'):
            out['entry'] = float(features['price'])
        return out

    # ---------------------------------------------------------------- plan
    def plan(self, signal, entry=None, fee_rate=None, slippage_bps=None):
        """Risk plan for an approved direction; stop geometry matches the rule engine."""
        if signal.get('side') not in ('LONG', 'SHORT'):
            return None
        features = signal.get('features') or {}
        price = finite(entry if entry is not None else features.get('price'))
        if price is None or price <= 0:
            return None
        atr = finite(features.get('atr')) or 0.0
        # One definition of the round trip, not two. This recomputed it from the taker rate
        # and the spread even when the entry was configured to rest on the book, so the
        # gate here demanded a move that cleared 12 bp while the edge gate upstream had
        # already accepted the same signal at 4 bp. The stricter of two numbers for one
        # quantity decides what happens, so the stricter one silently won and every plan
        # was refused: measured on ETHUSDT, model_move 1.136 against a threshold of 3.029.
        # An explicit fee_rate/slippage_bps override still prices itself, because a caller
        # that names its own cost is describing a different fill than the configured one.
        if fee_rate is None and slippage_bps is None:
            round_trip = price * self.round_trip_cost_pct
        else:
            fee = float(self.fee_rate if fee_rate is None else fee_rate)
            slippage = float(self.slippage_bps if slippage_bps is None else slippage_bps)
            round_trip = price * (2 * fee + 2 * slippage / 10000)
        # Geometry comes from the active exit policy instead of a constant, so stop
        # width and reward:risk are adjustable without editing code.
        policy = self.exit_policy or exit_policy.get_policy()
        levels = exit_policy.plan_levels(policy, price, atr, signal['side'])
        if levels is None:
            return None
        distance = levels['stop_distance']
        # The model's own expected move, and the geometry's target, are two different
        # numbers and the gate has to be about the right one. This compared
        # `max(geometry_target, model_expected)` against the cost floor -- and because the
        # policy target is essentially always the larger of the two, the gate was measuring
        # the stop/target ratio, which is a constant, rather than the model's edge, which is
        # the only thing that varies. It passed unconditionally.
        #
        # Requiring the MODEL's expected move to clear min_edge_multiple round trips is the
        # test the parameter was named for, and it is what makes MIN_EDGE_MULTIPLE an
        # independent gate rather than a restatement of the exit policy.
        model_move = abs(finite(signal.get('expected_return')) or 0.0) * price
        if model_move < round_trip * self.min_edge_multiple:
            return None
        # The target still has to be at least what the model expects, or the plan would
        # exit before the move it was taken for. It is never allowed below the policy
        # ratio either, since that would quietly change what the exit policy means.
        expected_move = max(levels['target_distance'], model_move)
        side = signal['side']
        return {**signal, 'entry': price,
                'stop': levels['stop'],
                'take_profit': price + expected_move if side == 'LONG' else price - expected_move,
                'expected_cost': round_trip, 'expected_edge': expected_move - round_trip,
                'stop_distance': distance, 'target_distance': expected_move,
                'exit_policy': policy.name}

    # -------------------------------------------------------------- plumbing
    def invalidate(self, symbol, bar_time):
        self._cache.pop((symbol, int(bar_time or 0)), None)

    def _store(self, key, signal):
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = signal

    def _remember(self, entry):
        self.recent.append({**entry, 'at': int(time.time() * 1000)})
        if len(self.recent) > self._recent_size:
            del self.recent[:len(self.recent) - self._recent_size]

    def brief(self):
        """Compact state for the high-frequency websocket payload."""
        return {'mode': self.runtime.mode, 'label': MODE_LABELS.get(self.runtime.mode, self.runtime.mode),
                'model_version': self._version_label(), 'counts': dict(self.counts),
                'last_signal': self.recent[-1] if self.recent else None}

    def status(self):
        runtime = self.runtime.status()
        return {'mode': runtime['mode'], 'label': runtime['label'], 'production': self._production(runtime),
                'rule_fallback': self.rule_fallback, 'min_edge_bps': self.min_edge_bps, 'ood_policy': self.ood_policy,
                'min_agreement': self.min_agreement, 'round_trip_cost_bps': round(self.round_trip_cost_pct * 10000, 4),
                'chronos_timeout_ms': self.chronos_timeout_ms,
                'feature_space': runtime.get('feature_space') or {},
                'members': runtime['members'], 'promotion': runtime['promotion'], 'chronos': runtime['chronos'],
                'load_errors': runtime['load_errors'], 'loaded': runtime['loaded'], 'load_seconds': runtime['load_seconds'],
                'counts': dict(self.counts), 'recent': list(self.recent)}

    def _production(self, runtime):
        for member in runtime['members']:
            if member['kind'] == 'production':
                return {'version': member['version'], 'status': member['status'], 'path': member['path']}
        return None
