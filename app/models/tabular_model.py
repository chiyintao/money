"""Native LightGBM/CatBoost artifacts with purged evaluation and no auto-promotion."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import os
import numpy as np
from .calibration import build_calibration
from .dataset_io import rows_digest
from .dataset_split import chronological_split
from ..features.feature_spec import FEATURE_VERSION, unproducible
from .fit_model import FEATURES, load
from .labels import sample_weights


def label_weights(rows):
    """Per-row fit weights from label overlap, or None when the rows carry no intervals.

    Returning None rather than a list of ones matters for LightGBM: an explicit uniform
    weight vector is not the same code path as no weights, and it made a dataset built
    before labels recorded their spans behave differently from one built after. Passing
    no weights leaves that case bit-identical to the previous behaviour.
    """
    if not rows:
        return None
    if not any('label_start' in row or 'bars_held' in row for row in rows):
        return None
    weights = np.asarray(sample_weights(rows), dtype=float)
    if weights.size != len(rows) or not np.isfinite(weights).all():
        return None
    return weights


def matrix(rows, features=FEATURES):
    """Design matrix for whichever feature list the caller is using.

    The list is an argument because a model is served on the features it was fit on, not on
    whatever the code's current constant happens to be. Selecting by a global constant made
    every feature addition an outage: the pipeline could compute the ten names a stored
    artifact needed, and refused to, because the module had since learned twenty more.
    """
    values = np.array([[float(row[key]) for key in features] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('non_finite_features')
    return values


def evaluate(actual, prediction, cost_bps):
    """Cost-aware metrics.

    Directional accuracy alone says nothing about profitability: a model can be right
    more often than not and still lose once the round-trip cost is paid. The active_*
    fields describe only the samples the model would actually have traded, which is the
    number that decides whether an edge survives costs.
    """
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    cost = cost_bps/10000
    side = np.where(prediction > cost, 1, np.where(prediction < -cost, -1, 0))
    net = side*actual - abs(side)*cost
    active = side != 0
    gross_active = (side*actual)[active]
    net_active = net[active]
    return {'rows': len(actual), 'mae': float(np.mean(abs(actual-prediction))),
            'directional_accuracy_pct': float(np.mean(np.sign(actual)==np.sign(prediction))*100),
            'mean_net_return': float(net.mean()), 'cost_bps': cost_bps,
            'active_samples': int(np.count_nonzero(side)),
            'active_share_pct': float(active.mean()*100),
            'active_hit_rate_pct': float(np.mean(gross_active > 0)*100) if net_active.size else 0.0,
            'active_net_edge_bps': float(net_active.mean()*10000) if net_active.size else 0.0,
            'evaluation': 'independent_overlapping_samples_not_portfolio'}


def per_symbol(rows, predictions, cost_bps):
    """Cost-aware metrics grouped by symbol.

    A pooled score can hide one symbol carrying the model while another loses on every
    trade; pooling without breaking it down makes that invisible.
    """
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.get('symbol', '?'), []).append(index)
    predictions = np.asarray(predictions)
    report = {}
    for symbol in sorted(groups):
        index = np.asarray(groups[symbol])
        actual = [rows[i]['future_return'] for i in groups[symbol]]
        report[symbol] = evaluate(actual, predictions[index], cost_bps)
    return report


def _sharpe_validation(rows, predictions, cost_bps, trials):
    """Deflated Sharpe of the traded net returns, and the trial count behind it.

    The Sharpe is computed on what the strategy would actually have earned: the signed
    return of each position the model would have taken, minus the round-trip cost. A
    Sharpe of the raw prediction error would measure a different thing entirely.
    """
    from .validation import validate as validate_sharpe

    actual = np.asarray([row['future_return'] for row in rows], dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    cost = float(cost_bps) / 10000.0
    side = np.where(prediction > cost, 1.0, np.where(prediction < -cost, -1.0, 0.0))
    net = side * actual - np.abs(side) * cost
    traded = net[side != 0.0]
    observations = int(traded.size)
    if observations < 2:
        return {"deflated_sharpe": None, "trials": int(trials or 0),
                "observations": observations, "survives": None,
                "reason": "no_traded_observations"}
    mean = float(traded.mean())
    deviation = float(traded.std(ddof=1))
    if deviation <= 0:
        return {"deflated_sharpe": None, "trials": int(trials or 0),
                "observations": observations, "survives": None,
                "reason": "zero_dispersion"}
    sharpe = mean / deviation
    centred = (traded - mean) / deviation
    skew = float((centred ** 3).mean())
    kurtosis = float((centred ** 4).mean())
    result = validate_sharpe(sharpe, trials, observations, skew=skew, kurtosis=kurtosis)
    result["per_observation_sharpe"] = round(sharpe, 8)
    result["traded_observations"] = observations
    result["mean_net_return"] = round(mean, 8)
    return result


def _threads():
    """How many threads a booster may use.

    Both backends were pinned at two threads, on a host with twenty-four. A fit that took
    minutes was using an eighth of the machine, and the thread count was a constant in the
    parameter dict rather than anything a caller could reach. MODEL_THREADS overrides it;
    the default is most of the machine, leaving a core or two for the service that is
    streaming quotes while the fit runs.
    """
    raw = os.getenv("MODEL_THREADS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return max(1, min(16, (os.cpu_count() or 2) - 2))


def _gpu_requested():
    """Whether the boosters were asked to use the GPU.

    LightGBM's pip wheel is built without the GPU tree learner, so asking it for one raises
    rather than falling back -- the device is only passed to a backend that can honour it,
    and CatBoost reports the failure itself if the driver is missing.
    """
    return os.getenv("MODEL_GBDT_DEVICE", "cpu").strip().lower() in ("gpu", "cuda")


def train_tabular(rows, backend='lightgbm', output_root='data/candidates', rounds=200,
                  cost_bps=12, trials=1):
    if backend not in ('lightgbm', 'catboost') or rounds < 1 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('invalid_model_parameters')
    train, valid, test, split = chronological_split(rows)
    if not split['ready'] or not split.get('label_intervals_verified'):
        raise ValueError({'reason': 'unverified_training_dataset', 'split': split})
    x, y = matrix(train), np.array([r['future_return'] for r in train])
    vx, vy = matrix(valid), np.array([r['future_return'] for r in valid])
    # Overlapping labels are nearly the same observation counted several times, so each row
    # is weighted by how much of its outcome window is its own. Without this a twelve-bar
    # horizon makes every row share eleven twelfths of its window with its neighbour, and
    # the fit treats one event as a dozen independent confirmations of itself.
    weights = label_weights(train)
    valid_weights = label_weights(valid)
    if backend == 'lightgbm':
        import lightgbm as lib
        parameters = {'objective': 'regression', 'learning_rate': .03, 'num_leaves': 15,
                      'min_data_in_leaf': 10, 'seed': 42, 'num_threads': _threads(),
                      'deterministic': True, 'force_col_wise': True, 'verbosity': -1}
        # The GPU tree learner is a compile-time option and the published wheel does not
        # include it, so this is only sent when asked for; LightGBM then says so itself.
        if _gpu_requested():
            parameters['device_type'] = 'gpu'
        model = lib.train(parameters,
                          lib.Dataset(x, label=y, weight=weights,
                                      feature_name=list(FEATURES)),
                          num_boost_round=rounds,
                          valid_sets=[lib.Dataset(vx, label=vy, weight=valid_weights)],
                          callbacks=[lib.early_stopping(20, verbose=False)])
        filename = 'model.txt'
    else:
        import catboost as lib
        parameters = {'iterations': rounds, 'depth': 5, 'learning_rate': .03,
                      'loss_function': 'RMSE', 'random_seed': 42, 'thread_count': _threads(),
                      'verbose': False, 'allow_writing_files': False}
        if _gpu_requested():
            parameters['task_type'] = 'GPU'
        model = lib.CatBoostRegressor(**parameters)
        model.fit(x, y, sample_weight=weights, eval_set=(vx, vy),
                  early_stopping_rounds=20)
        filename = 'model.cbm'
    parts = [('train', train), ('validation', valid), ('test', test)]
    predictions = {name: model.predict(matrix(part)) for name, part in parts}
    reports = {name: evaluate([r['future_return'] for r in part], predictions[name], cost_bps)
               for name, part in parts}
    by_symbol = per_symbol(test, predictions['test'], cost_bps)
    # The calibrator. The registry gate requires `calibration.fitted` and a calibrator
    # artifact, and this function never produced either -- so every tabular candidate was
    # refused for a missing calibrator regardless of how much edge it had, and the gate that
    # was supposed to be about evidence was really about a step nobody had implemented.
    # Fitted on the validation split and scored on test, so the reported improvement is
    # measured on rows the fit never saw.
    calibrator, calibration = build_calibration(predictions['validation'],
                                                [r['future_return'] for r in valid])
    # How much of the reported edge is selection rather than signal. `trials` is how many
    # configurations were compared before this one was kept -- the number the audit found
    # nobody was recording, which makes a Sharpe ratio uninterpretable. The deflation is
    # computed on the TEST split so it is not itself an in-sample figure, and it is
    # reported as advisory: a model can be worth shipping on grounds this does not capture.
    validation = _sharpe_validation(test, predictions['test'], cost_bps, trials)
    root = Path(output_root) / (backend + '-' + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    model.save_model(str(root/filename))
    # The calibrator travels with the candidate, in the file the loader looks for. A metrics
    # block that claims a fitted calibrator while no artifact exists is the state the gate
    # exists to catch, so the two are written together or not at all.
    if calibrator is not None:
        (root/'calibration.json').write_text(json.dumps(calibrator, allow_nan=False),
                                             encoding='utf-8')
        reports = {**reports, 'calibration': calibration}
    manifest = {'status': 'candidate', 'backend': backend, 'library_version': lib.__version__,
                'created_at': int(time.time()*1000), 'features': list(FEATURES),
                'feature_version': FEATURE_VERSION, 'target': 'gross_return', 'split': split,
                'parameters': parameters, 'cost_bps': cost_bps, 'metrics': reports,
                'metrics_by_symbol': by_symbol,
                'validation': validation,
                'dataset_rows': len(rows),
                'dataset_symbols': sorted({r['symbol'] for r in rows}),
                'model_file': filename, 'sha256': hashlib.sha256((root/filename).read_bytes()).hexdigest(),
                'dataset_sha256': rows_digest(sorted(rows, key=lambda r: (r['timestamp'], r['symbol'])))}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return {'path': str(root), **manifest}


class TabularPredictor:
    def __init__(self, folder, mode='shadow'):
        root = Path(folder)
        self.mode = str(mode or 'shadow')
        self.manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        meta = self.manifest
        declared = list(meta.get('features') or ())
        if not declared or meta.get('target') != 'gross_return':
            raise ValueError('incompatible_model_schema')
        # The artifact's own list is the contract. It must be a list this code can produce
        # in full: a name outside the known universe would have to be filled from nowhere,
        # and silently substituting a different column set is how a model trained on one
        # thing gets served another.
        unknown = unproducible(declared)
        if unknown:
            raise ValueError('unproducible_features:%s' % ','.join(unknown))
        if len(set(declared)) != len(declared):
            raise ValueError('duplicate_features')
        self.features = declared
        filename = meta['model_file']
        if Path(filename).name != filename:
            raise ValueError('invalid_model_path')
        path = root/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('model_checksum_mismatch')
        if meta['backend'] == 'lightgbm':
            import lightgbm
            self.model = lightgbm.Booster(model_file=str(path))
        elif meta['backend'] == 'catboost':
            import catboost
            self.model = catboost.CatBoostRegressor()
            self.model.load_model(str(path))
        else:
            raise ValueError('unsupported_backend')

    def predict(self, features):
        # Exactly the declared columns, in the declared order. The row handed in carries
        # every feature the code can build; the model wants the subset it was fit on.
        value = float(self.model.predict(matrix([features], self.features))[0])
        if not np.isfinite(value):
            raise ValueError('non_finite_prediction')
        cost = self.manifest['cost_bps']/10000
        side = 'LONG' if value > cost else 'SHORT' if value < -cost else 'FLAT'
        return {'source': self.manifest['backend'], 'expected_return': value,
                'expected_net_return': abs(value)-cost if side != 'FLAT' else 0.0,
                'side': side, 'mode': self.mode, 'cost_bps': self.manifest['cost_bps']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--backend', choices=('lightgbm', 'catboost'), default='lightgbm')
    parser.add_argument('--output-root', default='data/candidates')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--cost-bps', type=float, default=12)
    args = parser.parse_args()
    print(json.dumps(train_tabular(load(args.dataset), args.backend, args.output_root, args.rounds, args.cost_bps), indent=2))
