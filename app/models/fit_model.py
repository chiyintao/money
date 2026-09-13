import argparse
import json
from pathlib import Path
import numpy as np
from .dataset_split import chronological_split

from ..features.feature_spec import FEATURES, FEATURE_VERSION  # noqa: F401  (re-exported contract)


def load(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit(rows, train_ratio=.7, valid_ratio=.15, output='data/model_baseline.json'):
    train, valid, test, split = chronological_split(rows, train_ratio, valid_ratio)
    if not split['ready']:
        return {'status': 'insufficient_data', 'rows': len(rows), 'split': split}
    def matrix(part):
        return np.array([[float(r[k]) for k in FEATURES] for r in part])
    x = matrix(train)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    def design(part):
        z = (matrix(part) - mean) / scale
        return np.c_[np.ones(len(z)), z]
    coef = np.linalg.lstsq(design(train), [r['future_return'] for r in train], rcond=None)[0]
    def report(part):
        actual = np.array([r['future_return'] for r in part])
        forecast = design(part) @ coef
        return {'rows': len(part), 'mae': float(np.mean(abs(actual-forecast))),
                'directional_accuracy_pct': float(np.mean(np.sign(actual)==np.sign(forecast))*100),
                'mean_actual': float(actual.mean()), 'mean_predicted': float(forecast.mean())}
    result = {'status': 'ok', 'model': 'linear_baseline', 'model_version': 'linear-v3',
              # Declared from the constant. The literal 'features-v1' here was the second
              # of the two places that outlived the set it named, so the linear baseline --
              # the model the registry is supposed to fall back to -- also declared a
              # version nothing produced.
              'feature_version': FEATURE_VERSION, 'rows': len(rows), 'features': FEATURES,
              'split': split, 'train': report(train), 'validation': report(valid), 'test': report(test),
              'normalization': {'mean': mean.tolist(), 'scale': scale.tolist()}, 'coefficients': coef.tolist()}
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='data/training_dataset.jsonl')
    parser.add_argument('--output', default='data/model_baseline.json')
    args = parser.parse_args()
    print(json.dumps(fit(load(args.dataset), output=args.output)))
