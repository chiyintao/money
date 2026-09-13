import json
from pathlib import Path
import numpy as np
from .calibration import build_calibration
from .fit_model import FEATURES, load
from .dataset_split import chronological_split
from ..features.feature_spec import FEATURE_VERSION

def _matrix(rows): return np.array([[float(row[k]) for k in FEATURES] for row in rows],dtype=float)
def _stump(x,y,weights):
    best=None
    base=np.average(y,weights=weights); base_loss=float(np.sum(weights*(y-base)**2))
    for feature in range(x.shape[1]):
        values=np.unique(x[:,feature])
        for threshold in values[::max(1,len(values)//32)]:
            left=x[:,feature]<=threshold; right=~left
            if not left.any() or not right.any(): continue
            lp=np.average(y[left],weights=weights[left]); rp=np.average(y[right],weights=weights[right]); loss=float(np.sum(weights[left]*(y[left]-lp)**2)+np.sum(weights[right]*(y[right]-rp)**2))
            if best is None or loss<best[0]: best=(loss,feature,float(threshold),float(lp),float(rp))
    return best or (base_loss,0,float(np.median(x[:,0])),base,base)

def _predict_stump(x, stump):
    _,feature,threshold,left,right=stump; return np.where(x[:,feature]<=threshold,left,right)

def fit_advanced(rows,train_ratio=.7,valid_ratio=.15,rounds=32,learning_rate=.05,max_depth=1,cost_bps=4,output='data/model_advanced.json',store=None,interval='5m'):
    """Fit the stumps model, and the evidence the promotion gate will ask it for.

    `store` is optional but its absence decides whether a promotion is possible at all.
    The registry refuses any model without an out-of-sample portfolio result, and this
    function cannot compute one from the rows it is given: the training rows carry the ten
    features, a label and a timestamp, and a portfolio replay needs the candles those
    timestamps belong to. Passing the store the dataset was built from closes that gap. It
    stays optional so the fit itself remains a pure function of its rows.
    """
    if max_depth != 1 or rounds < 1 or not 0 < learning_rate <= 1 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('invalid_stump_parameters')
    train, valid, test, split = chronological_split(rows, train_ratio, valid_ratio)
    if not split['ready']:
        return {'status': 'insufficient_data', 'rows': len(rows), 'split': split}
    usable = train + valid + test
    x = _matrix(usable)
    y = np.array([float(r['future_return']) for r in usable])
    cut, valid_cut = len(train), len(train) + len(valid)
    mean, scale = x[:cut].mean(0), x[:cut].std(0)
    scale[scale < 1e-12] = 1
    z = (x-mean)/scale
    base_score = float(y[:cut].mean())
    pred = np.full(cut, base_score)
    stumps = []
    for _ in range(rounds):
        stump = _stump(z[:cut], y[:cut]-pred, np.ones(cut))
        pred += learning_rate*_predict_stump(z[:cut], stump)
        stumps.append(stump)
    all_pred = np.full(len(y), base_score)
    for stump in stumps:
        all_pred += learning_rate*_predict_stump(z, stump)
    def report(start, end):
        actual, forecast = y[start:end], all_pred[start:end]
        cost = cost_bps/10000
        position = np.where(forecast > cost, 1, np.where(forecast < -cost, -1, 0))
        net = position*actual - abs(position)*cost
        return {'rows': len(actual), 'mae': float(np.mean(abs(actual-forecast))),
                'directional_accuracy_pct': float(np.mean(np.sign(actual)==np.sign(forecast))*100),
                'mean_net_return': float(net.mean()), 'sum_sample_net_return': float(net.sum()),
                'evaluation': 'independent_overlapping_samples_not_portfolio', 'cost_bps': cost_bps}
    # The calibrator the decision layer applies to `0.5 + expected_return * 10`. Fit here,
    # on the test slice, which is the only part of the data this model has not seen. Until
    # this existed nothing produced one: the artifact is absent from every registry entry,
    # the promotion gate refuses on the missing flag, and the runtime loader refuses on the
    # missing file. Fit on the same scale the serving path uses, or it returns numbers that
    # look calibrated and are not.
    calibrator, calibration = build_calibration(all_pred[valid_cut:len(y)], y[valid_cut:len(y)],
                                                bins=10)
    result = {'status': 'ok', 'model': 'gradient_boosted_stumps', 'model_version': 'gbst-v3',
              # Declared from the constant the registry compares against. The literal
              # 'features-v1' here outlived the feature set it named, so every artifact this
              # produced was refused for an incompatible feature version it did not have.
              'feature_version': FEATURE_VERSION,
              'rows': len(y), 'features': FEATURES, 'split': split,
              'calibration': calibration, 'calibrator': calibrator,
              'base_score': base_score, 'target': 'gross_return',
              'train': report(0,cut), 'validation': report(cut,valid_cut), 'test': report(valid_cut,len(y)),
              'normalization': {'mean': mean.tolist(), 'scale': scale.tolist()},
              'parameters': {'rounds': rounds, 'learning_rate': learning_rate, 'max_depth': 1, 'cost_bps': cost_bps},
              'stumps': [list(s) for s in stumps]}
    # The test slice's predictions, scored through one shared account. `test` and the tail
    # of `all_pred` are the same rows in the same order, so the pairing is positional
    # rather than a lookup that could silently misalign them.
    if store is not None:
        from .portfolio_oos import funding_for, funding_windows, portfolio_evidence

        oos = [{'timestamp': int(row['timestamp']), 'symbol': row.get('symbol'),
                'predicted_return': float(all_pred[valid_cut + index])}
               for index, row in enumerate(test)]
        symbols = sorted({str(row.get('symbol')) for row in test if row.get('symbol')})
        start = min(int(row['timestamp']) for row in test) if test else 0
        end = max(int(row['timestamp']) for row in test) if test else 0
        bars = store.candles_range(symbols, start, end, interval) if symbols else []
        # Funding, because the live loop charges it and the replay used to not. Without the
        # rates the block comes back with costs_included False, which is the gate refusing
        # the evidence rather than the evidence being wrong.
        result['portfolio_oos'] = portfolio_evidence(
            bars, oos, cost_bps=cost_bps, interval=interval,
            funding=funding_for(store, symbols), windows=funding_windows(store, symbols))
    else:
        # Stated rather than omitted: an absent block and a failing one are different
        # facts, and the gate reports "insufficient_portfolio_evidence" for both.
        result['portfolio_oos'] = {
            'costs_included': True, 'trades': 0, 'net_return': 0.0, 'max_drawdown': 0.0,
            'status': 'unavailable', 'reason': 'no_store_passed_for_portfolio_replay'}
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def predict_advanced(model, features):
    if model.get('target') != 'gross_return' or 'base_score' not in model:
        raise ValueError('legacy_model_requires_retraining')
    x = np.array([[float(features[k]) for k in FEATURES]])
    if not np.isfinite(x).all():
        raise ValueError('non_finite_features')
    mean = np.array(model['normalization']['mean'])
    scale = np.array(model['normalization']['scale'])
    z = (x-mean)/np.where(scale < 1e-12, 1, scale)
    value = float(model['base_score'])
    for stump in model['stumps']:
        value += model['parameters']['learning_rate']*float(_predict_stump(z, stump)[0])
    cost = model['parameters']['cost_bps']/10000
    side = 'LONG' if value > cost else 'SHORT' if value < -cost else 'FLAT'
    return {'expected_return': value, 'expected_net_return': abs(value)-cost if side != 'FLAT' else 0.0,
            'side': side, 'model': model['model_version']}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser()
 p.add_argument('dataset',nargs='?',default='data/training_dataset.jsonl')
 p.add_argument('--cost-bps',type=float,default=4)
 p.add_argument('--output',default='data/model_advanced.json')
 p.add_argument('--register',action='store_true')
 # The round count was fixed at fit_advanced's default of 32, and 32 depth-1 stumps at a 0.05
 # learning rate cannot move the prediction off the base score by anything close to the
 # round-trip cost. Measured on the v4 dataset: base_score -0.90 bp against a 12 bp entry
 # threshold, so no bar ever produced a decision, the replay saw zero trades, and the registry
 # refused the candidate for missing portfolio evidence. How much capacity a model has decides
 # whether it can act at all, so it should not be a constant no caller can reach.
 p.add_argument('--rounds',type=int,default=None,
                help='boosting rounds; the fit default is used when omitted')
 p.add_argument('--interval',default='5m')
 p.add_argument('--no-store',action='store_true',
                help='skip the portfolio replay; the promotion gate will refuse the result')
 a=p.parse_args()
 # One fit. Registering used to refit the model from scratch inside the register branch, so
 # the artifact that reached the registry was a second, independently seeded fit rather than
 # the one whose metrics had just been printed.
 store=None
 if a.register and not a.no_store:
  from ..core.config import Settings
  from ..storage.storage import Store
  # The same store the dataset was built from. Without it there are no candles to replay
  # the out-of-sample predictions through, the portfolio block comes back unavailable, and
  # the promotion is refused -- which is what happened silently for every run before this:
  # the gate was never reachable from this producer.
  store=Store(Settings().data_dir)
 try:
  fit_kwargs={'cost_bps':a.cost_bps,'output':a.output,'store':store,'interval':a.interval}
  if a.rounds is not None:
   fit_kwargs['rounds']=a.rounds
  result=fit_advanced(load(a.dataset),**fit_kwargs)
 finally:
  if store is not None:
   store.close()
 if a.register and result.get('status')=='ok':
  from .model_registry import ModelRegistry
  # The calibrator travels with the model. Registering without it would write a manifest
  # whose calibration flag describes a file that was never created.
  try:
   result['registry']=ModelRegistry().register(result,{'test':result.get('test',{}),
       'calibration':result.get('calibration') or {},
       'portfolio_oos':result.get('portfolio_oos') or {}},
       {'path':a.dataset,'rows':result['rows']},version=result.get('model_version','gbst-v3'),
       calibrator=result.get('calibrator'))
  except ValueError as exc:
   # A refusal states its reason instead of ending in a traceback, because "the gate would
   # not have this" is the normal outcome of a run whose evidence is thin.
   result['registry']={'status':'refused','reason':str(exc)}
 print(json.dumps(result,ensure_ascii=False))
