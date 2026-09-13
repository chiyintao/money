import hashlib, json, time, math
from pathlib import Path

from ..features.feature_spec import features_for, unproducible

class ModelRegistry:
    def __init__(self, root='data/models'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def register(self, model, metrics, dataset_manifest, version=None, calibrator=None):
        # The calibrator is written here, into the version folder, because that is where the
        # runtime loader reads it from. Registering a metrics block that claims a calibrator
        # while writing no artifact let the promotion gate pass on a flag describing a file
        # that did not exist; the model then fell back at load time with nothing reporting
        # which of the two was lying.
        from .calibration import valid_calibrator

        metrics = dict(metrics or {})
        claimed = bool((metrics.get('calibration') or {}).get('fitted'))
        if claimed and not valid_calibrator(calibrator):
            raise ValueError('calibration_artifact_missing')
        if calibrator is not None and not valid_calibrator(calibrator):
            raise ValueError('calibration_artifact_invalid')
        if not claimed and valid_calibrator(calibrator):
            metrics['calibration'] = {'fitted': True, 'method': calibrator.get('method'),
                                      'rows': calibrator.get('rows')}
        version=version or f'model-{int(time.time())}'; folder=self.root/version; folder.mkdir(exist_ok=False)
        payload=json.dumps(model,sort_keys=True).encode(); (folder/'model.json').write_bytes(payload)
        if valid_calibrator(calibrator):
            (folder/'calibration.json').write_text(
                json.dumps(calibrator, sort_keys=True, allow_nan=False), encoding='utf-8')
        # The declared input contract belongs in the manifest. Without it the promotion
        # gate could see which feature VERSION an artifact claimed but not which columns it
        # actually reads, so an artifact naming a column nothing can compute was promoted
        # and then served with that column silently absent.
        manifest={'version':version,'created_at':int(time.time()*1000),'sha256':hashlib.sha256(payload).hexdigest(),'metrics':metrics,'dataset':dataset_manifest,'feature_version':model.get('feature_version'),'features':list(model.get('features') or ()),'model_version':model.get('model_version',version),'split':model.get('split'),'status':'candidate','calibrated':valid_calibrator(calibrator)}
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); return manifest
    def promote(self,version, *, min_test_rows=100, max_age_ms=21600000, min_directional_accuracy_pct=50.0,
                allow_calibrator_missing=False):
        target=self.root/version/'manifest.json'
        if not target.exists(): raise FileNotFoundError(version)
        manifest=json.loads(target.read_text(encoding='utf-8'))
        metrics=manifest.get('metrics') or {}
        test=metrics.get('test') or metrics
        if manifest.get('status') not in ('candidate','approved','production'):
            raise ValueError('model_not_promotable')
        if int(test.get('rows',0)) < min_test_rows:
            raise ValueError('insufficient_test_rows')
        if float(test.get('directional_accuracy_pct',0)) < min_directional_accuracy_pct:
            raise ValueError('insufficient_test_accuracy')
        split=manifest.get('split') or {}
        if not split.get('ready') or not split.get('label_intervals_verified'):
            raise ValueError('unverified_label_intervals')
        # The required version is the one the code currently produces. Hardcoding the
        # string 'features-v1' while the feature spec had moved to 'features-v3' made this
        # gate impossible to pass: no candidate could ever be promoted, and the service ran
        # unpromoted weights forever with nothing reporting a problem.
        #
        # Name equality alone is not enough, in both directions.
        #
        # A name that this code cannot compute is refused, because serving would have to
        # leave the column out and hand the model a missing input it was fitted on. The
        # loader does check this (TabularPredictor), but it is checked at serve time, on a
        # path that degrades to a fallback -- which is how an unservable artifact gets
        # promoted and only then reported.
        declared = manifest.get('features') or ()
        unknown = unproducible(declared)
        if unknown:
            raise ValueError('unproducible_declared_features:%s' % ','.join(unknown))
        # A feature version exists precisely so an artifact can name a stable contract
        # while the implementation adds columns. Demanding name equality with whatever
        # the code produces right now makes extending the contract impossible: a genuine
        # v3 artifact served today would be refused, and more importantly a MODEL THAT
        # NAMES A COLUMN THAT DOES NOT EXIST YET would be accepted, because the two
        # strings match. What matters is that the artifact version is one this code still
        # knows and that every column it declares is one this code can fill.
        known = features_for(manifest.get('feature_version'))
        if known is None:
            raise ValueError('incompatible_feature_version')
        if declared and not set(declared) <= set(known):
            outside = sorted(set(declared) - set(known))
            raise ValueError('features_outside_declared_version:%s' % ','.join(outside))
        evidence=metrics.get('portfolio_oos') or {}
        if (evidence.get('costs_included') is not True or evidence.get('trades',0) < 30
                or not all(math.isfinite(float(evidence.get(k,float('nan')))) for k in ('net_return','max_drawdown'))
                or evidence.get('net_return',0) <= 0 or not 0 <= evidence.get('max_drawdown',1) <= .2):
            raise ValueError('insufficient_portfolio_evidence')
        if max_age_ms is None or max_age_ms <= 0:
            raise ValueError('expiration_required')
        payload=(target.parent/'model.json').read_bytes()
        if hashlib.sha256(payload).hexdigest()!=manifest.get('sha256'):
            raise ValueError('model_checksum_mismatch')
        if not math.isfinite(float(test.get('directional_accuracy_pct',0))):
            raise ValueError('non_finite_metrics')
        # The probabilities the decision layer reports are only meaningful if a calibrator
        # was fit and shipped alongside the weights. Nothing produced one, so every
        # 'probability' was a linear rescaling of the predicted return -- a number with no
        # defined meaning. Promoting such a model silently is what let that go unnoticed.
        calibration=metrics.get('calibration') or {}
        if not calibration.get('fitted') and not allow_calibrator_missing:
            raise ValueError('calibrator_missing')
        # The flag and the file, checked together. Either alone is satisfiable while the
        # model is unusable: a flag with no file promotes something the loader will refuse,
        # and a file with no flag is refused here for no reason.
        if not allow_calibrator_missing:
            from .calibration import load_calibrator

            if load_calibrator(self.root, version) is None:
                raise ValueError('calibrator_artifact_missing')
        now=int(time.time()*1000)
        if max_age_ms is not None and now-int(manifest.get('created_at',0)) > int(max_age_ms):
            raise ValueError('model_expired')
        manifest['status']='production'; manifest['promoted_at']=now
        target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        (self.root/'production.json').write_text(json.dumps({'version':version,'promoted_at':now,'expires_at':now+int(max_age_ms) if max_age_ms else None},indent=2),encoding='utf-8')
        return version
    def current(self, now_ms=None):
        path=self.root/'production.json'
        if not path.exists():
            return None
        current=json.loads(path.read_text(encoding='utf-8'))
        expires_at=current.get('expires_at')
        if expires_at and int(now_ms or time.time()*1000) >= int(expires_at):
            return {**current,'status':'expired'}
        return {**current,'status':'active'}

    def load_production(self, now_ms=None):
        current=self.current(now_ms)
        if not current:
            raise ValueError('no_production_model')
        if current.get('status') != 'active':
            raise ValueError('production_model_expired')
        model_path=self.root/current['version']/'model.json'
        manifest_path=self.root/current['version']/'manifest.json'
        if not model_path.exists() or not manifest_path.exists():
            raise ValueError('production_model_missing')
        payload=model_path.read_bytes()
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if hashlib.sha256(payload).hexdigest() != manifest.get('sha256'):
            raise ValueError('production_model_checksum_mismatch')
        return json.loads(payload.decode('utf-8')), manifest
