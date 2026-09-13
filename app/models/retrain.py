import asyncio
import time
from .training_policy import retrain_required


class RetrainScheduler:
    """Small supervised scheduler; training is explicit and never auto-promoted."""

    def __init__(self, train_fn, manifest_fn, *, interval_ms=6 * 60 * 60 * 1000,
                 feature_version=None, drift_fn=None):
        self.train_fn = train_fn
        self.manifest_fn = manifest_fn
        # Injected rather than imported: the scheduler is constructed before the monitor
        # exists in some deployments, and a missing monitor must read as "not checked"
        # rather than as "clean".
        self.drift_fn = drift_fn
        self.interval_ms = int(interval_ms)
        self.feature_version = feature_version
        self.last_run_ms = 0
        self.last_status = 'never_run'
        self.last_reason = 'not_evaluated'
        self._task = None
        self._stopping = False

    def drifted_features(self):
        """The drifted columns the monitor is holding, or None when there is no verdict."""
        if self.drift_fn is None:
            return None
        try:
            report = self.drift_fn() or {}
        except Exception:
            return None
        if report.get('status') not in ('drift', 'ok'):
            return None
        return list(report.get('drifted') or [])

    def evaluate(self, *, now_ms=None, dataset_hash=None):
        manifest = self.manifest_fn()
        required, reason = retrain_required(
            manifest, now_ms=now_ms, max_age_ms=self.interval_ms,
            current_dataset_hash=dataset_hash, current_feature_version=self.feature_version,
            drifted=self.drifted_features())
        self.last_reason = reason
        return required, reason

    async def run_once(self, *, now_ms=None, dataset_hash=None):
        required, reason = self.evaluate(now_ms=now_ms, dataset_hash=dataset_hash)
        if not required:
            self.last_status = 'fresh'
            return {'status': 'skipped', 'reason': reason}
        result = self.train_fn()
        if hasattr(result, '__await__'):
            result = await result
        self.last_run_ms = int(now_ms or time.time() * 1000)
        self.last_status = result.get('status', 'unknown') if isinstance(result, dict) else 'completed'
        return {'status': 'trained', 'reason': reason, 'result': result}

    async def _run(self):
        while not self._stopping:
            try:
                await self.run_once()
            except Exception as exc:
                self.last_status = 'failed'
                self.last_reason = repr(exc)
            await asyncio.sleep(min(self.interval_ms / 1000, 60))

    def start(self):
        if self._task is not None and not self._task.done():
            raise RuntimeError('scheduler_already_started')
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        return self._task

    async def stop(self):
        self._stopping = True
        if self._task is not None:
            await self._task
            self._task = None

    def snapshot(self):
        return {'last_run_ms': self.last_run_ms, 'last_status': self.last_status, 'last_reason': self.last_reason, 'interval_ms': self.interval_ms, 'feature_version': self.feature_version, 'drift_wired': self.drift_fn is not None, 'stopping': self._stopping}
