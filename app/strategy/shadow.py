"""Bounded coalescing research inference that reuses the trading model runtime."""
import asyncio
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .live_models import RealModelRuntime


class ShadowModels:
    """Research-only prediction worker.

    It shares one RealModelRuntime with the trading decision path, so the real
    weights are loaded exactly once. Predictions here are observations: they never
    size a position and never submit an order.
    """

    def __init__(self, data_dir='data', device='cpu', max_symbols=20, runtime=None):
        self.root = Path(data_dir)
        self.device = device
        self.max_symbols = max_symbols
        self.pending = {}
        self.results = {}
        self.seen = {}
        self.current_symbol = ''
        self.runtime = runtime if runtime is not None else RealModelRuntime(data_dir, device=device)
        self._owns_runtime = runtime is None
        self.event = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='shadow-models')
        self.task = None
        self.stopping = False

    @property
    def load_errors(self):
        return self.runtime.load_errors

    def _predict(self, rows):
        # current_symbol is assigned by the single-threaded worker right before each
        # awaited dispatch, so this is race-free by construction.
        return self.runtime.research_predict(self.current_symbol, rows)

    def submit(self, symbol, rows):
        closed = [r for r in rows if r.get('is_closed')]
        if self.stopping or len(closed) < 50:
            return False
        stamp = closed[-1]['close_time']
        if stamp <= self.seen.get(symbol, -1):
            return False
        if symbol not in self.seen and len(self.seen) >= self.max_symbols:
            return False
        self.seen[symbol] = stamp
        self.pending[symbol] = copy.deepcopy(closed[-512:])
        self.event.set()
        return True

    async def _run(self):
        while not self.stopping:
            await self.event.wait()
            while self.pending and not self.stopping:
                symbol = next(iter(self.pending))
                rows = self.pending.pop(symbol)
                started = time.monotonic()
                self.current_symbol = symbol
                try:
                    result = await asyncio.get_running_loop().run_in_executor(self.executor, self._predict, rows)
                except Exception as exc:
                    result = {'error': repr(exc), 'predictions': {}}
                self.results[symbol] = {'symbol': symbol, 'mode': 'shadow', 'bar_time': rows[-1]['close_time'],
                                        'completed_at': int(time.time() * 1000), 'latency_ms': (time.monotonic()-started)*1000, **result}
            self.event.clear()

    def start(self):
        if self.task is not None:
            raise RuntimeError('shadow_already_started')
        self.task = asyncio.create_task(self._run())

    async def close(self):
        self.stopping = True
        self.event.set()
        if self.task:
            await self.task
        self.executor.shutdown(wait=True)
        if self._owns_runtime:
            await self.runtime.close()

    def status(self):
        now = int(time.time()*1000)
        return {'production': None, 'mode': 'shadow', 'queued': len(self.pending),
                'models': [{**r, 'data_age_ms': now-r['bar_time']} for r in self.results.values()]}
