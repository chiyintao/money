"""Serialize database operations on one thread with its own SQLite connection."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from .storage import Store


class DatabaseWorker:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paper-db')
        self.store = None
        self.closed = False
        self.allowed = {'record_events', 'upsert_candles_batch', 'set_runtime',
                        'save_order', 'record_trade', 'record_equity', 'record_derivatives',
                        'prune_events', 'prune_archive'}

    def _call(self, method, args):
        if self.store is None:
            self.store = Store(self.data_dir)
        return getattr(self.store, method)(*args)

    async def call(self, method, *args):
        if self.closed or method not in self.allowed:
            raise RuntimeError('database_operation_unavailable')
        snapshot = copy.deepcopy(args)
        return await asyncio.get_running_loop().run_in_executor(self.executor, self._call, method, snapshot)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        def close_connection():
            if self.store is not None:
                self.store.close()
        await asyncio.get_running_loop().run_in_executor(self.executor, close_connection)
        self.executor.shutdown(wait=True)
