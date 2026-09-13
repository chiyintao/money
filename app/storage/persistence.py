import asyncio
from collections.abc import Callable


class BatchWriter:
    """Asynchronously flush bounded items in batches with explicit loss accounting."""

    def __init__(self, writer: Callable, *, maxsize=4096, batch_size=100, flush_interval=0.1):
        if int(maxsize) <= 0 or int(batch_size) <= 0 or float(flush_interval) <= 0:
            raise ValueError('invalid_writer_config')
        self.writer = writer
        self.queue = asyncio.Queue(maxsize=int(maxsize))
        self.batch_size = int(batch_size)
        self.flush_interval = float(flush_interval)
        self.accepted = 0
        self.dropped = 0
        self.flushed = 0
        self.failures = 0
        self.last_error = None
        self._task = None
        self._stopping = False

    def start(self):
        if self._task is not None and not self._task.done():
            raise RuntimeError('writer_already_started')
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        return self._task

    def submit_nowait(self, item):
        if self._stopping:
            if self.last_error:
                raise RuntimeError('writer_failed') from RuntimeError(self.last_error)
            raise RuntimeError('writer_stopping')
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        self.accepted += 1
        return True

    async def _flush(self, batch):
        if not batch:
            return
        try:
            result = self.writer(batch)
            if hasattr(result, '__await__'):
                await result
            self.flushed += len(batch)
            for _ in batch:
                self.queue.task_done()
        except Exception as exc:
            self.failures += len(batch)
            self.last_error = repr(exc)
            self._stopping = True
            raise

    async def _run(self):
        batch = []
        loop = asyncio.get_running_loop()
        deadline = None
        while not self._stopping or not self.queue.empty():
            if batch and (len(batch) >= self.batch_size or loop.time() >= deadline):
                await self._flush(batch)
                batch, deadline = [], None
                continue
            timeout = max(0.000001, deadline-loop.time()) if deadline else self.flush_interval
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            if not batch:
                deadline = loop.time() + self.flush_interval
            batch.append(item)
        if batch:
            await self._flush(batch)

    async def stop(self):
        self._stopping = True
        if self._task is not None:
            await self._task
            self._task = None

    async def drain(self):
        if self._task is None:
            raise RuntimeError('writer_not_started')
        join = asyncio.create_task(self.queue.join())
        try:
            done, _ = await asyncio.wait((join, self._task), return_when=asyncio.FIRST_COMPLETED)
            if self._task in done:
                await self._task
            await join
        finally:
            if not join.done():
                join.cancel()
                await asyncio.gather(join, return_exceptions=True)

    def stats(self):
        return {'accepted': self.accepted, 'dropped': self.dropped, 'flushed': self.flushed, 'failures': self.failures, 'last_error': self.last_error, 'queued': self.queue.qsize(), 'stopping': self._stopping}
