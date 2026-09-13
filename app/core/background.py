"""Small owner for fire-and-forget background tasks."""
import asyncio


class BackgroundTasks:
    """Track spawned tasks so shutdown can cancel them and failures get logged."""

    def __init__(self, logger):
        self.logger = logger
        self.tasks = set()

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)

        def finished(done):
            self.tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                self.logger.error('background task failed: %r', done.exception())

        task.add_done_callback(finished)
        return task

    async def cancel_all(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
