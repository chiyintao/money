import asyncio

from app.storage.persistence import BatchWriter


def test_batch_writer_flushes_by_size_and_time():
    async def scenario():
        batches = []
        writer = BatchWriter(batches.append, batch_size=2, flush_interval=.01)
        writer.start()
        assert writer.submit_nowait(1)
        assert writer.submit_nowait(2)
        await asyncio.wait_for(writer.drain(), timeout=1)
        assert batches == [[1, 2]]
        assert writer.submit_nowait(3)
        await writer.stop()
        assert batches == [[1, 2], [3]]
        assert writer.stats()['flushed'] == 3
    asyncio.run(scenario())


def test_batch_writer_reports_backpressure():
    writer = BatchWriter(lambda batch: None, maxsize=1)
    assert writer.submit_nowait(1)
    assert not writer.submit_nowait(2)
    assert writer.stats()['dropped'] == 1


def test_batch_writer_surfaces_persistence_failure():
    async def scenario():
        def fail(_batch):
            raise OSError('disk-full')
        writer = BatchWriter(fail, batch_size=1, flush_interval=.01)
        task = writer.start()
        writer.submit_nowait(1)
        try:
            await task
        except OSError:
            pass
        assert writer.stats()['failures'] == 1
        assert writer.stats()['last_error'] == "OSError('disk-full')"
        try:
            writer.submit_nowait(2)
        except RuntimeError as exc:
            assert str(exc) == 'writer_failed'
        else:
            raise AssertionError('failed writer accepted new work')
    asyncio.run(scenario())


def test_drain_waits_for_writer_and_propagates_failure():
    import pytest
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked(_batch):
            entered.set()
            await release.wait()
            raise OSError('disk-full')
        writer = BatchWriter(blocked, batch_size=1)
        writer.start()
        writer.submit_nowait(1)
        await entered.wait()
        drain = asyncio.create_task(writer.drain())
        await asyncio.sleep(0)
        assert not drain.done()
        release.set()
        with pytest.raises(OSError):
            await asyncio.wait_for(drain, 1)
    asyncio.run(scenario())


def test_database_worker_owns_separate_thread_and_commits(tmp_path):
    from app.storage.database_worker import DatabaseWorker
    from app.storage.storage import Store
    import threading
    async def scenario():
        worker = DatabaseWorker(str(tmp_path))
        main_thread = threading.get_ident()
        try:
            await worker.call('set_runtime', 'test', {'value': 7})
            worker_thread = await asyncio.get_running_loop().run_in_executor(worker.executor, threading.get_ident)
            assert worker_thread != main_thread
            reader = Store(str(tmp_path))
            try:
                assert reader.get_runtime('test') == {'value': 7}
            finally:
                reader.close()
        finally:
            await worker.close()
    asyncio.run(scenario())

