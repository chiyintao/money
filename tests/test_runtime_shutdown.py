import asyncio
from types import SimpleNamespace
from app import main
from app.market import websocket_feed
from app.core.config import Settings


def test_runtime_cancellation_cleans_tasks(tmp_path, monkeypatch):
    async def scenario():
        ready=asyncio.Event()
        class API:
            session=None
            def __init__(self, *args): pass
            async def market_snapshot(self):
                await asyncio.Event().wait()
            async def price_snapshot(self):
                await asyncio.Event().wait()
            async def mark_snapshot(self):
                await asyncio.Event().wait()
        class Site:
            def __init__(self, *args): pass
            async def start(self): ready.set()
        async def feed_run(self):
            await asyncio.Event().wait()
        settings=Settings(data_dir=str(tmp_path))
        monkeypatch.setattr(main, 'Settings', lambda:settings)
        monkeypatch.setattr(main, 'BinancePublic', API)
        monkeypatch.setattr(main.web, 'TCPSite', Site)
        monkeypatch.setattr(websocket_feed.BinanceWebSocketFeed, 'run', feed_run)
        monkeypatch.setattr(main, 'BinanceWebSocketFeed', websocket_feed.BinanceWebSocketFeed, raising=False)
        baseline=set(asyncio.all_tasks())
        task=asyncio.create_task(main.run())
        await asyncio.wait_for(ready.wait(),2)
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        await asyncio.sleep(0)
        remaining=[t for t in asyncio.all_tasks() if t not in baseline and not t.done()]
        assert not remaining
    asyncio.run(scenario())
