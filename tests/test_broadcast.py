import asyncio
import json
from types import SimpleNamespace

from app.core import app_keys
from app.web.api_routes import build_app


def _app(equity=100):
    runtime = SimpleNamespace(session=object(), broker=object())
    return build_app(runtime,
                     state_fn=lambda: {"equity": equity},
                     realtime_state_fn=lambda: {"equity": equity},
                     chart_fn=lambda symbol: {},
                     on_session_start_fn=lambda session_id: None,
                     close_position_fn=lambda symbol: None,
                     reset_risk_fn=lambda: None,
                     model_state_fn=lambda: {})


def test_slow_socket_does_not_delay_fast_delivery_or_duplicate_broadcast():
    async def run():
        received = asyncio.Event()

        class Socket:
            def __init__(self, slow=False):
                self.slow = slow
                self.messages = []
                self.closed = False

            async def send_str(self, payload):
                if self.slow:
                    await asyncio.Event().wait()
                self.messages.append(json.loads(payload))
                received.set()

            async def close(self):
                self.closed = True

        app = _app()
        fast, slow = Socket(), Socket(True)
        app[app_keys.SUBSCRIBERS].update((fast, slow))
        first = asyncio.create_task(app[app_keys.BROADCAST]())
        await asyncio.wait_for(received.wait(), .2)
        await app[app_keys.BROADCAST]()
        await first
        assert fast.messages == [{"equity": 100}]
        assert slow.closed and slow not in app[app_keys.SUBSCRIBERS]
        assert fast in app[app_keys.SUBSCRIBERS]

    asyncio.run(run())


def test_route_handlers_are_not_shadowed_by_data_providers():
    # A lambda bound to the name "state" once replaced the /api/state view and every
    # request failed with a TypeError; assert the routes resolve to real handlers.
    app = _app()
    paths = {resource.canonical for resource in app.router.resources()}
    for path in ("/api/state", "/api/chart/{symbol}", "/api/positions/{symbol}/close"):
        assert any(path in p for p in paths), (path, sorted(paths))


def test_state_endpoint_reports_equity():
    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        client = TestClient(TestServer(_app(equity=123)))
        await client.start_server()
        try:
            async with client.get("/api/state") as response:
                assert response.status == 200
                assert (await response.json())["equity"] == 123
        finally:
            await client.close()

    asyncio.run(run())
