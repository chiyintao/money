"""The Prometheus endpoint answered 500 to every scrape.

`web.metrics` built its response with

    content_type='text/plain; version=0.0.4; charset=utf-8'

and aiohttp raises ValueError("charset must not be in content_type argument") for that
shape -- the charset has to be passed as its own argument. Every request to /metrics
therefore returned 500, and a failed scrape is the quietest failure in the system: no
alert fires, the dashboard is unaffected, and the only symptom is an empty graph.

Measured against the running service: 500 before, 200 with 103 lines after.
"""
import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.core import app_keys
from app.web.web import metrics


def _app(state=None):
    application = web.Application()
    application[app_keys.STATE] = state or (lambda: {'equity': 12345.0, 'cash': 100.0})
    application.router.add_get('/metrics', metrics)
    return application


def _get(application):
    async def run():
        client = TestClient(TestServer(application))
        await client.start_server()
        try:
            async with client.get('/metrics') as response:
                return response.status, await response.text(), dict(response.headers)
        finally:
            await client.close()
    return asyncio.run(run())


def test_metrics_returns_two_hundred():
    """The regression itself."""
    status, _body, _headers = _get(_app())
    assert status == 200, 'a scraper must not receive 500'


def test_metrics_returns_a_prometheus_body():
    """Not just 200: a parseable exposition with the text version declared."""
    status, body, headers = _get(_app())
    assert status == 200
    ctype = headers.get('Content-Type', '').lower()
    assert 'text/plain' in ctype
    assert 'charset=utf-8' in ctype
    assert 'version=0.0.4' not in ctype, (
        'the version belongs in a header: aiohttp rejects it inside content_type')
    assert headers.get('X-Prometheus-Text-Version') == '0.0.4'
    assert '# HELP' in body or '# TYPE' in body, 'the body must be an exposition'


def test_metrics_reflects_the_state_it_is_given():
    """A scrape that ignores the account is worse than no scrape."""
    state = {'equity': 54321.0, 'account': {'cash': 7.0}}
    status, body, _headers = _get(_app(lambda: state))
    assert status == 200
    assert 'paper_equity 54321' in body
    assert 'paper_cash 7' in body
