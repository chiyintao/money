"""The mark pump must judge data age, not round-trip time.

The bug these lock down: the freshness check compared the venue's own timestamp against a
clock read *after* the response arrived, so the difference was the request's duration. On
a link where /fapi/v1/premiumIndex takes ~5.1s that made the 5s window reject every row --
pending stayed empty, the derivatives table went 47 hours without a write, four funding
features were permanently degraded, and every decision was vetoed to FLAT.
"""
import asyncio
import sys

import pytest

from app.market import pumps


class _Response(dict):
    """One mark row as the venue returns it.

    A dict, not an object: the pump reads rows with row['symbol'] and row.get(...).
    """

    def __init__(self, symbol, stamp, mark=100.0, funding=0.0001):
        super().__init__(symbol=symbol, time=stamp, markPrice=mark,
                         lastFundingRate=funding)


class _Api:
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    async def mark_snapshot(self):
        self.calls += 1
        # A slow venue: the response is built now, but the caller sees it much later.
        await asyncio.sleep(0)
        return self._rows


class _Store:
    def __init__(self):
        self.written = []

    def record_derivatives(self, rows):
        self.written.extend(rows)


class _Errors:
    def __init__(self):
        self.notes = []

    def note(self, key, exc=None):
        self.notes.append((key, exc))

    def ok(self, key):
        pass


class _Session:
    status = 'running'

    def mark(self, *a, **k):
        pass


class _Runtime:
    def __init__(self, rows, known):
        self.api = _Api(rows)
        self.store = _Store()
        self.errors = _Errors()
        self.session = _Session()
        self.cache = type('C', (), {'latest': {s: {'symbol': s} for s in known}})()
        self.account = type('A', (), {'positions': {}})()
        # The pump records the arrival of each mark here; it is how "this symbol has a
        # live book" is tracked, and it is written before the persist step.
        self.feeds = type('F', (), {'last_mark_event': {}})()

    def feature_source(self):
        return type('F', (), {'invalidate': lambda self: None})()


def _run_one_pass(runtime):
    """Drive the pump loop until it has written once, then stop it.

    The loop is written as an infinite 'while True: ... await sleep(2)', so the test
    bounds it by outcome rather than by iteration count.
    """
    async def drive():
        task = asyncio.ensure_future(pumps.mark_pump(runtime))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if runtime.store.written or runtime.api.calls > 1:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())


def test_stamp_is_compared_against_the_clock_read_before_the_request(monkeypatch):
    """A stamp older than the skew bound is refused; one inside it is kept."""
    now = 1_700_000_000_000
    monkeypatch.setattr(pumps.time, 'time', lambda: now / 1000.0)
    old = _Response('BTCUSDT', now - pumps.MARK_STAMP_SKEW_MS - 1)
    fresh = _Response('ETHUSDT', now - 1000)
    runtime = _Runtime([old, fresh], {'BTCUSDT', 'ETHUSDT'})
    _run_one_pass(runtime)
    written = {row['symbol'] for row in runtime.store.written}
    assert 'ETHUSDT' in written
    assert 'BTCUSDT' not in written


def test_a_slow_request_does_not_discard_the_batch(monkeypatch):
    """The regression itself: the stamp is current, but the answer takes 30s to arrive.

    Under the old check the batch was dropped whenever the request outran the window.
    """
    start = 1_700_000_000_000
    ticks = iter([start, start + 30_000])          # before the request, then after it
    monkeypatch.setattr(pumps.time, 'time',
                        lambda: next(ticks, start + 30_000) / 1000.0)
    row = _Response('BTCUSDT', start)
    runtime = _Runtime([row], {'BTCUSDT'})
    _run_one_pass(runtime)
    assert [r['symbol'] for r in runtime.store.written] == ['BTCUSDT']


def test_unknown_symbols_are_still_skipped(monkeypatch):
    """The cache filter is a separate concern and must keep working."""
    now = 1_700_000_000_000
    monkeypatch.setattr(pumps.time, 'time', lambda: now / 1000.0)
    runtime = _Runtime([_Response('NOTWATCHED', now)], {'BTCUSDT'})
    _run_one_pass(runtime)
    assert runtime.store.written == []


def test_skew_bound_is_generous_enough_for_observed_venue_drift():
    """The venue's clock was measured ~6.8s ahead; the bound must absorb that."""
    assert pumps.MARK_STAMP_SKEW_MS >= 30_000
