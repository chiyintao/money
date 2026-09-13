"""Labels, the archive, and the gates that could not fire.

Three failures with one shape: something reported success while being incapable of doing
its job. A label measured from a price no order could reach; an archive that was written to
and never read, growing without bound; an out-of-distribution gate whose bounds were the
clip range its own inputs were clipped into.
"""
import json
import math
import random

import pytest

from app.models.dataset_io import bounds_are_informative, stream_profile
from app.features.feature_spec import FEATURE_CLIP, FEATURES, LABEL_VERSION
from app.storage.storage import Store


# ------------------------------------------------------------- label timing
def _bars(count=60, interval_ms=300_000, seed=3):
    rng = random.Random(seed)
    price = 100.0
    bars = []
    for index in range(count):
        opened = price
        price = price * (1 + rng.gauss(0.0005, 0.004))
        bars.append({'open_time': index * interval_ms, 'close_time': index * interval_ms + interval_ms - 1,
                     'open': opened, 'high': max(opened, price) * 1.001,
                     'low': min(opened, price) * 0.999, 'close': price,
                     'volume': 100.0, 'is_closed': True})
    return bars


WINDOW = 60


def test_the_label_enters_at_the_next_bar_open(tmp_path):
    # label-v1 measured close[i+h]/close[i] - 1, a price the live loop can never fill at:
    # it decides on a closed bar and the order fills at whatever the market offers next.
    # The bias ran in the direction of every label and no cost estimate can remove it,
    # because it is not a cost, it is a different price.
    from app.models.train import build_dataset

    store = Store(tmp_path)
    try:
        bars = _bars(200)
        store.upsert_candles('BTCUSDT', '5m', bars)
        result = build_dataset(str(tmp_path), symbols=['BTCUSDT'], interval='5m',
                               window=WINDOW, horizon=3, output=tmp_path / 'set.jsonl')
        assert result['label_version'] == LABEL_VERSION
        assert result['label_entry'] == 'next_bar_open'
        with open(result['output'], encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    finally:
        store.close()
    assert rows
    row = rows[0]
    # Locate the bar the features were built from, by its close time.
    index = next(i for i, bar in enumerate(bars) if bar['close_time'] == row['timestamp'])
    assert row['entry_time'] == bars[index + 1]['open_time']
    assert row['entry_price'] == pytest.approx(bars[index + 1]['open'])
    expected = bars[index + 1 + 3]['close'] / bars[index + 1]['open'] - 1
    assert row['future_return'] == pytest.approx(max(-0.2, min(0.2, expected)))


def test_the_label_interval_does_not_overlap_the_next_decision(tmp_path):
    from app.models.train import build_dataset

    store = Store(tmp_path)
    try:
        bars = _bars(200)
        store.upsert_candles('BTCUSDT', '5m', bars)
        result = build_dataset(str(tmp_path), symbols=['BTCUSDT'], interval='5m',
                               window=WINDOW, horizon=6, output=tmp_path / 'set.jsonl')
        with open(result['output'], encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    finally:
        store.close()
    for row in rows:
        assert row['label_end_time'] > row['entry_time'] > row['timestamp']
        # The label must not resolve before the entry the live path would have got.
        assert row['entry_time'] - row['timestamp'] <= 300_000


# ------------------------------------------------------------ edge bucketing
def test_the_edge_bucket_follows_the_configured_interval():
    # The de-duplication key was the literal 300000, so on a 1m configuration every
    # decision was folded into a five-minute bucket and an arbitrary one was kept.
    from app.core.config import interval_to_ms

    assert interval_to_ms('1m') == 60_000
    assert interval_to_ms('15m') == 900_000
    assert interval_to_ms('5m') == 300_000
    # Unknown intervals fall back rather than raising inside the measurement path.
    assert interval_to_ms('7q') > 0


# -------------------------------------------------------------- the archive
def _event(symbol, at, kind='strategy_decision'):
    # A distinct event_time per row: retention keeps every row that shares the cutoff
    # millisecond, so a fixture with one timestamp cannot exercise a row window.
    from app.core.domain import Event
    return Event(kind, {'symbol': symbol, 'session_id': 's1', 'at': at,
                        'event_time': 1_000_000 + at}).json()


def test_the_archive_is_bounded_and_readable(tmp_path):
    store = Store(tmp_path)
    try:
        for index in range(400):
            store.record_event(_event('A', 1000 + index))
        store.prune_events(100)
        counts = store.event_counts()
        assert counts['events'] == 100
        assert counts['archived'] == 300

        # Retention claims nothing is lost; before this reader existed the rows it moved
        # were unreachable, so the claim was false in every practical sense.
        recovered = store.archived_events(limit=10)
        assert len(recovered) == 10
        assert recovered[-1]['payload']['at'] > recovered[0]['payload']['at']
        assert store.archived_events(symbol='A', limit=5)[0]['payload']['symbol'] == 'A'
        assert store.archived_events(symbol='NOPE') == []

        # And the archive is itself bounded, which it never was: it accumulated every row
        # retention had ever moved, 1.02M and rising, with nothing to read or remove them.
        assert store.prune_archive(50) == 250
        assert store.event_counts()['archived'] == 50
    finally:
        store.close()


def test_pruning_the_archive_keeps_the_newest_rows(tmp_path):
    store = Store(tmp_path)
    try:
        for index in range(200):
            store.record_event(_event('A', 1000 + index))
        store.prune_events(0)
        assert store.event_counts()['archived'] == 200
        store.prune_archive(20)
        kept = store.archived_events(limit=100)
        assert len(kept) == 20
        assert kept[-1]['payload']['at'] == 1199
    finally:
        store.close()


def test_archive_retention_does_nothing_when_disabled(tmp_path):
    store = Store(tmp_path)
    try:
        for index in range(30):
            store.record_event(_event('A', 1000 + index))
        store.prune_events(0)
        assert store.prune_archive(0) == 0
        assert store.event_counts()['archived'] == 30
    finally:
        store.close()


# ------------------------------------------------------------ the OOD gate
def test_a_bound_equal_to_the_clip_range_cannot_refuse_anything():
    # The served bounds are read off the training data, and the training data has already
    # been clipped to FEATURE_CLIP. A feature that ever touches its limit therefore gets a
    # bound equal to the clip range, and the serving gate then tests values its own
    # pipeline has clipped into that range. It was unreachable for four of the ten
    # features in the stored dataset and reported green for all ten.
    saturated = {name: list(limits) for name, limits in FEATURE_CLIP.items()}
    quality = bounds_are_informative(saturated)
    assert quality['informative'] is False
    assert set(quality['degenerate']) == set(FEATURE_CLIP)


def test_the_gate_skips_a_bound_it_cannot_use(tmp_path):
    from app.strategy.live_models import RealModelRuntime

    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False)
    runtime.feature_space = {
        'status': 'ok',
        'bounds': {'rsi': [0.0, 100.0], 'ema20_gap': [-0.3, 0.3]},
        'bounds_quality': {'informative': False, 'degenerate': ['rsi']},
    }
    # rsi is at its bound and outside would-be limits, but the bound is the clip range, so
    # it is not evidence of anything.
    assert runtime.out_of_range({'rsi': 100.0, 'ema20_gap': 0.0}) == []
    assert runtime.out_of_range({'rsi': 50.0, 'ema20_gap': 0.9}) == ['ema20_gap']
    # A feature present in training and absent at serving is itself out of distribution.
    assert runtime.out_of_range({'rsi': 50.0}) == ['ema20_gap']


def test_the_serving_gate_can_fire_on_the_stored_bounds_shape(tmp_path):
    from app.strategy.live_models import RealModelRuntime

    runtime = RealModelRuntime(str(tmp_path), chronos_enabled=False)
    runtime.feature_space = {
        'status': 'ok', 'bounds': {'return_10': [-0.12, 0.11]},
        'bounds_quality': {'informative': True, 'degenerate': []},
    }
    assert runtime.out_of_range({'return_10': 0.05}) == []
    assert runtime.out_of_range({'return_10': 0.4}) == ['return_10']


def test_bounds_survive_a_profile_round_trip(tmp_path):
    rows = []
    rng = random.Random(1)
    for index in range(500):
        # Values drawn inside each feature's clip range, which is what the pipeline
        # actually produces: the profile is read off already-clipped data.
        item = {}
        for name in FEATURES:
            low, high = FEATURE_CLIP[name]
            centre = (low + high) / 2
            spread = (high - low) / 2 * 0.8
            item[name] = min(high, max(low, rng.gauss(centre, spread / 3)))
        item['symbol'] = 'BTCUSDT'
        item['timestamp'] = index * 300_000
        rows.append(item)
    path = tmp_path / 'set.jsonl'
    path.write_text(chr(10).join(json.dumps(item, sort_keys=True) for item in rows), encoding='utf-8')
    profile = stream_profile(str(path))
    assert profile['bounds_method'].startswith('tail_')
    assert profile['rows'] == 500
    quality = bounds_are_informative(profile['bounds'])
    assert quality['informative'] is True


# --------------------------------------------------------------- health depth
def test_the_health_snapshot_says_whether_the_ood_gate_is_informative():
    from app.backtest.snapshot import _feature_space, _derivatives

    class Models:
        feature_space = {'status': 'ok', 'verified': True, 'rows': 100, 'sha256': 'abc',
                         'bounds_method': 'tail_0.001',
                         'bounds_quality': {'informative': False, 'degenerate': ['rsi']}}

    runtime = type('R', (), {'models': Models()})()
    view = _feature_space(runtime)
    assert view['verified'] is True
    assert view['ood_gate_informative'] is False
    assert view['ood_degenerate_features'] == ['rsi']
    assert view['bounds_method'] == 'tail_0.001'


def test_health_reports_an_absent_collector_rather_than_omitting_it():
    from app.backtest.snapshot import _derivatives

    assert _derivatives(type('R', (), {})()) == {'enabled': False}

    class Collector:
        def health(self):
            return {'runs': 3, 'rows': 90}

    view = _derivatives(type('R', (), {'derivatives_collector': Collector()})())
    assert view == {'enabled': True, 'runs': 3, 'rows': 90}
