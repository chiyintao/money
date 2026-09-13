"""One assertion that would have caught the four dead features.

The service trained a model on ten features and served it nine-plus-a-constant. No test
noticed because nothing compared the two paths: the training job had its own funding
lookup and the live path had none, and every aggregate the dashboard reported looked
plausible from either side.

These tests compare the two paths directly on the same bar, using the same store. If a
future change makes one path compute something the other does not, they fail.
"""
import asyncio
import json

import pytest

from app.features.feature_source import FUNDING_FEATURES, FeatureSource, funding_age_ms
from app.features.feature_spec import FEATURES, FEATURES_V3, FLOW_FEATURES, ORDER_FLOW_FEATURES, POSITIONING_FEATURES
from app.strategy.live_models import ModelDecision
from app.market.funding import funding_features, funding_series, mark_series
from app.features.features import snapshot
from app.storage.storage import Store
from app.models.train import build_dataset

HOUR = 3_600_000
BAR = 5 * 60_000


def _bars(count=120, start=1_700_000_000_000):
    """Closed 5m candles shaped like the ingest output.

    Including indices 7, 8 and 9. Binance has returned quote volume, trade count and taker
    buy volume on every kline since the endpoint existed; the reader dropped them, so a
    dataset built from a response that already contained taker imbalance had no way to
    express it. A fixture without them tests a database no live ingest produces.
    """
    rows = []
    for index in range(count):
        open_time = start + index * BAR
        close = 100.0 + index * 0.05
        volume = 10.0 + index % 5
        taker_share = 0.45 + 0.1 * ((index % 5) - 2) / 2
        rows.append({'open_time': open_time, 'close_time': open_time + BAR - 1,
                     'open': close - 0.02, 'high': close + 0.05, 'low': close - 0.05,
                     'close': close, 'volume': volume, 'is_closed': True,
                     'quote_volume': volume * close, 'trades': 40 + index % 11,
                     'taker_buy_volume': volume * taker_share})
    return rows


def _seed_flow(store, symbol, bars):
    """Per-minute aggressor and liquidation buckets covering the bars.

    A third source and the one with the shortest life: the endpoints behind it keep thirty
    days and archive nothing. It has to be collected on a schedule, which the collector
    does -- and which nothing consumed, because no feature in the contract mentioned it.
    """
    rows = []
    for index, bar in enumerate(bars):
        traded = 30.0 + index % 7
        liquidated = 5.0 if index % 11 == 0 else 0.0
        rows.append({'symbol': symbol, 'event_time': int(bar['close_time']),
                     'buy_volume': traded * (0.5 + 0.05 * ((index % 5) - 2)),
                     'sell_volume': traded * (0.5 - 0.05 * ((index % 5) - 2)),
                     'trades': 40 + index % 11, 'buy_trades': 20 + index % 5,
                     'notional': traded * float(bar['close']),
                     'largest_trade': 2.0 + index % 3,
                     'delta_volume': traded * 0.1 * ((index % 5) - 2),
                     'delta_ratio': 0.1 * ((index % 5) - 2),
                     'liquidations': 1 if liquidated else 0,
                     'liquidation_long_volume': liquidated,
                     'liquidation_short_volume': 0.0,
                     'liquidation_long_notional': liquidated * float(bar['close']),
                     'liquidation_short_notional': 0.0,
                     'liquidation_delta': liquidated,
                     'liquidation_volume': liquidated,
                     'largest_liquidation': liquidated,
                     'liquidation_share': 0.2 if liquidated else 0.0})
    store.record_flow(rows)
    return rows


def _seed_funding(store, symbol, bars, interval_ms=8 * HOUR):
    """Real funding publications covering the bars, plus contemporaneous mark prices.

    The two have different cadences, which is the point. Funding settles every eight hours
    and carries a mark price from that settlement; used as a basis it is the price change
    since the settlement, and on the shipped dataset that saturated the +/-2% clip for
    9.7% of rows. A deployment therefore stores a five-minute mark as well -- open interest
    value over open interest, which the derivatives collector has always fetched -- and the
    fixture has to do the same or it tests a configuration no deployment runs.
    """
    first = int(bars[0]['open_time'])
    last = int(bars[-1]['close_time'])
    rows = []
    event_time = first - (first % interval_ms) + interval_ms
    rate = 0.00005
    while event_time <= last + 2 * interval_ms:
        rows.append({'symbol': symbol, 'event_time': event_time,
                     'funding_rate': rate, 'mark_price': 100.0 + (event_time - first) / BAR * 0.05})
        rate = round(rate + 0.00001, 8) if rate < 0.0004 else 0.00005
        event_time += interval_ms
    store.record_derivatives(rows)
    marks = []
    for index, row in enumerate(bars):
        close = float(row['close'])
        # A real basis, a few basis points wide and moving, so a test that asserts the
        # feature is not constant is testing the feature rather than the fixture.
        basis = 0.0002 * ((index % 7) - 3)
        # Open interest and the ratio endpoints move too. The collector fetches all of
        # them; a fixture that pins open interest at a constant makes the positioning
        # family a constant, and a test asserting it "moved" would be asserting nothing.
        interest = 1000.0 + (index % 13) * 10
        marks.append({'symbol': symbol, 'event_time': int(row['close_time']),
                      'open_interest': interest,
                      'open_interest_value': interest * close * (1 + basis),
                      'long_short_ratio': 1.5 + 0.1 * ((index % 5) - 2),
                      'top_account_ratio': 1.1 + 0.05 * ((index % 3) - 1),
                      'global_account_ratio': 2.0 + 0.2 * ((index % 7) - 3),
                      'taker_buy_sell_ratio': 1.0 + 0.05 * ((index % 4) - 1),
                      'taker_buy_volume': 5.0 + index % 3,
                      'taker_sell_volume': 5.0 + (index + 1) % 3})
    store.record_derivatives_detail(marks)
    return rows


@pytest.fixture()
def store(tmp_path):
    instance = Store(tmp_path)
    yield instance
    instance.close()


def test_live_and_training_features_are_identical_for_the_same_bar(store):
    """The regression that matters: one bar, two paths, one answer."""
    symbol = 'BTCUSDT'
    bars = _bars()
    _seed_funding(store, symbol, bars)
    for row in bars:
        store.upsert_candles(symbol, '5m', [row])

    bar_index = 90
    bar_time = int(bars[bar_index]['close_time'])
    price = float(bars[bar_index]['close'])

    window = bars[bar_index - 49:bar_index + 1]

    # The training formula, written out exactly as build_dataset applied it before the
    # shared source existed. This is the specification the serving path must now meet.
    times, rates, marks = funding_series(store, symbol)
    expected = snapshot(window, funding_features(times, rates, marks, bar_time, price,
                                                 mark_series=mark_series(store, symbol)))

    live_row, degraded = FeatureSource(store, features=FEATURES_V3).snapshot(
        window, symbol, bar_time, price)
    assert degraded == (), 'the funding family must not fall back to placeholders'
    for name in FEATURES_V3:
        assert live_row[name] == expected[name], name


def test_the_dataset_builder_writes_real_funding_values(tmp_path):
    """End to end: a stored funding history reaches the rows a model is trained on."""
    symbol = 'BTCUSDT'
    bars = _bars(count=200)
    store = Store(tmp_path)
    try:
        for row in bars:
            store.upsert_candles(symbol, '5m', [row])
        _seed_funding(store, symbol, bars)
    finally:
        store.close()
    out = tmp_path / 'dataset.jsonl'
    report = build_dataset(data_dir=tmp_path, symbols=(symbol,), interval='5m',
                           horizon=12, window=50, output=out, strict=True)
    assert report['symbols'][0]['rows'] > 0
    # Named, not counted. The builder reported one number for "some modelled feature was a
    # placeholder", which does not say whether to rebuild the funding history or wait for
    # the flow collector.
    missing = report['symbols'][0]['degraded_features']
    for name in FUNDING_FEATURES:
        assert name not in missing, name
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert len({round(row['funding_rate'], 12) for row in rows}) > 1
    assert len({round(row['funding_z'], 12) for row in rows}) > 1


def test_the_funding_family_is_not_constant_when_history_exists(store):
    # The failure mode was not "wrong value", it was "always zero". A feature that never
    # moves cannot be validated by comparing it to anything, so it is checked directly.
    symbol = 'ETHUSDT'
    bars = _bars()
    _seed_funding(store, symbol, bars)
    source = FeatureSource(store)
    values = {name: set() for name in FUNDING_FEATURES}
    for index in range(60, len(bars)):
        row, degraded = source.snapshot(bars[index - 49:index + 1], symbol,
                                        int(bars[index]['close_time']),
                                        float(bars[index]['close']))
        assert not set(FUNDING_FEATURES) & set(degraded)
        for name in FUNDING_FEATURES:
            values[name].add(round(row[name], 12))
    for name, seen in values.items():
        assert len(seen) > 1, name + ' never moved across 60 bars'


def test_without_funding_history_every_funding_feature_is_reported_degraded():
    # Scoped to a v3 artifact's declared list: that is what the service asks a source for
    # when the weights it loaded name ten columns. A source asked for the whole contract
    # additionally reports the collected families, which its own test covers.
    source = FeatureSource(store=None, features=FEATURES_V3)
    row, degraded = source.snapshot(_bars(), 'BTCUSDT', 1_700_000_000_000, 100.0)
    assert set(degraded) == set(FUNDING_FEATURES)
    assert set(FUNDING_FEATURES) <= set(FEATURES)
    for name in FUNDING_FEATURES:
        assert row[name] == 0.0
    assert source.health()['degraded_rows'] == 1


def test_a_source_with_history_reports_itself_healthy(store):
    symbol = 'BTCUSDT'
    bars = _bars()
    _seed_funding(store, symbol, bars)
    _seed_flow(store, symbol, bars)
    source = FeatureSource(store, features=FEATURES_V3)
    source.snapshot(bars[-50:], symbol, int(bars[-1]['close_time']), float(bars[-1]['close']))
    health = source.health()
    assert health['degraded_rows'] == 0
    assert health['funding_rows'] == 1
    assert health['degraded_features'] == []


def test_degraded_features_block_the_decision(store):
    """A placeholder input is not an out-of-range input, so it needs its own gate."""
    from test_live_models import StubRuntime, rows

    runtime = StubRuntime({'lightgbm': .002, 'catboost': .002})
    decision = ModelDecision(runtime, feature_source=FeatureSource(store=None))
    signal = asyncio.run(decision.signal('BTCUSDT', rows()))
    assert signal['side'] == 'FLAT'
    assert 'feature_degraded' in signal['reason_codes']
    assert 'feature_degraded:funding_rate' in signal['reason_codes']
    # The v3 members declare ten columns, so the four funding names are the whole
    # deficiency -- not the twenty collected ones the model never asked for.
    assert signal['degraded_features'] == list(FUNDING_FEATURES)


def test_the_same_decision_trades_once_its_inputs_are_real(store):
    """The mirror image: identical votes, healthy inputs, so the gate must not fire."""
    from test_live_models import StubRuntime, rows

    runtime = StubRuntime({'lightgbm': .002, 'catboost': .002})
    source = FeatureSource(store)
    bars = rows()
    # rows() steps one millisecond per bar, so an hourly interval puts the first publication
    # at t=3,600,000 -- three and a half million milliseconds after the last bar. None is
    # visible, every funding feature is a placeholder, and this test was passing on those
    # placeholders while asserting that healthy inputs do not trip the gate. The interval
    # has to be scaled to the fixture's clock, and to land inside it: a quarter of the bar
    # count puts several publications before the last bar, which is what the gate needs.
    _seed_funding(store, 'BTCUSDT', bars, interval_ms=max(2, len(bars) // 4))
    decision = ModelDecision(runtime, feature_source=source, degraded_blocks=True)
    signal = asyncio.run(decision.signal('BTCUSDT', bars, price=100.0))
    assert signal['side'] == 'LONG'
    assert 'feature_degraded' not in signal['reason_codes']


def test_ttl_caching_does_not_change_the_answer(store):
    symbol = 'BTCUSDT'
    bars = _bars()
    _seed_funding(store, symbol, bars)
    cached = FeatureSource(store, ttl_ms=60_000)
    first = cached.snapshot(bars[-50:], symbol, int(bars[-1]['close_time']), float(bars[-1]['close']))
    # A new publication lands, but the cache is still inside its window. The value is
    # inside the training clip, so it survives to the feature row unclipped.
    store.record_derivatives([{'symbol': symbol, 'event_time': int(bars[-1]['close_time']),
                               'funding_rate': 0.0009, 'mark_price': 100.0}])
    second = cached.snapshot(bars[-50:], symbol, int(bars[-1]['close_time']), float(bars[-1]['close']))
    assert first[0] == second[0]
    cached.invalidate(symbol)
    third = cached.snapshot(bars[-50:], symbol, int(bars[-1]['close_time']), float(bars[-1]['close']))
    assert third[0]['funding_rate'] == 0.0009


def test_funding_age_is_measured_from_the_visible_publication(store):
    symbol = 'BTCUSDT'
    bars = _bars()
    rows = _seed_funding(store, symbol, bars)
    bar_time = int(bars[-1]['close_time'])
    age = funding_age_ms(store, symbol, bar_time)
    assert age is not None and age >= 0
    assert age < 8 * HOUR
    # Before the first publication nothing is visible and the caller is told so.
    assert funding_age_ms(store, symbol, int(rows[0]['event_time']) - 1) is None
    assert funding_age_ms(None, symbol, bar_time) is None



def test_the_dataset_builder_writes_the_collected_families_nothing_used(tmp_path):
    """Twenty features the system has been collecting for its whole life, in the rows a
    model is trained on.

    app/order_flow.py could compute three families -- candle taker imbalance, open interest
    and long/short positioning, and the per-minute aggressor and liquidation buckets -- and
    no code outside its own tests called it. The collection was wired and the features were
    not, which is the same shape of defect as the mark price that was fetched, stored, and
    never used to compute the basis.

    The assertion is that each one moves. A feature that is present but constant passes
    every "is it in the row" check and carries no information.
    """
    symbol = "BTCUSDT"
    bars = _bars(count=200)
    store = Store(tmp_path)
    try:
        for row in bars:
            store.upsert_candles(symbol, "5m", [row])
        _seed_funding(store, symbol, bars)
        _seed_flow(store, symbol, bars)
    finally:
        store.close()
    out = tmp_path / "dataset.jsonl"
    report = build_dataset(data_dir=tmp_path, symbols=(symbol,), interval="5m",
                           horizon=12, window=50, output=out, strict=True)
    entry = report["symbols"][0]
    assert entry["rows"] > 0
    assert entry["degraded_features"] == {}, entry["degraded_features"]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for name in ORDER_FLOW_FEATURES + POSITIONING_FEATURES + FLOW_FEATURES:
        values = {round(row[name], 12) for row in rows}
        assert len(values) > 1, name + " never moved across the dataset"
    # And the same bar through the serving path produces the same row, because there is
    # one source. Two paths is how four funding features came to be constant at serving
    # time while carrying variance in training.
    source = FeatureSource(store=Store(tmp_path))
    try:
        index = 120
        window = bars[index - 49:index + 1]
        live, degraded = source.snapshot(window, symbol, int(bars[index]["close_time"]),
                                        float(bars[index]["close"]))
    finally:
        source.store.close()
    assert degraded == (), degraded
    trained = next(row for row in rows if row["timestamp"] == int(bars[index]["close_time"]))
    for name in FEATURES:
        assert abs(float(live[name]) - float(trained[name])) < 1e-9, name


def test_a_feature_value_does_not_depend_on_which_block_was_read_first(tmp_path):
    """Bounded on both sides, so cache alignment cannot change an answer.

    The block cache exists because a query per training row is untenable: 150,000 rows
    times three reads. The first version filtered only on "not after bar_time", which made
    the same bar produce a different flow z-score depending on how far back the cached
    block happened to reach -- 47 buckets at the start of a block and 13 one bar later.
    Training and serving would then disagree about the same bar for a reason neither side
    could see, which is the exact failure this module was written to end.
    """
    symbol = "BTCUSDT"
    bars = _bars(count=200)
    store = Store(tmp_path)
    try:
        for row in bars:
            store.upsert_candles(symbol, "5m", [row])
        _seed_funding(store, symbol, bars)
        _seed_flow(store, symbol, bars)
        target = 150
        walked = FeatureSource(store)
        during = None
        for index in range(49, 190):
            row, _ = walked.snapshot(bars[index - 49:index + 1], symbol,
                                     int(bars[index]["close_time"]),
                                     float(bars[index]["close"]))
            if index == target:
                during = dict(row)
        assert during is not None
        # A second source with an empty cache, reading only the one bar.
        fresh = FeatureSource(store)
        after, degraded = fresh.snapshot(bars[target - 49:target + 1], symbol,
                                         int(bars[target]["close_time"]),
                                         float(bars[target]["close"]))
    finally:
        store.close()
    assert degraded == (), degraded
    for name in FEATURES:
        assert abs(float(after[name]) - float(during[name])) < 1e-12, name


def test_the_open_interest_quadrant_has_a_price_leg():
    """The pair the module calls "the whole content of the signal" was always 0.0.

    positioning_features read the price direction from a price_change column on the
    derivatives_detail row. The positioning endpoints return open interest and account
    ratios and no price at all, and the schema has no such column, so the quadrant was
    0.0 on every row the system ever built while the comment above it described a signal.
    The caller that has the candles passes the change instead.
    """
    from app.features.order_flow import positioning_features

    rows = [{"event_time": index, "open_interest": 1000.0 + index * 10} for index in range(50)]
    rising = positioning_features(rows)
    assert rising["oi_change_5m"] > 0
    # No price leg on the row, and none supplied: the quadrant is unmeasured, not neutral.
    assert rising["oi_price_quadrant"] == 0.0
    assert positioning_features(rows, price_change=0.01)["oi_price_quadrant"] == 1.0
    assert positioning_features(rows, price_change=-0.01)["oi_price_quadrant"] == -1.0
    falling = [{"event_time": index, "open_interest": 1000.0 - index * 10}
               for index in range(50)]
    assert positioning_features(falling, price_change=0.01)["oi_price_quadrant"] == -1.0
    assert positioning_features(falling, price_change=-0.01)["oi_price_quadrant"] == 1.0


def test_a_lower_bound_alone_is_not_a_window(tmp_path):
    """ORDER BY ... DESC LIMIT n with only a floor returns the newest rows overall.

    The first version of the block reads passed since=bar_time - span and no ceiling, then
    filtered to "not after bar_time". On a table shorter than the limit that works, which
    is why a 200-bar fixture passed. On a real one the newest n rows are the whole future
    of an early bar, the filter leaves nothing, and the family reports itself unmeasured --
    on every row, for every symbol, silently. Both bounds or neither.
    """
    symbol = "BTCUSDT"
    bars = _bars(count=1500)
    store = Store(tmp_path)
    try:
        for row in bars:
            store.upsert_candles(symbol, "5m", [row])
        _seed_funding(store, symbol, bars)
        _seed_flow(store, symbol, bars)
        out = tmp_path / "dataset.jsonl"
        report = build_dataset(data_dir=tmp_path, symbols=(symbol,), interval="5m",
                               horizon=12, window=50, output=out, strict=True)
    finally:
        store.close()
    entry = report["symbols"][0]
    # 1500 bars is well past the 240-row read limit, so an unbounded upper end degrades
    # everything before the last few hundred bars.
    assert entry["degraded_features"] == {}, entry["degraded_features"]
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) > 1000
    for name in FLOW_FEATURES + POSITIONING_FEATURES:
        values = {round(row[name], 12) for row in rows}
        assert len(values) > 1, name + " never moved"


def test_a_live_source_may_not_reuse_a_block_that_predates_its_rows(tmp_path):
    """The offline block reaches forward; a live one must not.

    Extending a fetched block past its anchor is what turns a query per training row into
    a query per few hundred, and it is correct when the history is closed: those rows were
    already there and came back with the read. In the live loop they do not exist yet. A
    block anchored at 12:00 and cached for three hours holds nothing for 13:00, so the
    window for that bar is filled from whatever stale buckets did exist -- a wrong number
    rather than a missing one, which is the harder failure to see.
    """
    symbol = "BTCUSDT"
    bars = _bars(count=140)
    store = Store(tmp_path)
    try:
        for row in bars:
            store.upsert_candles(symbol, "5m", [row])
        _seed_funding(store, symbol, bars)
        # The collector has been running for a while before the walk starts, so the first
        # bar read already has a full window behind it.
        _seed_flow(store, symbol, bars[:50])
        live = FeatureSource(store, ttl_ms=0)
        walked = {}
        for index in range(50, 130):
            # The bucket for this minute lands, then the bar closes and is read.
            _seed_flow(store, symbol, bars[index:index + 1])
            row, degraded = live.snapshot(bars[index - 49:index + 1], symbol,
                                         int(bars[index]["close_time"]),
                                         float(bars[index]["close"]))
            walked[index] = (dict(row), degraded)
    finally:
        store.close()
    for index, (_row, degraded) in walked.items():
        assert degraded == (), (index, degraded)
    for index in (60, 90, 129):
        fresh = FeatureSource(store=Store(tmp_path), ttl_ms=0)
        try:
            row, degraded = fresh.snapshot(bars[index - 49:index + 1], symbol,
                                           int(bars[index]["close_time"]),
                                           float(bars[index]["close"]))
        finally:
            fresh.store.close()
        assert degraded == ()
        for name in FEATURES:
            assert abs(float(row[name]) - float(walked[index][0][name])) < 1e-12, (index, name)
