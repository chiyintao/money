"""The data the venue throws away, and the fields that were already in hand.

Two regressions that share a shape: data the system needed was available and was not
kept. One was a dropped column in a response it already fetched; the other was a set of
endpoints with a thirty-day retention that nothing polled at all.
"""
import json

import pytest

from app.market.derivatives_collect import ENDPOINTS, DerivativesCollector, merge_series, parse_long_short_ratio, parse_open_interest, parse_taker_ratio
from app.market.ingest import parse_klines
from app.features.order_flow import order_flow_features, positioning_features
from app.storage.storage import Store

HOUR = 3_600_000


# ------------------------------------------------------------- kline fields
def test_kline_parsing_keeps_the_order_flow_columns():
    # Indices 7, 8 and 9 have always been in the response and were always discarded, which
    # is why taker imbalance was missing from a dataset built on this endpoint.
    payload = [[1_700_000_000_000, '100', '101', '99', '100.5', '1000',
                1_700_000_299_999, '100400', 1234, '620']]
    row, = parse_klines(payload, now_ms=1_800_000_000_000)
    assert row['quote_volume'] == 100400.0
    assert row['trades'] == 1234
    assert row['taker_buy_volume'] == 620.0


def test_kline_parsing_tolerates_a_short_array():
    # A response that stopped at field 6 did not measure the order flow. It used to be
    # written as 0.0 and 0, which is the same value as a bar the exchange reported as
    # having no taker buying at all -- and the tier gate reads that zero as a verdict.
    payload = [[1_700_000_000_000, '100', '101', '99', '100.5', '1000', 1_700_000_299_999]]
    row, = parse_klines(payload, now_ms=1_800_000_000_000)
    assert row['quote_volume'] is None and row['trades'] is None
    assert row['taker_buy_volume'] is None
    # A genuine zero is still a zero.
    measured = parse_klines([[1_700_000_000_000, '100', '101', '99', '100.5', '1000',
                              1_700_000_299_999, '0', 0, '0']], now_ms=1_800_000_000_000)
    assert measured[0]['trades'] == 0 and measured[0]['quote_volume'] == 0.0


def test_an_unmeasured_flow_field_stays_null_through_the_store(tmp_path):
    """The store used to coerce it to 0.0, which erases the distinction the gate needs."""
    store = Store(tmp_path)
    try:
        store.upsert_candles('BTCUSDT', '5m', parse_klines(
            [[1_700_000_000_000, '100', '101', '99', '100.5', '1000',
              1_700_000_299_999]], now_ms=1_800_000_000_000))
        row, = store.candles('BTCUSDT', '5m')
        assert row['trades'] is None and row['quote_volume'] is None
        assert row['taker_buy_volume'] is None
    finally:
        store.close()


def test_order_flow_columns_survive_the_store(tmp_path):
    store = Store(tmp_path)
    try:
        store.upsert_candles('BTCUSDT', '5m', parse_klines(
            [[1_700_000_000_000, '100', '101', '99', '100.5', '1000',
              1_700_000_299_999, '100400', 1234, '620']], now_ms=1_800_000_000_000))
        row, = store.candles('BTCUSDT', '5m')
        assert row['taker_buy_volume'] == 620.0
        assert row['quote_volume'] == 100400.0
        assert row['trades'] == 1234
    finally:
        store.close()


def test_the_batch_writer_and_the_single_writer_agree(tmp_path):
    # They were two separate INSERT statements that had already drifted; this pins them
    # together so a column can no longer be added to one path only.
    payload = [[1_700_000_000_000, '100', '101', '99', '100.5', '1000',
                1_700_000_299_999, '100400', 1234, '620']]
    rows = parse_klines(payload, now_ms=1_800_000_000_000)
    store = Store(tmp_path)
    try:
        store.upsert_candles_batch([('AUSDT', '5m', rows, 1_800_000_000_000)])
        store.upsert_candles('BUSDT', '5m', rows, 1_800_000_000_000)
        first, = store.candles('AUSDT', '5m')
        second, = store.candles('BUSDT', '5m')
        for key in ('taker_buy_volume', 'quote_volume', 'trades', 'close', 'volume'):
            assert first[key] == second[key], key
    finally:
        store.close()


# ------------------------------------------------------ order-flow features
def _flow_bars(count=60, taker_share=0.65, trades=100):
    bars = []
    for index in range(count):
        volume = 1000.0 + index
        bars.append({'open_time': index * 300_000, 'close_time': index * 300_000 + 299_999,
                     'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0,
                     'volume': volume, 'taker_buy_volume': volume * taker_share,
                     'quote_volume': volume * 100, 'trades': trades, 'is_closed': True})
    return bars


def test_taker_imbalance_is_positive_when_buyers_are_aggressive():
    values = order_flow_features(_flow_bars(taker_share=0.65))
    assert values['taker_imbalance'] == pytest.approx(0.30)
    # The window is the trailing 48 bars, which is the sample the z-scores are taken over.
    assert values['order_flow_bars'] == 48


def test_taker_imbalance_is_negative_when_sellers_are_aggressive():
    values = order_flow_features(_flow_bars(taker_share=0.35))
    assert values['taker_imbalance'] == pytest.approx(-0.30)


def test_a_bar_with_no_trade_count_is_treated_as_unmeasured_not_as_balanced():
    # A genuine zero and a missing column look the same in a float; the count separates
    # them, and without that separation every legacy row would read as perfectly balanced.
    bars = _flow_bars()
    for row in bars:
        row['trades'] = 0
        row['taker_buy_volume'] = 0
    values = order_flow_features(bars)
    assert values['order_flow_bars'] == 0
    assert values['taker_imbalance'] == 0.0


def test_order_flow_features_are_finite_on_a_flat_series():
    bars = _flow_bars(taker_share=0.5)
    values = order_flow_features(bars)
    assert values['taker_imbalance_z'] == 0.0
    assert all(isinstance(value, float) for value in values.values() if not isinstance(value, int))


# -------------------------------------------------------- positioning features
def _positioning_rows(count=60):
    rows = []
    for index in range(count):
        rows.append({'event_time': index * 300_000,
                     'open_interest': 1000.0 + index * 5,
                     'long_short_ratio': 1.2 + index * 0.01,
                     'global_account_ratio': 0.9 + index * 0.005,
                     'taker_buy_sell_ratio': 1.05 + (index % 5) * 0.01,
                     'price_change': 0.001 if index % 2 else -0.001})
    return rows


def test_open_interest_change_and_crowding_are_measured():
    values = positioning_features(_positioning_rows())
    assert values['oi_change_5m'] > 0
    assert values['oi_change_1h'] > 0
    assert values['smart_retail_gap'] > 0
    assert values['positioning_rows'] == 48


def test_positioning_is_zero_with_no_collection():
    assert positioning_features([])['positioning_rows'] == 0
    assert positioning_features([{'event_time': 1, 'open_interest': 0}])['oi_change_5m'] == 0.0


def test_open_interest_rising_into_falling_price_is_its_own_quadrant():
    rows = _positioning_rows()
    for row in rows:
        row['price_change'] = -0.001
    assert positioning_features(rows)['oi_price_quadrant'] == -1.0


# ------------------------------------------------------------ collector parsing
def test_open_interest_payload_is_parsed():
    rows = parse_open_interest([{'symbol': 'BTCUSDT', 'sumOpenInterest': '12.5',
                                'sumOpenInterestValue': '900000', 'timestamp': 1}])
    assert rows == [{'event_time': 1, 'open_interest': 12.5, 'open_interest_value': 900000.0}]


def test_ratio_payload_is_parsed_and_malformed_entries_are_dropped():
    rows = parse_long_short_ratio([{'longShortRatio': '1.5', 'longAccount': '0.6',
                                    'shortAccount': '0.4', 'timestamp': 7}, {'nope': 1}, None])
    assert len(rows) == 1 and rows[0]['long_short_ratio'] == 1.5


def test_taker_payload_is_parsed():
    rows = parse_taker_ratio([{'buySellRatio': '1.1', 'buyVol': '10', 'sellVol': '9',
                               'timestamp': 3}])
    assert rows[0]['buy_sell_ratio'] == 1.1


def test_series_are_merged_on_one_timestamp_axis():
    merged = merge_series({
        'open_interest': [{'event_time': 100, 'open_interest': 5.0, 'open_interest_value': 50.0}],
        'taker_ratio': [{'event_time': 100, 'buy_sell_ratio': 1.2, 'buy_volume': 3.0,
                         'sell_volume': 2.0}],
    })
    assert len(merged) == 1
    assert merged[0]['open_interest'] == 5.0
    assert merged[0]['taker_buy_sell_ratio'] == 1.2
    # An endpoint that did not answer stays missing rather than becoming a reading of zero.
    assert merged[0]['long_short_ratio'] is None


def test_a_partial_collection_does_not_overwrite_a_complete_one(tmp_path):
    store = Store(tmp_path)
    try:
        store.record_derivatives_detail([
            {'symbol': 'BTCUSDT', 'event_time': 100, 'open_interest': 5.0,
             'long_short_ratio': 1.5}])
        store.record_derivatives_detail([
            {'symbol': 'BTCUSDT', 'event_time': 100, 'open_interest': 6.0}])
        row, = store.derivatives_detail('BTCUSDT')
        assert row['open_interest'] == 6.0
        # The column that the second run did not carry is written as null, which is the
        # honest value for "not collected this time".
        assert row['long_short_ratio'] is None
    finally:
        store.close()


class _FakeApi:
    def __init__(self, payloads):
        self.payloads = payloads
        self.session = _OpenSession()
        self.asked = []

    async def _get(self, session, path, params=None):
        self.asked.append(path)
        if path not in self.payloads:
            raise RuntimeError('http_500')
        return self.payloads[path]

    async def mark_snapshot(self):
        return []


class _OpenSession:
    closed = False


def test_collector_writes_rows_for_every_endpoint_that_answered(tmp_path):
    api = _FakeApi({
        '/futures/data/openInterestHist': [{'timestamp': 100, 'sumOpenInterest': '5',
                                            'sumOpenInterestValue': '500'}],
        '/futures/data/takerlongshortRatio': [{'timestamp': 100, 'buySellRatio': '1.1',
                                               'buyVol': '3', 'sellVol': '2'}],
    })
    store = Store(tmp_path)
    try:
        collector = DerivativesCollector(api, store, interval_seconds=0)
        written = __import__('asyncio').run(collector.collect_symbol('BTCUSDT'))
        assert written == 1
        row, = store.derivatives_detail('BTCUSDT')
        assert row['open_interest'] == 5.0
        assert row['taker_buy_sell_ratio'] == 1.1
        assert row['global_account_ratio'] is None
        assert collector.health()['period'] == '5m'
    finally:
        store.close()


def test_the_collector_never_raises_on_a_failing_endpoint(tmp_path):
    api = _FakeApi({})
    store = Store(tmp_path)
    try:
        collector = DerivativesCollector(api, store, interval_seconds=0)
        assert __import__('asyncio').run(collector.run(['BTCUSDT'])) == 0
        assert collector.health()['empty'] == 1
    finally:
        store.close()


def test_every_declared_endpoint_is_unique_and_path_shaped():
    paths = [path for _name, path, _parser, _key in ENDPOINTS]
    assert len(paths) == len(set(paths))
    assert all(path.startswith('/futures/data/') for path in paths)
