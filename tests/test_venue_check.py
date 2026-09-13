"""The venue check, and the configuration it was written to catch.

The failure this guards against has no symptom. A testnet stream produces prices, fills
orders, writes bars and reports an equity curve exactly as a production one does; the only
difference is that every number in the system came from a book that is not the market.
"""
import pytest

from app.market.venue_check import classify_host, host_mismatch, summarise, verdict


# The measured pair, reproduced from twenty seconds of BTCUSDT bookTicker. The spread
# varies bar to bar on both venues, so the fixtures vary it too: the reported figure is the
# median, and a constant spread fixture would not exercise that.
def _book(count, mid, spread_bps_median, bid_qty, ask_qty, jitter=0.4):
    import random

    rng = random.Random(17)
    rows = []
    for index in range(count):
        spread = spread_bps_median * (1 + rng.uniform(-jitter, jitter))
        half = mid * spread / 2 / 10000
        rows.append({'b': '%.2f' % (mid - half), 'a': '%.2f' % (mid + half),
                     'B': '%.3f' % bid_qty, 'A': '%.3f' % ask_qty})
    return rows


TESTNET_SAMPLE = _book(129, 77866.8, 1.2, 0.002, 353.8)
PRODUCTION_SAMPLE = _book(10645, 77869.65, 0.01, 2.156, 10.864)


def test_the_hosts_are_classified_by_what_they_actually_are():
    assert classify_host('wss://fstream.binance.com') == 'production'
    assert classify_host('https://fapi.binance.com') == 'production'
    assert classify_host('wss://fstream.binancefuture.com') == 'testnet'
    assert classify_host('wss://stream.binancefuture.com') == 'testnet'
    assert classify_host('wss://demo-fstream.binance.com') == 'testnet'
    assert classify_host('wss://example.invalid') == 'unknown'
    assert classify_host('') == 'unknown'


def test_mixing_a_testnet_stream_with_a_production_rest_host_is_reported():
    # This is exactly what .env configured: production REST and a testnet stream. Nothing
    # compared them, so features, bars and paper fills all came from the testnet book while
    # mark price, funding and instrument filters came from production.
    mismatch = host_mismatch('wss://fstream.binancefuture.com', 'https://fapi.binance.com')
    assert mismatch is not None
    assert mismatch['reason'] == 'mixed_venue'
    assert mismatch['ws'] == 'testnet' and mismatch['rest'] == 'production'


def test_a_matched_venue_is_not_reported():
    assert host_mismatch('wss://fstream.binance.com', 'https://fapi.binance.com') is None
    assert host_mismatch('wss://stream.binancefuture.com', 'https://testnet.binancefuture.com') is None


def test_the_testnet_book_is_identified_by_its_shape():
    # The tell that actually separated the two venues was the touch size: 0.002 BTC on the
    # bid against 353.8 BTC on the offer. A production book does not quote one side five
    # orders of magnitude larger than the other, and the depth-based fill model reads
    # exactly those two numbers to decide whether an order fills at the touch at all.
    sample = summarise(TESTNET_SAMPLE)
    assert sample['messages'] == 129
    assert sample['bid_qty'] == pytest.approx(0.002)
    assert sample['size_imbalance'] > 1000
    report = verdict(sample, rest_last=77869.6, elapsed_seconds=20.0)
    assert report['status'] == 'suspect'
    assert 'one_sided_touch_size' in report['findings']


def test_a_wide_spread_is_also_flagged():
    wide = _book(200, 77866.8, 6.0, 5.0, 6.0)
    report = verdict(summarise(wide), rest_last=77866.8, elapsed_seconds=20.0)
    assert 'spread_wider_than_a_production_book' in report['findings']
    assert report['status'] == 'suspect'


def test_the_production_book_passes():
    sample = summarise(PRODUCTION_SAMPLE)
    report = verdict(sample, rest_last=77869.6, elapsed_seconds=20.0)
    assert report['status'] == 'ok'
    assert report['findings'] == []
    assert report['spread_bps'] < 0.1
    assert abs(report['deviation_bps']) < 1.0


def test_nothing_sampled_is_unknown_not_agreement():
    # Reporting "ok" because there was nothing to compare is the bug being fixed: the
    # previous arrangement looked fine for the project's whole life for exactly that reason.
    report = verdict(None)
    assert report['status'] == 'unknown'
    assert report['findings'] == ['no_book_ticker_received']


def test_a_mid_far_from_the_rest_price_is_flagged():
    sample = summarise([{'b': '100', 'a': '100.02', 'B': '1', 'A': '1'}] * 50)
    report = verdict(sample, rest_last=105.0, elapsed_seconds=5.0)
    assert 'mid_deviates_from_rest_price' in report['findings']
    assert report['deviation_bps'] < -400


def test_summarise_ignores_unusable_messages():
    rows = [{'b': '0', 'a': '1'}, {'b': '2', 'a': '1'}, {'nope': 1}, None,
            {'b': '100', 'a': '100.01', 'B': '1', 'A': '1'}]
    sample = summarise(rows)
    assert sample['messages'] == 1


def test_the_report_carries_the_numbers_behind_the_verdict():
    report = verdict(summarise(TESTNET_SAMPLE), rest_last=77869.6, elapsed_seconds=20.0)
    for key in ('messages', 'messages_per_second', 'mid', 'rest_last', 'deviation_bps',
                'spread_bps', 'size_imbalance', 'thresholds'):
        assert key in report, key
    assert report['messages_per_second'] == pytest.approx(6.45, rel=0.01)


def test_the_default_configuration_is_a_matched_pair():
    # The shipped default was already correct; only the environment file disagreed with it,
    # which is why nothing caught it. This pins the default to the production host.
    from app.core.config import Settings

    settings = Settings()
    assert classify_host(settings.ws_url) == 'production'
    assert classify_host(settings.base_url) == 'production'
    assert host_mismatch(settings.ws_url, settings.base_url) is None
