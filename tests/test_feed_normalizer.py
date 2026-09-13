from app.market.feed_normalizer import FeedNormalizer


def test_normalizer_deduplicates_trade_across_reconnects():
    normalizer = FeedNormalizer()
    event = {'type': 'trade', 'symbol': 'BTCUSDT', 'event_time': 10, 'trade_id': '7', 'price': 100, 'quantity': 1}
    first = normalizer.normalize(event)
    assert first['sequence'] == 0 and first['source'] == 'binance.websocket'
    assert normalizer.normalize(dict(event)) is None
    assert normalizer.stats()['duplicates'] == 1


def test_normalizer_rejects_missing_exchange_time():
    normalizer = FeedNormalizer()
    assert normalizer.normalize({'type': 'trade', 'symbol': 'BTCUSDT', 'price': 100}) is None
    assert normalizer.stats()['rejected'] == 1


def test_normalizer_builds_engine_event_and_sequences_new_events():
    normalizer = FeedNormalizer()
    first = normalizer.to_market_event({'type': 'mark_price', 'symbol': 'BTCUSDT', 'event_time': 10, 'price': 100})
    second = normalizer.to_market_event({'type': 'mark_price', 'symbol': 'BTCUSDT', 'event_time': 11, 'price': 101})
    assert first.sequence == 0 and second.sequence == 1
    assert second.payload['event_key']
