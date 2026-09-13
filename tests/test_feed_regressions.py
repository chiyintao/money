from app.market.feed_normalizer import FeedNormalizer
from app.market.connector import ConnectorHealth


def test_forming_and_closed_candle_are_distinct_events():
    feed = FeedNormalizer()
    event = {'type': 'kline', 'symbol': 'BTCUSDT', 'open_time': 60000,
             'event_time': 61000, 'is_closed': False, 'close': 100}
    assert feed.normalize(event)
    assert feed.normalize({**event, 'event_time': 119999, 'is_closed': True, 'close': 101})
    assert feed.normalize(event) is None


def test_health_does_not_compare_unrelated_streams():
    health = ConnectorHealth('binance')
    assert health.observe(2000, ('BTCUSDT', 'trade'))
    assert health.observe(1999, ('ETHUSDT', 'trade'))
    assert health.observe(1998, ('BTCUSDT', 'mark_price'))
    assert not health.observe(1999, ('BTCUSDT', 'trade'))
