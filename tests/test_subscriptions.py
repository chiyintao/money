import asyncio
from app.market.websocket_feed import BinanceWebSocketFeed


def test_configured_interval_and_unique_symbols():
    feed=BinanceWebSocketFeed('wss://example', ['BTCUSDT','BTCUSDT'], None, interval='5m')
    assert feed.stream_url.count('btcusdt@kline_5m') == 1
    assert 'kline_1m' not in feed.stream_url
    assert '/market/stream?' in feed.stream_url
    assert 'bookTicker' not in feed.stream_url
    public=BinanceWebSocketFeed('wss://fstream.binance.com',['BTCUSDT'],None,category='public')
    assert '/public/stream?' in public.stream_url
    assert 'markPrice' not in public.stream_url


def test_subscription_change_closes_old_connection():
    class Socket:
        closed=False
        async def close(self):
            self.closed=True
    async def run():
        feed=BinanceWebSocketFeed('wss://example', ['BTCUSDT'], None, category='public')
        socket=Socket()
        feed.socket=socket
        assert not await feed.set_symbols(['BTCUSDT'])
        assert not socket.closed
        assert await feed.set_symbols(['ETHUSDT'])
        assert socket.closed
        assert 'ethusdt@bookTicker' in feed.stream_url
        assert 'btcusdt' not in feed.stream_url
    asyncio.run(run())
