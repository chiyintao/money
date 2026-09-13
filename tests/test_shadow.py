import asyncio
from app.strategy.shadow import ShadowModels


def bars():
    return [{'close_time': i, 'close': 100+i, 'high': 101+i, 'low': 99+i, 'volume': 10, 'is_closed': True} for i in range(60)]


def test_shadow_coalesces_and_never_submits_orders(tmp_path):
    async def run():
        shadow = ShadowModels(tmp_path, max_symbols=1)
        finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        def predict(rows):
            loop.call_soon_threadsafe(finished.set)
            return {'predictions': {'test': {'side': 'LONG'}}}
        shadow._predict = predict
        assert shadow.submit('BTC', bars())
        assert not shadow.submit('BTC', bars())
        assert not shadow.submit('ETH', bars())
        assert not shadow.submit('BTC', [{**r, 'is_closed': False} for r in bars()])
        shadow.start()
        await asyncio.wait_for(finished.wait(), 2)
        await shadow.close()
        result = shadow.status()
        assert result['production'] is None
        assert result['models'][0]['mode'] == 'shadow'
        assert result['models'][0]['bar_time'] == 59
    asyncio.run(run())
