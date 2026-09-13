import asyncio
import json
import aiohttp
from app.market.websocket_feed import BinanceWebSocketFeed


async def check(category):
    feed=BinanceWebSocketFeed('wss://fstream.binance.com',['BTCUSDT'],None,category=category)
    expected={'bookTicker'} if category=='public' else {'aggTrade','markPriceUpdate','kline'}
    found=set()
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.ws_connect(feed.stream_url,heartbeat=20,timeout=15) as socket:
                async def receive():
                    while found!=expected:
                        message=await socket.receive_json()
                        event=message.get('data',message).get('e')
                        if event in expected: found.add(event)
                await asyncio.wait_for(receive(),20)
        return {'route':category,'url':feed.stream_url,'events':sorted(found),'status':'ok'}
    except Exception as exc:
        return {'route':category,'events':sorted(found),'status':'error','error':repr(exc)}


async def main():
    for category in ('public','market'):
        print(json.dumps(await check(category)),flush=True)

if __name__=='__main__': asyncio.run(main())
