import asyncio, json
import aiohttp
from .feed_normalizer import FeedNormalizer
from .connector import ConnectorHealth

def parse_market_event(message):
    data=message.get('data',message); event=data.get('e')
    if event=='kline':
        k=data.get('k',{}); return {'type':'kline','symbol':k.get('s',data.get('s')),'event_time':int(data.get('E',0)),'open_time':int(k.get('t',0)),'close_time':int(k.get('T',0)),'open':float(k.get('o',0)),'high':float(k.get('h',0)),'low':float(k.get('l',0)),'close':float(k.get('c',0)),'volume':float(k.get('v',0)),'is_closed':bool(k.get('x',False))}
    if event=='markPriceUpdate':
        return {'type':'mark_price','symbol':data.get('s'),'event_time':int(data.get('E',0)),'price':float(data.get('p',0) or 0),'mark_price':float(data.get('p',0) or 0),'funding_rate':float(data.get('r',0) or 0)}
    if event in ('aggTrade','trade'):
        return {'type':'trade','symbol':data.get('s'),'event_time':int(data.get('T',data.get('E',0))),'trade_id':str(data.get('a',data.get('t',''))),'price':float(data.get('p',0) or 0),'quantity':float(data.get('q',0) or 0),'side':'SELL' if data.get('m') else 'BUY'}
    if event=='forceOrder':
        from .flow_collect import parse_force_order

        return parse_force_order(data)
    if event=='bookTicker':
        return {'type':'book','symbol':data.get('s'),'event_time':int(data.get('E',0)),'bid':float(data.get('b',0) or 0),'ask':float(data.get('a',0) or 0),'bid_qty':float(data.get('B',0) or 0),'ask_qty':float(data.get('A',0) or 0),'bids':[[float(data.get('b',0) or 0),float(data.get('B',0) or 0)]],'asks':[[float(data.get('a',0) or 0),float(data.get('A',0) or 0)]]}
    if event in ('depthUpdate','depth'):
        bids=[[float(level[0]),float(level[1])] for level in data.get('b',[]) if len(level)>=2]
        asks=[[float(level[0]),float(level[1])] for level in data.get('a',[]) if len(level)>=2]
        return {'type':'book','symbol':data.get('s'),'event_time':int(data.get('E',data.get('T',0))),'bid':bids[0][0] if bids else 0,'ask':asks[0][0] if asks else 0,'bid_qty':bids[0][1] if bids else 0,'ask_qty':asks[0][1] if asks else 0,'bids':bids,'asks':asks}
    return None

def parse_mini_ticker(message): return parse_market_event(message)

class BinanceWebSocketFeed:
    def __init__(self,ws_url,symbols,on_event,interval='1m', category='market', proxy=None):
        if category not in ('public','market'):
            raise ValueError('invalid_stream_category')
        self.category=category
        self.ws_url=ws_url.rstrip('/')
        # Some networks block the market-data host while allowing the REST host, or the
        # other way round. Here fapi.binance.com answers directly and fstream.binance.com
        # times out, so this socket needs the proxy the REST calls do not. aiohttp reads
        # proxy settings from the environment, and a Windows system proxy is not in the
        # environment, so trust_env alone left this connection direct and it never
        # completed: no klines, no book, no fills, and every order refused for
        # missing_book_time.
        self.proxy=proxy or None
        self.symbols=tuple(sorted({x.lower() for x in symbols}))
        self.interval=interval
        self.socket=None
        self.on_event=on_event
        self.running=True
        self.reconnects=0
        self.normalizer=FeedNormalizer(source='binance.'+category)
        self.health=ConnectorHealth('binance.'+category)
    @property
    def stream_url(self):
        streams=[]
        for symbol in self.symbols:
            streams.extend([f'{symbol}@bookTicker'] if self.category=='public' else [f'{symbol}@kline_{self.interval}',f'{symbol}@markPrice@1s',f'{symbol}@aggTrade'])
        # The all-market liquidation stream is one subscription rather than one per symbol,
        # and it is the only source of forced-order flow. Binance throttles it to the
        # largest liquidation per symbol per second, which is the one that matters.
        if self.category=='market':
            streams.append('!forceOrder@arr')
        return f"{self.ws_url}/{self.category}/stream?streams={'/'.join(streams)}"
    async def set_symbols(self, symbols):
        selected=tuple(sorted({x.lower() for x in symbols}))
        if selected == self.symbols:
            return False
        self.symbols=selected
        if self.socket is not None and not self.socket.closed:
            await self.socket.close()
        return True

    async def run(self):
        delay=1
        while self.running:
            try:
                if not self.symbols:
                    await asyncio.sleep(1)
                    continue
                subscribed=self.symbols
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.ws_connect(self.stream_url,heartbeat=20,autoping=True,receive_timeout=45,proxy=self.proxy) as socket:
                        self.socket=socket
                        if subscribed != self.symbols:
                            await socket.close()
                            continue
                        delay=1
                        async for message in socket:
                            if message.type==aiohttp.WSMsgType.TEXT:
                                try:
                                    event=parse_market_event(json.loads(message.data))
                                    event=self.normalizer.normalize(event)
                                except (ValueError,TypeError,KeyError):
                                    self.health.last_error = 'invalid_market_message'
                                    continue
                                if event and event['symbol'].lower() in self.symbols and self.health.observe(event.get('event_time', 0), (event['symbol'], event['type'])):
                                    try:
                                        await self.on_event(event)
                                    except Exception as exc:
                                        self.health.last_error = 'callback_error:' + repr(exc)
                                        __import__('logging').getLogger(__name__).exception('market callback failed')
                            elif message.type in (aiohttp.WSMsgType.ERROR,aiohttp.WSMsgType.CLOSED): break
            except (aiohttp.ClientError,asyncio.TimeoutError,OSError) as exc:
                self.reconnects+=1; self.health.disconnected(exc); await asyncio.sleep(delay); delay=min(delay*2,60)
    def stop(self): self.running=False
