import io

# 1) 给 BinanceWebSocketFeed 加 proxy 参数
p = 'app/market/websocket_feed.py'
s = io.open(p, encoding='utf-8').read()
old = "    def __init__(self,ws_url,symbols,on_event,interval='1m', category='market'):"
new = ("    def __init__(self,ws_url,symbols,on_event,interval='1m', category='market', proxy=None):")
assert old in s
s = s.replace(old, new, 1)
old2 = "        self.ws_url=ws_url.rstrip('/')"
new2 = ("        self.ws_url=ws_url.rstrip('/')" + chr(10) +
        "        # Some networks block the market-data host while allowing the REST host, or the" + chr(10) +
        "        # other way round. Here fapi.binance.com answers directly and fstream.binance.com" + chr(10) +
        "        # times out, so the socket needs the proxy the REST calls do not. aiohttp reads" + chr(10) +
        "        # proxy settings from the environment, and a Windows system proxy is not in the" + chr(10) +
        "        # environment, so trust_env alone left this connection direct and it never" + chr(10) +
        "        # completed: no klines, no book, no fills, and every order refused for" + chr(10) +
        "        # missing_book_time." + chr(10) +
        "        self.proxy=proxy or None")
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = "                    async with session.ws_connect(self.stream_url,heartbeat=20,autoping=True,receive_timeout=45) as socket:"
new3 = "                    async with session.ws_connect(self.stream_url,heartbeat=20,autoping=True,receive_timeout=45,proxy=self.proxy) as socket:"
assert old3 in s
s = s.replace(old3, new3, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('websocket_feed: proxy wired')

# 2) Settings
p = 'app/core/config.py'
s = io.open(p, encoding='utf-8').read()
old4 = '    ws_url: str = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com")'
new4 = (old4 + chr(10) +
        "    # Proxy for the market-data websocket only, empty for a direct connection." + chr(10) +
        "    # It is separate from any REST proxy because the two hosts can be reachable by" + chr(10) +
        "    # different routes, and because aiohttp only reads proxies from the environment --" + chr(10) +
        "    # a Windows system proxy is not in the environment, so it has to be named here." + chr(10) +
        "    ws_proxy: str = os.getenv("BINANCE_WS_PROXY", "").strip()")
assert old4 in s
s = s.replace(old4, new4, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('config: ws_proxy added')