from app.market.websocket_feed import parse_market_event

def test_depth_event_parser():
    event=parse_market_event({'data':{'e':'depthUpdate','E':10,'s':'BTCUSDT','b':[['100','2'],['99','3']],'a':[['101','4']]}})
    assert event['type']=='book' and event['bids']==[[100.0,2.0],[99.0,3.0]] and event['asks']==[[101.0,4.0]]

def test_kline_event_parser():
    event=parse_market_event({'data':{'e':'kline','E':10,'k':{'s':'BTCUSDT','t':0,'T':60000,'o':'1','h':'2','l':'1','c':'1.5','v':'4','x':True}}})
    assert event['type']=='kline' and event['is_closed'] and event['close']==1.5
