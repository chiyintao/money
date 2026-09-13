from app.market.websocket_feed import parse_mini_ticker

def test_mark_price_parser():
    row=parse_mini_ticker({'stream':'btcusdt@markPrice@1s','data':{'e':'markPriceUpdate','E':123,'s':'BTCUSDT','p':'100.5','r':'0.0001'}})
    assert row['symbol']=='BTCUSDT'
    assert row['mark_price']==100.5
    assert row['funding_rate']==0.0001

def test_unknown_message_is_ignored():
    assert parse_mini_ticker({'data':{'e':'unknown'}}) is None
