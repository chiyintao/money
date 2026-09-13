from app.models.walkforward import walk_forward

def test_walk_forward_has_multiple_time_windows():
    rows=[{'timestamp':i,'symbol':'X','ema20':i+1,'ema50':i,'rsi':50,'atr':1,'return_10':.01,'volume':10,'future_return':.001*i} for i in range(70)]
    result=walk_forward(rows,40,10,10); assert result['windows']==3; assert result['reports'][0]['start']==0
