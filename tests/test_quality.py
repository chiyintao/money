from app.features.quality import validate_dataset, validate_ohlcv
from helpers import row as build_row

def test_dataset_quality_rejects_duplicate():
    row=build_row(1, symbol='BTCUSDT')
    result=validate_dataset([row,row]); assert not result['ready']; assert result['error_count']==1

def test_ohlcv_quality_reports_gap_and_invalid_bar():
    rows=[{'open_time':0,'close_time':59999,'open':1,'high':2,'low':1,'close':1.5,'volume':1}, {'open_time':120000,'close_time':179999,'open':2,'high':1,'low':1.5,'close':1.8,'volume':1}]
    result=validate_ohlcv(rows,60000)
    assert not result['ready']
    assert {error['reason'] for error in result['errors']} == {'gap','invalid_ohlcv'}

def test_ohlcv_quality_accepts_contiguous_series():
    rows=[{'open_time':i*60000,'close_time':i*60000+59999,'open':1,'high':2,'low':.5,'close':1.5,'volume':1} for i in range(3)]
    assert validate_ohlcv(rows,60000)['ready']
