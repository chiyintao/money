import json
from app.models.advanced_model import fit_advanced, predict_advanced
from helpers import row

def test_advanced_model_cost_aware(tmp_path):
 rows=[row(i, future_return=(.01 if i%2 else -.005)) for i in range(100)]
 result=fit_advanced(rows,cost_bps=4,output=str(tmp_path/'model.json'),rounds=4); assert result['status']=='ok' and result['test']['cost_bps']==4
 model=json.loads((tmp_path/'model.json').read_text()); assert predict_advanced(model,rows[-1])['model']=='gbst-v3'
