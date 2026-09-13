import json
from app.models.fit_model import fit
from helpers import rows as build_rows

def test_fit_model_uses_chronological_splits(tmp_path, monkeypatch):
    rows=build_rows(60)
    monkeypatch.chdir(tmp_path); result=fit(rows); assert result['status']=='ok'; assert result['train']['rows']==42; assert result['validation']['rows']==9; assert result['test']['rows']==9; assert (tmp_path/'data/model_baseline.json').exists()
