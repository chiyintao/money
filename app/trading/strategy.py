from . import exit_policy
from ..features.features import snapshot


def predict(symbol, rows):
    f=snapshot(rows)
    trend_strength=(f['ema20']/f['ema50']-1) if f['ema50'] else 0.0
    momentum=f['return_10']
    long_score=(1 if trend_strength>0.001 else 0)+(1 if momentum>0.002 else 0)+(1 if 42<=f['rsi']<=68 else 0)+(1 if f['volume_ratio']>=1.05 else 0)
    short_score=(1 if trend_strength<-0.001 else 0)+(1 if momentum<-0.002 else 0)+(1 if 32<=f['rsi']<=58 else 0)+(1 if f['volume_ratio']>=1.05 else 0)
    volatility_ok=.0015<=f['atr_pct']<=.04
    side='LONG' if volatility_ok and long_score>=3 and long_score>short_score else ('SHORT' if volatility_ok and short_score>=3 and short_score>long_score else 'FLAT')
    strength=max(long_score,short_score)/4 if side!='FLAT' else max(long_score,short_score)/8
    reasons=['trend','momentum','rsi','relative_volume','volatility_regime']
    return {'symbol':symbol,'side':side,'confidence':round(strength,4),'features':f,'reason_codes':reasons,'scores':{'long':long_score,'short':short_score}}


def plan(signal, fee_rate=.0004, slippage_bps=2.0, policy=None):
    if signal['side']=='FLAT': return None
    f=signal['features']; p=f['price']
    policy=policy or exit_policy.get_policy()
    levels=exit_policy.plan_levels(policy,p,f.get('atr') or 0.0,signal['side'])
    if levels is None: return None
    distance=levels['stop_distance']; expected_move=levels['target_distance']
    round_trip_cost=p*(2*fee_rate+2*slippage_bps/10000)
    if expected_move < round_trip_cost*3: return None
    return {**signal,'entry':p,'stop':levels['stop'],'take_profit':levels['take_profit'],'expected_cost':round_trip_cost,'expected_edge':expected_move-round_trip_cost,'stop_distance':distance,'target_distance':expected_move,'exit_policy':policy.name}
