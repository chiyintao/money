from .feature_spec import clip_features

# Used only when no funding history is available. Serving passes real values; the zeros
# exist so a caller without a funding feed still gets a complete feature row.
FUNDING_DEFAULTS = {"funding_rate": 0.0, "funding_z": 0.0, "funding_carry_24h": 0.0,
                    "mark_basis": 0.0}


def ema(values, period):
    if not values or period <= 0: return []
    alpha=2/(period+1); out=[float(values[0])]
    for value in values[1:]: out.append(alpha*float(value)+(1-alpha)*out[-1])
    return out

def atr(rows, period=14):
    if not rows or period <= 0: return 0.0
    true_ranges=[]; previous=None
    for row in rows:
        high,low,close=float(row['high']),float(row['low']),float(row['close'])
        if min(high,low,close) <= 0 or high < low: return 0.0
        true_ranges.append(max(high-low, abs(high-(previous if previous is not None else close)), abs(low-(previous if previous is not None else close))))
        previous=close
    sample=true_ranges[-period:]
    return sum(sample)/len(sample)

def rsi(values, period=14):
    if len(values)<2 or period <= 0: return 50.0
    changes=[float(values[i])-float(values[i-1]) for i in range(1,len(values))][-period:]
    gains=sum(max(x,0) for x in changes); losses=sum(max(-x,0) for x in changes)
    if losses == 0: return 100.0 if gains else 50.0
    return 100-100/(1+gains/losses)

def snapshot(rows, funding=None):
    """Feature snapshot for one symbol.

    Two families of keys are returned:

    * raw levels (price, ema20, ema50, atr, volume) -- used by the rule baseline,
      the stop-distance calculation and the dashboard;
    * scale-free features (ema20_gap, ema50_gap, atr_pct, volume_ratio) -- the ones the
      model trains on.

    Only the scale-free family can be pooled across symbols: an absolute ema20 of
    78000 (BTC) and of 0.0006 (a small alt) are not comparable, which is why a model
    fed raw levels only worked for the two symbols it was trained on.
    """
    if len(rows) < 50: raise ValueError('warmup requires at least 50 closed candles')
    closes=[float(row['close']) for row in rows]
    if any(value <= 0 for value in closes): raise ValueError('invalid close price')
    volumes=[float(row.get('volume',0)) for row in rows]
    avg_volume=sum(volumes[-21:-1])/max(1,len(volumes[-21:-1]))
    price=closes[-1]
    fast=ema(closes,20)[-1]; slow=ema(closes,50)[-1]; volatility=atr(rows)
    previous=closes[-11]
    raw={'price':price,'ema20':fast,'ema50':slow,'rsi':rsi(closes),'atr':volatility,
         'return_10':price/previous-1 if previous>0 else 0.0,
         'volume':volumes[-1],
         'ema20_gap':fast/price-1 if price>0 else 0.0,
         'ema50_gap':slow/price-1 if price>0 else 0.0,
         'atr_pct':volatility/price if price>0 else 0.0,
         'volume_ratio':volumes[-1]/avg_volume if avg_volume>0 else 0.0}
    # The modelled features are bounded so a single flash-crash bar cannot push the model
    # outside everything it was trained on. Raw levels are left untouched for the rule
    # baseline, the stop distance and the dashboard.
    raw.update(funding or FUNDING_DEFAULTS)
    raw.update(clip_features(raw))
    return raw
