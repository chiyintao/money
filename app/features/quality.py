import math

from .feature_spec import FEATURES


def validate_dataset(rows):
    # Derived from FEATURES rather than repeated here: a hand-written copy silently
    # drifts out of step whenever the feature set changes.
    required=('timestamp','symbol','future_return')+tuple(FEATURES)
    errors=[]; seen=set(); valid=0
    for index,row in enumerate(rows):
        missing=[key for key in required if key not in row]
        if missing: errors.append({'row':index,'reason':'missing_fields','fields':missing}); continue
        timestamp = row['timestamp']
        end = row.get('label_end_time', timestamp)
        if (not isinstance(row['symbol'], str) or not row['symbol']
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)
                or not isinstance(end, (int, float)) or not math.isfinite(end) or end < timestamp):
            errors.append({'row': index, 'reason': 'invalid_label_interval'}); continue
        key=(row['symbol'],timestamp)
        if key in seen: errors.append({'row':index,'reason':'duplicate_timestamp'}); continue
        seen.add(key)
        values=[row[key] for key in required if key not in ('symbol','timestamp')]
        if any(not isinstance(value,(int,float)) or not math.isfinite(float(value)) for value in values): errors.append({'row':index,'reason':'non_finite'}); continue
        valid+=1
    return {'rows':len(rows),'valid_rows':valid,'error_count':len(errors),'errors':errors[:20],'ready':valid>=20 and not errors}


def validate_ohlcv(rows, interval_ms=None):
    required=('open_time','close_time','open','high','low','close','volume')
    errors=[]; previous=None; seen=set(); valid=0
    for index,row in enumerate(rows):
        missing=[key for key in required if key not in row]
        if missing:
            errors.append({'row':index,'reason':'missing_fields','fields':missing}); continue
        open_time=int(row['open_time'])
        close_time=int(row['close_time'])
        if close_time < open_time:
            errors.append({'row':index,'reason':'invalid_time_range'})
        if interval_ms is not None and close_time != open_time + int(interval_ms) - 1:
            errors.append({'row':index,'reason':'invalid_close_time','expected':open_time + int(interval_ms) - 1,'actual':close_time})
        if previous is not None and interval_ms is not None and open_time-previous > int(interval_ms):
            errors.append({'row':index,'reason':'gap','missing_intervals':(open_time-previous)//int(interval_ms)-1})
        if open_time in seen:
            errors.append({'row':index,'reason':'duplicate_open_time'}); continue
        seen.add(open_time)
        numeric=[row[key] for key in ('open','high','low','close','volume')]
        if any(not isinstance(value,(int,float)) or not math.isfinite(float(value)) for value in numeric):
            errors.append({'row':index,'reason':'non_finite'}); continue
        opening,high,low,closing,volume=(float(value) for value in numeric)
        if min(opening,high,low,closing) <= 0 or volume < 0 or high < max(opening,closing) or low > min(opening,closing):
            errors.append({'row':index,'reason':'invalid_ohlcv'}); continue
        if previous is not None:
            if open_time <= previous:
                errors.append({'row':index,'reason':'out_of_order'}); continue
        previous=open_time; valid+=1
    return {'rows':len(rows),'valid_rows':valid,'error_count':len(errors),'errors':errors[:50],'ready':valid == len(rows) and valid > 0}
