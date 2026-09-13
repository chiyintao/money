from dataclasses import dataclass, asdict
from typing import Any
import math
import time, uuid

def event_id(): return uuid.uuid4().hex

@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any]
    event_id: str = ''
    event_time: int = 0
    def __post_init__(self):
        if not self.event_id: object.__setattr__(self,'event_id',event_id())
        # Market events carry exchange time in the payload; local time is only a fallback.
        if not self.event_time:
            source_time = self.payload.get('event_time') or self.payload.get('E') or self.payload.get('T')
            object.__setattr__(self,'event_time',int(source_time or time.time()*1000))
    def json(self): return asdict(self)

@dataclass(frozen=True)
class OrderIntent:
    symbol: str; side: str; quantity: float; order_type: str='MARKET'; reduce_only: bool=False; limit_price: float=0; stop_price: float=0; strategy_version: str='baseline-v1'; order_id: str=''

@dataclass(frozen=True)
class Fill:
    order_id: str; symbol: str; side: str; quantity: float; price: float; fee: float; liquidity: str='taker'; event_time: int=0; participation: float=0.0

@dataclass(frozen=True)
class PaperOrder:
    order_id: str; symbol: str; side: str; quantity: float; order_type: str='MARKET'; limit_price: float=0.0; filled_quantity: float=0.0; status: str='OPEN'; created_at: int=0; updated_at: int=0; reduce_only: bool=False; plan: dict=None; expires_at: int=0

def validate_order(intent: OrderIntent):
    if intent.side not in ('BUY','SELL','LONG','SHORT'): return False,'invalid_side'
    if intent.quantity <= 0: return False,'invalid_quantity'
    if intent.order_type not in ('MARKET','LIMIT'): return False,'invalid_order_type'
    if intent.order_type=='LIMIT' and intent.limit_price<=0: return False,'invalid_limit_price'
    return True,'ok'

def finite(value):
    """Return a finite float, or None when the value is not one.

    Every price, size and prediction that crosses a decision boundary goes through this.
    It lived as two private copies -- finite() in strategy and _finite() in trading --
    with identical bodies and thirty-one call sites between them, which is one copy too
    many for the rule that keeps NaN and infinity out of an order.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
