"""Exit policy: how far the stop and target sit, and how they move once in a trade.

The distances were hardcoded and duplicated in two places -- strategy.plan and
LiveModels.plan both computed max(atr * 1.5, price * 0.004) for the stop and 1.8x that
for the target. There was no way to change either without editing code, and once a
position was open nothing ever touched its stop again: no trailing, no move to
breakeven, no time stop.

This module makes the whole exit geometry one adjustable object. It is deliberately
separate from risk_profiles because the two answer different questions. A risk profile
decides how much money a trade may lose; an exit policy decides where the loss is taken
and where the profit is booked. Tying them together would mean you could not run a tight
stop with a wide size budget, which is a legitimate combination.

Levels are expressed as distances from the entry rather than absolute prices, so one
policy applies to a 0.00001 BTC tick and a 2,700 ETH tick alike.
"""
from dataclasses import asdict, dataclass, replace

POLICY_ORDER = ('tight', 'scalp', 'balanced', 'trend', 'wide')


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    label: str
    # Stop distance as a multiple of ATR, floored by a fraction of price. The floor
    # matters on quiet symbols where ATR collapses and an ATR-only stop would sit inside
    # the spread.
    stop_atr_multiple: float
    stop_price_floor: float
    # Target distance as a multiple of the stop distance. This is the reward:risk ratio
    # the strategy is asked to deliver, and it is what the edge test is measured against.
    target_rr: float
    # Move the stop to entry once price has travelled this fraction of the target.
    # Zero disables it.
    breakeven_at_rr: float
    # Trail the stop this many ATR behind the best price reached, once in profit.
    # Zero disables it.
    trail_atr_multiple: float
    # Close after this many bars regardless of price. Zero disables it.
    time_stop_bars: int
    description: str

    def snapshot(self):
        return asdict(self)


# The presets trade differently rather than merely being bigger or smaller. 'scalp' books
# small moves quickly; 'trend' accepts a wider stop and lets the target run, which only
# pays if the trailing rule is what actually closes the trade.
PRESETS = {
    'tight': ExitPolicy(
        name='tight', label='极紧',
        stop_atr_multiple=1.0, stop_price_floor=.003, target_rr=1.5,
        breakeven_at_rr=.5, trail_atr_multiple=1.0, time_stop_bars=8,
        description='止损 1.0 ATR，盈亏比 1.5，走到一半即保本并 1.0 ATR 跟踪。回撤最小，但容易被噪声扫损。'),
    'scalp': ExitPolicy(
        name='scalp', label='短线',
        stop_atr_multiple=1.2, stop_price_floor=.003, target_rr=1.6,
        breakeven_at_rr=.6, trail_atr_multiple=1.2, time_stop_bars=12,
        description='快速落袋：止损 1.2 ATR，盈亏比 1.6，达 60% 目标移保本并跟踪，12 根 K 线强制离场。'),
    'balanced': ExitPolicy(
        name='balanced', label='均衡',
        stop_atr_multiple=1.5, stop_price_floor=.004, target_rr=1.8,
        breakeven_at_rr=.8, trail_atr_multiple=1.5, time_stop_bars=0,
        description='原始硬编码几何：止损 1.5 ATR（下限 0.4%），盈亏比 1.8。当前默认。'),
    'trend': ExitPolicy(
        name='trend', label='趋势',
        stop_atr_multiple=2.0, stop_price_floor=.005, target_rr=3.0,
        breakeven_at_rr=1.0, trail_atr_multiple=1.8, time_stop_bars=0,
        description='宽止损 2.0 ATR，目标 3 倍盈亏比，达 1R 后保本并以 1.8 ATR 跟踪，让趋势跑完。'),
    'wide': ExitPolicy(
        name='wide', label='宽松',
        stop_atr_multiple=2.5, stop_price_floor=.008, target_rr=4.0,
        breakeven_at_rr=1.5, trail_atr_multiple=2.2, time_stop_bars=0,
        description='止损 2.5 ATR，盈亏比 4.0，跟踪 2.2 ATR。胜率最低但单笔期望最高，需要趋势行情。'),
}

DEFAULT_POLICY = 'balanced'

LIMITS = {
    'stop_atr_multiple': (0.1, 20.0),
    'stop_price_floor': (0.0, 0.20),
    'target_rr': (0.1, 50.0),
    'breakeven_at_rr': (0.0, 10.0),
    'trail_atr_multiple': (0.0, 20.0),
    'time_stop_bars': (0, 10000),
}


def policy_names():
    return list(POLICY_ORDER)


def get_policy(name=None, fallback=DEFAULT_POLICY):
    return PRESETS.get(name or '', PRESETS[fallback])


def custom_policy(base, overrides):
    """Apply overrides to a preset and validate.

    A trailing stop that sits further out than the stop itself would never bind, and a
    breakeven trigger beyond the target would never fire, so those combinations are
    rejected rather than silently accepted as dead settings.
    """
    base = get_policy(base) if isinstance(base, str) else base
    clean = {}
    for key, value in (overrides or {}).items():
        if key not in LIMITS:
            raise ValueError('unknown_exit_field:' + str(key))
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError('invalid_exit_value:' + str(key))
        low, high = LIMITS[key]
        if key == 'time_stop_bars':
            number = int(number)
        if not low <= number <= high:
            raise ValueError('exit_field_out_of_range:%s:%s-%s' % (key, low, high))
        clean[key] = number
    if not clean:
        return base
    merged = replace(base, name='custom', label='自定义', description='会话自定义离场参数。', **clean)
    if merged.trail_atr_multiple > merged.stop_atr_multiple:
        raise ValueError('exit_trail_exceeds_stop')
    if merged.breakeven_at_rr > merged.target_rr:
        raise ValueError('exit_breakeven_beyond_target')
    return merged


def resolve(name=None, overrides=None):
    return custom_policy(get_policy(name), overrides)


def describe():
    return {'default': DEFAULT_POLICY,
            'order': policy_names(),
            'limits': LIMITS,
            'policies': [PRESETS[name].snapshot() for name in POLICY_ORDER]}


# ------------------------------------------------------------------ level geometry

def stop_distance(policy, price, atr):
    """Absolute stop distance for one instrument at one moment."""
    price = float(price or 0.0)
    atr = float(atr or 0.0)
    if price <= 0:
        return 0.0
    return max(atr * policy.stop_atr_multiple, price * policy.stop_price_floor)


def target_distance(policy, price, atr):
    return stop_distance(policy, price, atr) * policy.target_rr


def plan_levels(policy, price, atr, side):
    """Entry, stop and target for a new position."""
    distance = stop_distance(policy, price, atr)
    target = target_distance(policy, price, atr)
    if distance <= 0 or target <= 0:
        return None
    up = side == 'LONG'
    return {'stop': price - distance if up else price + distance,
            'take_profit': price + target if up else price - target,
            'stop_distance': distance, 'target_distance': target}


def manage_position(policy, position, price, atr, bars_held=0):
    """The stop and target this open position should have right now.

    Returns the new levels plus why they changed, or None when nothing moves. Only the
    stop and target are ever returned; entry, size and direction belong to the position
    and are never touched here.

    The stop only ever tightens. Widening it would raise the risk the position was sized
    for and invalidate the risk budget that allowed the trade in the first place.
    """
    entry = float(position.entry or 0.0)
    if entry <= 0 or price is None:
        return None
    current_stop = float(position.stop or 0.0)
    current_target = float(position.target or 0.0)
    up = position.side == 'LONG'
    # Risk is the distance the trade was PLANNED with, not the distance the stop happens
    # to sit at now. Reading it off the current stop made every rule below unreachable
    # the moment the first one fired: breakeven sets the stop to the entry, so on the next
    # evaluation risk was zero, this returned None, and the trailing rule never ran again.
    # A position that had been moved to breakeven was therefore frozen there for the rest
    # of its life -- which is exactly what the dashboard showed, a stop price identical to
    # the entry price on trade after trade, and never a trailing stop.
    original = float(getattr(position, 'initial_stop', 0.0) or 0.0)
    risk = abs(entry - original) if original else 0.0
    if risk <= 0:
        # A position opened before the initial levels were recorded, or one carried in a
        # snapshot from before they existed. The plan's geometry is recoverable from what
        # the snapshot does have: the target sits target_rr risk units away, so dividing
        # the target distance by that ratio gives back the unit the trade was sized on.
        # Refusing to manage such a position instead left it frozen at whatever stop it
        # happened to be carrying -- which for one already moved to breakeven was forever.
        target_distance = abs(current_target - entry) if current_target else 0.0
        if target_distance and policy.target_rr:
            risk = target_distance / float(policy.target_rr)
    if risk <= 0:
        risk = abs(entry - current_stop) if current_stop else 0.0
    if risk <= 0:
        return None

    # Progress in R multiples: how many times the original risk the trade has made.
    progress = (price - entry) if up else (entry - price)
    progress_r = progress / risk

    new_stop = current_stop
    reason = []

    if policy.time_stop_bars and bars_held >= policy.time_stop_bars:
        return {'stop': current_stop, 'target': current_target, 'exit_now': True,
                'reason': 'time_stop', 'reason_codes': ['time_stop']}

    if policy.breakeven_at_rr and progress_r >= policy.breakeven_at_rr:
        candidate = entry
        if (up and candidate > new_stop) or (not up and candidate < new_stop):
            new_stop = candidate
            reason.append('breakeven')

    if policy.trail_atr_multiple and atr and atr > 0:
        trail = atr * policy.trail_atr_multiple
        candidate = price - trail if up else price + trail
        if (up and candidate > new_stop) or (not up and candidate < new_stop):
            new_stop = candidate
            reason.append('trail')

    if not reason:
        return None
    return {'stop': new_stop, 'target': current_target, 'exit_now': False,
            'reason': '+'.join(reason), 'reason_codes': reason}
