"""Hard validity checks on a trade plan's stop and target geometry.

The audit found 16 trades whose take-profit sat on the losing side of the entry, all of
them essentially at the entry price. A target that is already behind the fill is not a
target; it fires on the next tick and books the spread plus two fees. That is how one
symbol produced 37 round trips under ten seconds with fees of 0.97 against a profit of
-0.32.

Nothing checked this. The levels were computed, stored, and acted on. These helpers make
the invariant explicit and are cheap enough to run on every plan.

The rule is directional and asymmetric on purpose:

    LONG   stop < entry < target
    SHORT  target < entry < stop

The stop may legitimately end up above entry for a LONG once a trailing stop has locked
in profit, so only the target is constrained to the profit side when re-checking an open
position. At plan time, before anything has moved, both bounds must hold.
"""
# A target closer to entry than this cannot pay for the round trip, so it is treated as
# invalid rather than merely tight. Expressed in bps of entry price.
MIN_TARGET_BPS = 1.0

# How much reward a plan must carry per unit of risk.
#
# Direction and a 1bp target were the only geometric requirements, and they admit plans
# that cannot work. The trade records contain a SHORT opened at 4.98 with its stop at
# 5.1918 (425 bp away) and its target at 4.9822 (4.4 bp away): a 1:97 ratio that grosses
# 4.4 bp against an 8 bp round trip, so even a WINNING touch loses money -- and it fires
# on the first tick that moves. 36 trades closed within three seconds of opening this way,
# and 41 inside two minutes.
#
# The floor is set where a plan stops being self-evidently absurd, not at a target ratio.
# The exit policy's own geometry (target_rr = 1.6) is what expresses the intended
# reward:risk; a gate that duplicated that number would either reject every policy worth
# running or quietly become the thing that decides the geometry. What this refuses is the
# tail where the reward is a rounding error against the risk.
MIN_REWARD_RISK = 0.25
# The same floor named in basis points, for callers that report ratios as bps.
MIN_REWARD_RISK_BPS = MIN_REWARD_RISK


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float('inf'), float('-inf')):
        return None
    return number


def bracket_problem(plan_or_position, entry_key='entry', require_stop=True):
    """Return a reason string when the levels are unusable, otherwise None.

    Works on a plan dict or a live position, since both carry entry/stop/target.
    """
    if plan_or_position is None:
        return 'no_plan'
    side = plan_or_position.get('side')
    if side not in ('LONG', 'SHORT'):
        return 'invalid_side'
    entry = _finite(plan_or_position.get(entry_key))
    stop = _finite(plan_or_position.get('stop'))
    target = _finite(plan_or_position.get('take_profit') or plan_or_position.get('target'))
    if entry is None or entry <= 0:
        return 'invalid_entry'
    if target is None or target <= 0:
        return 'missing_target'
    # Distance to the target has to cover the cost of getting there and back.
    target_bps = abs(target - entry) / entry * 10000.0
    if target_bps < MIN_TARGET_BPS:
        return 'target_inside_entry'
    if side == 'LONG':
        if target <= entry:
            return 'target_behind_entry'
        if require_stop:
            if stop is None or stop <= 0:
                return 'missing_stop'
            if stop >= entry:
                return 'stop_beyond_entry'
    else:
        if target >= entry:
            return 'target_behind_entry'
        if require_stop:
            if stop is None or stop <= 0:
                return 'missing_stop'
            if stop <= entry:
                return 'stop_beyond_entry'
    # And the reward has to be worth the risk being taken. A plan whose target sits a
    # fraction of its stop away is not a trade, it is a fee donation with a directional
    # opinion attached: it needs a near-perfect hit rate to break even, and because the
    # target is close it resolves within seconds of the fill.
    #
    # Checked after the direction tests on purpose. "The target is on the losing side" is
    # the more specific and more serious fault, and it is the reason string operators and
    # tests already key on; reporting a ratio problem instead would hide it.
    if require_stop and stop is not None and stop > 0:
        stop_bps = abs(stop - entry) / entry * 10000.0
        if stop_bps > 0 and target_bps / stop_bps < MIN_REWARD_RISK:
            return 'reward_below_risk'
    return None


def is_valid(plan_or_position, entry_key='entry', require_stop=True):
    return bracket_problem(plan_or_position, entry_key, require_stop) is None


def repair_target(plan, entry_key='entry'):
    """Push a profit-side target out to the stop's distance, or return None to reject.

    Only used where a target is wrong but the direction and stop are sound, so the trade
    keeps the risk it was sized for instead of being silently resized or dropped.
    """
    entry = _finite(plan.get(entry_key))
    stop = _finite(plan.get('stop'))
    if entry is None or entry <= 0 or stop is None or stop <= 0:
        return None
    distance = abs(entry - stop)
    if distance <= 0:
        return None
    side = plan.get('side')
    fixed = dict(plan)
    # Mirror of the stop distance, which is the smallest target that preserves 1:1 and
    # therefore always clears the entry.
    fixed['take_profit'] = entry + distance if side == 'LONG' else entry - distance
    return fixed
