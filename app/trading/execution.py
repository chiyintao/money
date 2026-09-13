"""Order settlement and decision serialization.

Extracted from run() where settle_order mixed broker mechanics, account mutation,
event recording and closure state. It now takes the runtime container explicitly so
the dependencies are visible in the signature.

Settlement also owns the rules that keep a resting entry honest. A plan is computed
from one closed bar, so it describes a price that existed for a moment. The order
carrying it used to rest forever: it filled hours later at whatever the market happened
to be while still carrying the stop and target from the original bar. Once the market
had drifted past that target the position was closed on the very next mark and booked
a fee-only loss. Three rules replace that behaviour:

* every order has a deadline and is cancelled once it passes;
* an entry is abandoned once the market has moved most of a stop distance against it;
* a fill re-anchors the stop and target by the slippage between plan and fill, so the
  realized risk is the risk the sizing model assumed.
"""
import time

from . import brackets; from ..core import order_state
from ..core.domain import Event, OrderIntent, finite as _finite
from ..core.price_state import execution_ready

# One definition, in order_state. These were separate tuples that happened to agree; adding
# a status meant finding every copy, and EXPIRED was the status that proved they could not
# stay in step -- a deadline expiry was written as CANCELED here as well as in the broker.
FINAL_STATUSES = order_state.TERMINAL_STATUSES
OPEN_STATUSES = order_state.OPEN_STATUSES


def _session_id(runtime):
    return getattr(getattr(runtime, 'session', None), 'session_id', '')


def _setting(runtime, name, default):
    return getattr(getattr(runtime, 'settings', None), name, default)


def decision_payload(signal, candidate, feed):
    """Split a model signal into the audit decision and the market context."""
    decision = {
        'side': signal['side'],
        # What the primary model proposed, before the meta-labeling gate acted. 'side' is
        # the post-gate answer, so it is FLAT for every blocked symbol -- and a blocked
        # symbol is precisely the one whose evidence has to keep accumulating. Without
        # this the audit trail could not distinguish "the model saw nothing" from "the
        # model saw something and the gate refused", and seed_from_history rebuilt its
        # evidence from 'side' alone: 1238 stored FLAT decisions produced exactly one
        # symbol with samples, so every new symbol started blind and was refused for the
        # 165 minutes it would have taken to earn 30 samples from live bars.
        'proposed_side': signal.get('proposed_side') or signal['side'],
        'confidence': signal['confidence'],
        'candidate': bool(candidate),
        'reason_codes': signal['reason_codes'],
        'source': signal.get('source'),
        'model_mode': signal.get('model_mode'),
        'model_version': signal.get('model_version'),
        'expected_return': signal.get('expected_return'),
        'expected_net_return': signal.get('expected_net_return'),
        'agreement': signal.get('agreement'),
        'edge_bps': signal.get('edge_bps'),
        'votes': signal.get('votes') or {},
        'stop': None if not candidate else candidate.get('stop'),
        'take_profit': None if not candidate else candidate.get('take_profit'),
    }
    market = {
        'price': float(signal['features'].get('price') or 0),
        'feed': feed,
        # The bar the decision was made on. Without it the audit row cannot be anchored to
        # a bar, and seed_from_history -- which rebuilds the gate's evidence from these
        # rows -- had to substitute the event's wall-clock time. That is the time the row
        # was *written*, not the bar it was about, and it lands mid-bar rather than on a
        # boundary, so replayed forecasts were scored against the wrong reference. Stored
        # decisions showed bar_time=None for exactly this reason.
        'bar_time': signal.get('bar_time'),
    }
    return decision, market


def reanchor_plan(plan, fill_price):
    """Translate stop and target by the slippage between the plan and the fill.

    The plan priced the trade at its entry while the fill happened at fill_price.
    Moving both levels by the same delta keeps each the same distance from the entry
    that actually traded, so the position risks what the risk engine sized it for
    instead of inheriting a target that is already behind the fill.
    """
    entry = _finite(plan.get('entry'))
    price = _finite(fill_price)
    if entry is None or price is None or entry <= 0 or price <= 0 or price == entry:
        return plan
    delta = price - entry
    adjusted = dict(plan)
    moved = False
    for key in ('stop', 'take_profit'):
        level = _finite(plan.get(key))
        if level is not None and level > 0:
            adjusted[key] = level + delta
            moved = True
    if not moved:
        return plan
    adjusted['planned_entry'] = entry
    adjusted['entry_deviation_bps'] = round(delta / entry * 10000, 4)
    # Translating by a large gap can push a level through the entry, which is how a
    # target ends up on the losing side and fires on the next tick. Only a bracket that
    # was sound and became unsound is reverted: a plan that was already unusable is left
    # alone for submit_entry to reject with a stated reason, so the fault is reported
    # instead of being masked by silently restoring bad levels.
    if brackets.is_valid(plan) and not brackets.is_valid(adjusted):
        adjusted.pop('planned_entry', None)
        adjusted.pop('entry_deviation_bps', None)
        return plan
    return adjusted


def reference_price(quote, side):
    """The price this side would actually pay or receive against the current book."""
    keys = ('ask', 'mark_price', 'price') if side in ('BUY', 'LONG') else ('bid', 'mark_price', 'price')
    for key in keys:
        value = _finite(quote.get(key))
        if value is not None and value > 0:
            return value
    return None


def plan_brackets(plan):
    """Planned entry plus the stop and target distances, as positive numbers."""
    entry = _finite(plan.get('entry'))
    if entry is None or entry <= 0:
        return None
    stop_distance = _finite(plan.get('stop_distance'))
    if stop_distance is None or stop_distance <= 0:
        stop = _finite(plan.get('stop'))
        stop_distance = abs(entry - stop) if stop is not None else 0.0
    target_distance = _finite(plan.get('target_distance'))
    if target_distance is None or target_distance <= 0:
        target = _finite(plan.get('take_profit'))
        target_distance = abs(target - entry) if target is not None else 0.0
    return entry, stop_distance, target_distance


def entry_drift_fraction(plan, reference):
    """Signed travel from the planned entry: positive is adverse, negative is in favour.

    Each direction is measured against its own bracket, so the number reads as "how far
    along the stop, or along the target, the market already is". Scaling by the bracket
    rather than by price keeps one threshold meaningful across symbols.
    """
    brackets = plan_brackets(plan)
    if brackets is None or reference is None:
        return 0.0
    entry, stop_distance, target_distance = brackets
    moved = (entry - reference) if plan.get('side') == 'LONG' else (reference - entry)
    if moved >= 0:
        return moved / stop_distance if stop_distance > 0 else 0.0
    return moved / target_distance if target_distance > 0 else 0.0


# ------------------------------------------------------- working-order exposure
# A resting entry is a commitment even though no position exists yet. Nothing accounted
# for it: risk.approve and limits.approve ran once, at submit time, against the book as it
# stood then, and the fill path checked only free margin. Several symbols could therefore
# each be sized against an empty book, rest, and fill one after another, taking the
# portfolio to a multiple of max_portfolio_risk and of max_gross_leverage with nothing in
# the audit trail showing a rule was broken. A reservation makes the commitment visible
# while the order is working and releases it the moment the order stops working.


def reserve_entry(runtime, order, plan):
    """Record the notional and risk a working entry has committed."""
    reservations = getattr(runtime, 'reservations', None)
    if reservations is None:
        reservations = {}
        runtime.reservations = reservations
    entry = _finite(plan.get('entry')) or 0.0
    stop = _finite(plan.get('stop'))
    quantity = _finite(order.quantity) or 0.0
    risk_cash = abs(entry - stop) * quantity if stop is not None else 0.0
    reservations[order.order_id] = {
        'symbol': order.symbol, 'side': order.side, 'quantity': quantity,
        'notional': entry * quantity, 'risk_cash': risk_cash, 'at': int(time.time() * 1000),
    }
    return reservations[order.order_id]


def release_entry(runtime, order_id):
    reservations = getattr(runtime, 'reservations', None)
    if reservations:
        reservations.pop(order_id, None)


def committed_exposure(runtime, exclude_order_id=None):
    """Notional and risk already committed by working orders, excluding one order."""
    reservations = getattr(runtime, 'reservations', None) or {}
    notional_by_symbol = {}
    risk_total = 0.0
    for order_id, item in reservations.items():
        if order_id == exclude_order_id:
            continue
        symbol = item.get('symbol')
        notional_by_symbol[symbol] = notional_by_symbol.get(symbol, 0.0) + float(item.get('notional') or 0.0)
        risk_total += float(item.get('risk_cash') or 0.0)
    return notional_by_symbol, risk_total


def fill_is_affordable(runtime, fill, plan):
    """Whether an incoming fill still fits the portfolio budget that was checked at submit.

    Returns (bool, reason). The check is against the book plus every *other* working
    order, because this order's own reservation is about to become a position.
    """
    account = getattr(runtime, 'account', None)
    if account is None:
        return True, 'no_account'
    equity = float(account.equity or 0.0)
    if equity <= 0:
        return False, 'no_equity'
    order_id = plan.get('order_id')
    reserved_notional, reserved_risk = committed_exposure(runtime, exclude_order_id=order_id)
    positions = dict(account.positions)
    open_notional = {symbol: float(position.entry) * float(position.qty)
                     for symbol, position in positions.items()}
    fill_risk = 0.0
    stop = _finite(plan.get('stop'))
    price = _finite(fill.price) or 0.0
    if stop is not None:
        fill_risk = abs(price - stop) * float(fill.quantity)
    # Replace this symbol's reservation with the fill, rather than adding to it: the
    # reservation is the same commitment in a different state.
    own_symbol = fill.symbol
    combined_symbol = (open_notional.get(own_symbol, 0.0)
                       + reserved_notional.pop(own_symbol, 0.0)
                       + price * float(fill.quantity))
    limits = getattr(runtime, 'limits', None)
    if limits is not None:
        projected = dict(open_notional)
        projected.pop(own_symbol, None)
        for symbol, notional in reserved_notional.items():
            projected[symbol] = projected.get(symbol, 0.0) + notional
        # A working order for a symbol that already has a position cannot become a second
        # position, so the count only grows when the symbol is new to the book.
        openings = len(projected) + (0 if own_symbol in positions else 1)
        max_positions = getattr(limits, 'max_positions', None)
        if max_positions is not None and openings > int(max_positions):
            return False, 'max_positions'
        # A stub limits object without the caps leaves them unenforced rather than raising:
        # the caps that exist are the ones worth checking.
        symbol_cap = getattr(limits, 'max_symbol_leverage', None)
        if symbol_cap is not None and equity > 0 and combined_symbol > equity * float(symbol_cap):
            return False, 'max_symbol_notional'
        total_cap = getattr(limits, 'max_gross_leverage', None)
        gross = combined_symbol + sum(projected.values())
        if total_cap is not None and equity > 0 and gross > equity * float(total_cap):
            return False, 'max_total_notional'
    risk = getattr(runtime, 'risk', None)
    budget_fn = getattr(risk, 'risk_budget', None)
    if callable(budget_fn):
        # The portfolio risk budget is what stops a basket of separately-sized entries
        # from risking several times the intended amount at once.
        budget = float(budget_fn(equity) or 0.0)
        if budget > 0 and fill_risk + reserved_risk > budget + 1e-9:
            return False, 'portfolio_risk_budget_exceeded'
    return True, 'ok'


def cancel_entry(runtime, order, reason):
    """Cancel a working entry and record why, so a no-trade stays explainable."""
    updated, cancel_reason = runtime.broker.cancel(order.order_id)
    runtime.pending_plans.pop(order.order_id, None)
    release_entry(runtime, order.order_id)
    runtime.store.record_event(Event('order_canceled', {
        'session_id': _session_id(runtime), 'order_id': order.order_id, 'symbol': order.symbol,
        'reason': reason, 'cancel_result': cancel_reason}).json())
    return updated


def expire_entries(runtime):
    """Sweep entries whose deadline passed without a quote arriving to settle them.

    settle_order cancels an expired order on the next book update; this catches the
    symbols that stopped publishing entirely and would otherwise sit in the book
    looking like live demand.
    """
    expired = runtime.broker.expire_orders()
    for order in expired:
        runtime.pending_plans.pop(order.order_id, None)
        release_entry(runtime, order.order_id)
        runtime.store.record_event(Event('order_canceled', {
            'session_id': _session_id(runtime), 'order_id': order.order_id,
            'symbol': order.symbol, 'reason': 'entry_expired', 'sweep': True}).json())
    return expired


def _record_fills(runtime, fills, status):
    for fill in fills:
        if fill is None:
            continue
        payload = {'session_id': _session_id(runtime), 'order_id': fill.order_id,
                   'symbol': fill.symbol, 'quantity': fill.quantity, 'price': fill.price,
                   'fee': fill.fee, 'event_time': fill.event_time, 'status': status}
        runtime.store.record_event(Event('order_filled', payload).json())


def _record_account_rejection(runtime, order, reason):
    runtime.store.record_event(Event('account_rejected_fill', {
        'session_id': _session_id(runtime), 'order_id': order.order_id,
        'symbol': order.symbol, 'reason': reason}).json())


def accept_open_fill(runtime, fill, plan):
    """Open a position and record when it opened.

    The exit policy's time stop counts bars held, so the open time has to be stamped
    where the fill is accepted rather than inferred later from trade history.
    """
    allowed, reason = fill_is_affordable(runtime, fill, plan)
    if not allowed:
        # Recorded rather than silently dropped: a refused fill is a rule doing its job,
        # and without the record it is indistinguishable from a fill that never happened.
        runtime.store.record_event(Event('fill_refused_by_risk', {
            'session_id': _session_id(runtime), 'order_id': plan.get('order_id'),
            'symbol': fill.symbol, 'reason': reason,
            'quantity': fill.quantity, 'price': fill.price}).json())
        return False
    accepted = runtime.account.open_fill(fill, reanchor_plan(plan, fill.price))
    state = getattr(runtime, 'decision_state', None)
    if accepted and state is not None:
        state.position_opened_at[fill.symbol] = int(fill.event_time or 0) or int(time.time() * 1000)
    return accepted



def _fill_from_depth(runtime, order, order_id, plan, bids, asks, volatility_bps=None):
    intent = OrderIntent(order.symbol, order.side, order.quantity - order.filled_quantity,
                         order_type='MARKET', order_id=order_id)
    fill, depth_reason = runtime.broker.execute_depth(intent, bids=bids, asks=asks,
                                                      volatility_bps=volatility_bps)
    if not fill:
        return [], depth_reason
    if not accept_open_fill(runtime, fill, plan):
        return [], 'account_rejected_fill'
    filled = order.filled_quantity + fill.quantity
    status = order_state.status_for_fill(order.quantity, filled)
    # Through the broker, like every other status change. This path built the updated order
    # itself and called track_order directly, which was a second implementation of the
    # lifecycle that the transition table could not see -- the depth fill and the quote fill
    # are the same event and only one of them was checked.
    updated, reason = runtime.broker.advance(order, status, fill.event_time, filled_quantity=filled)
    if updated is None:
        return [], reason
    return [fill], status


def settle_order(runtime, order_id, volatility_bps=None):
    """Try to fill a resting order against the freshest quote.

    Terminal parameters (market_price/bid/ask) are intentionally ignored: the live
    quote is always the authoritative source, and the previous echo-through of
    caller arguments was dead work.
    """
    plan = runtime.pending_plans.get(order_id)
    if plan is None:
        return
    order = runtime.broker.orders.get(order_id)
    if order is None or order.status not in OPEN_STATUSES:
        runtime.pending_plans.pop(order_id, None)
        return
    quote = runtime.cache.latest.get(order.symbol, {})
    now_ms = int(time.time() * 1000)
    if order.expires_at and now_ms >= order.expires_at:
        cancel_entry(runtime, order, 'entry_expired')
        return
    ready, _ = execution_ready(quote, now_ms)
    if not ready:
        return
    adverse_limit = _finite(_setting(runtime, 'max_entry_adverse_fraction', 0.0)) or 0.0
    chase_limit = _finite(_setting(runtime, 'max_entry_chase_fraction', 0.0)) or 0.0
    drift = entry_drift_fraction(plan, reference_price(quote, order.side))
    if adverse_limit > 0 and drift > adverse_limit:
        cancel_entry(runtime, order, 'entry_price_moved_against_plan')
        return
    if chase_limit > 0 and drift < -chase_limit:
        # The move this plan was built for has largely happened. Entering now would be
        # chasing it, so the entry is dropped instead of filled at the far end.
        cancel_entry(runtime, order, 'entry_price_already_past_target')
        return
    available_qty = quote.get('ask_qty', 0) if order.side in ('BUY', 'LONG') else quote.get('bid_qty', 0)
    if available_qty <= 0:
        return
    bids, asks = quote.get('bids'), quote.get('asks')
    if bids and asks and order.order_type == 'MARKET':
        fills, status = _fill_from_depth(runtime, order, order_id, plan, bids, asks,
                                         volatility_bps=volatility_bps)
    else:
        market_price = (quote['bid'] + quote['ask']) / 2
        fills, status = runtime.broker.process(
            order_id, market_price, quote['bid'], quote['ask'], available_qty,
            accept_fill=lambda fill: accept_open_fill(runtime, fill, plan),
            timestamp=now_ms, volatility_bps=volatility_bps)
    if fills:
        _record_fills(runtime, fills, status)
        # The quiet period belongs to the fill, not to the synchronous submit call. A
        # delayed fill opens a real position and needs the same cooldown as an instant
        # one; setting it only on the submit path left every resting order free to fill
        # again seconds later.
        set_cooldown = getattr(runtime.session, 'set_cooldown', None)
        if set_cooldown:
            minutes = _finite(_setting(runtime, 'entry_cooldown_minutes', 5))
            set_cooldown(order.symbol, minutes if minutes and minutes > 0 else 5)
    elif status == 'account_rejected_fill':
        _record_account_rejection(runtime, order, status)
    if status in FINAL_STATUSES:
        runtime.pending_plans.pop(order_id, None)
        release_entry(runtime, order_id)
