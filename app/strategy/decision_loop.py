"""Model-driven decision loop shared by the tick and closed-bar paths.

Both entry paths used to carry their own copy of the risk -> guard -> limits ->
broker sequence, which is how they drifted apart. They now funnel through
submit_entry() so there is exactly one place where an order can be created.
"""
import asyncio
import datetime
import time
import uuid

from ..trading import brackets, exit_policy
from ..core.config import interval_to_ms
from ..core.domain import Event, OrderIntent
from ..trading.execution import decision_payload, expire_entries, reserve_entry, settle_order
from ..features.features import snapshot as features_snapshot
from ..core.price_state import execution_ready
from .session_control import close_position

MIN_ROWS = 50
# How long to wait before re-picking symbols when nothing the session holds can be traded.
# Without it a market where no symbol passes the gate would re-rank on every refresh.
RESELECT_COOLDOWN_MS = 300000


def _book_feed_live(runtime):
    """Whether the order-book transport is delivering anything at all.

    Read from the sockets' own health, not from the session's book timestamps. The
    session cannot make this distinction -- a dead symbol and a dead socket both look
    like an empty book_seen_at -- and treating the second as the first is what churned
    the whole selection every ~70 seconds through a five-hour outage.

    True when any socket is connected, and also when no socket was ever started: an
    unconfigured feed is not evidence that the market is silent, and refusing to reselect
    on that basis would turn a missing socket into a permanently stuck selection.
    """
    running = False
    for name in ('public', 'ws'):
        feed = getattr(runtime.feeds, name, None)
        if feed is None or not getattr(feed, 'running', False):
            continue
        running = True
        if getattr(getattr(feed, 'health', None), 'connected', False):
            return True
    return not running


def _edge_tier(runtime):
    """Selection tier function from the meta-labeling gate, or None when it is off."""
    tracker = getattr(getattr(runtime, 'decisions', None), 'symbol_edge', None)
    if tracker is None or not tracker.enabled:
        return None
    return tracker.tier


def _candidates(market, source, tier):
    """The ranking rows, widened to include every gate-eligible symbol.

    The ranking only lists the top movers, which is a few dozen of the several hundred
    markets on offer. A symbol the gate has proven profitable can easily sit outside that
    slice, and then its proven edge is unreachable no matter how the session ranks it.
    Eligible symbols are therefore added to the pool from the full market list.
    """
    rows = list(market.get(source, []))
    if tier is None:
        return rows
    present = {row.get('symbol') for row in rows}
    extra = [row for row in market.get('all', [])
             if row.get('symbol') and row.get('symbol') not in present
             and row.get('price', 0) > 0 and tier(row['symbol']) >= 2]
    return rows + extra


def _now_ms():
    return int(time.time() * 1000)


def passive_entry_price(quote, side, offset_bps=0.0):
    """The resting price for a passive entry, or 0 when the quote cannot support one.

    Joins the near touch by default: the bid for a buy, the ask for a sell. That is the
    most aggressive price that can still rest as a maker, so it maximises the fill rate
    among prices that do not cross. A positive offset improves on the touch, which trades
    fill probability for a better price.

    Returns 0 rather than guessing when the quote has no bid or ask. A missing side means
    the book is not readable, and inventing a price from the mid would submit an order at a
    level nothing quoted.
    """
    try:
        offset = float(offset_bps or 0.0) / 10000.0
    except (TypeError, ValueError):
        offset = 0.0
    buying = str(side).upper() in ('BUY', 'LONG')
    touch = quote.get('bid') if buying else quote.get('ask')
    try:
        touch = float(touch)
    except (TypeError, ValueError):
        return 0.0
    if touch <= 0:
        return 0.0
    price = touch * (1 - offset) if buying else touch * (1 + offset)
    return price if price > 0 else 0.0


def _stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S.%f')[:-3]


def record_signal(runtime, symbol, signal, candidate, feed, extra=None):
    """Publish the decision to the audit trail and to the dashboard feed.

    The dashboard row is refreshed on every call; the audit row is only written when
    the decision actually changed or the heartbeat interval elapsed. Writing every
    evaluation produced roughly 750k rows describing a few hundred distinct decisions.
    """
    decision, market = decision_payload(signal, candidate, feed)
    state = runtime.decision_state
    signature = (feed, decision['side'], decision['candidate'], decision.get('source'),
                 decision.get('model_version'), tuple(decision.get('reason_codes') or ()))
    now_ms = _now_ms()
    previous = state.last_signal.get(symbol) or {}
    changed = previous.get('signature') != signature
    interval = int(getattr(runtime.settings, 'strategy_audit_interval_ms', 60000) or 0)
    heartbeat = interval > 0 and now_ms - int(previous.get('at', 0)) >= interval
    runtime.session.record_decision(symbol, decision, market, audit=changed or heartbeat)
    if changed or heartbeat:
        state.last_signal[symbol] = {'signature': signature, 'at': now_ms}
    features = signal['features']
    entry = {**runtime.cache.latest.get(symbol, {}), 'symbol': symbol,
             'side': signal['side'], 'confidence': signal['confidence'],
             'rsi': features.get('rsi'), 'atr': features.get('atr'),
             'model_source': signal.get('source'), 'model_mode': signal.get('model_mode'),
             'model_version': signal.get('model_version'),
             'expected_return': signal.get('expected_return'),
             'agreement': signal.get('agreement'), 'updated': _stamp()}
    if extra:
        entry.update(extra)
    runtime.cache.latest[symbol] = entry


def observe_features(runtime, signal):
    """Record the feature snapshot a decision was made on, for the drift monitor."""
    monitor = getattr(runtime, 'drift', None)
    if monitor is None:
        return
    features = (signal or {}).get('features') or {}
    if features:
        monitor.observe(features)


def _drift_entry_check(runtime):
    """Whether the drift verdict permits a new entry. Returns (allowed, reason).

    DRIFT_POLICY=block was documented as a reason code the decision layer could refuse on,
    and the decision layer never read it: the monitor recorded a feature_drift event,
    /health reported drifted, and entries carried on. A safety policy that only writes a
    row is worse than no policy, because the operator who set it believes the refusals are
    happening. The verdict is read through blocks_entries(), so the policy is still the one
    place that decides, and a monitor that has not produced a verdict yet permits trading
    rather than stopping it on a check that never ran.
    """
    monitor = getattr(runtime, 'drift', None)
    if monitor is None:
        return True, None
    try:
        if not monitor.blocks_entries():
            return True, None
        drifted = [str(name) for name in ((monitor.last_report or {}).get('drifted') or [])]
    except Exception as exc:
        # A broken monitor must not silently stop the session either. It is reported.
        runtime.errors.note('feature_drift', exc)
        return True, None
    return False, ('feature_drift:' + ','.join(drifted)) if drifted else 'feature_drift'


def _bar_volatility_bps(runtime, symbol):
    """Per-bar volatility in basis points, for the fill model's latency term.

    0.0 when there is no measurement. The fill model reads that as "charge no latency",
    not as "volatility is zero", so a deployment without enough history fills exactly as
    it did before latency existed.
    """
    return max(0.0, float(_realized_volatility(runtime, symbol) or 0.0)) * 10000.0


def _realized_volatility(runtime, symbol):
    """Per-bar return standard deviation of the symbol's recent closed bars."""
    from ..trading.concentration import realized_volatility

    # getattr throughout: the cache is duck-typed and a substitute without the closed-bar
    # store should fall back to the database rather than raise inside the entry path.
    cached=(getattr(runtime.cache,'closed_rows',None) or {}).get(symbol) or []
    closes=[row.get('close') for row in cached if row.get('close')]
    if len(closes)<20:
        reader=getattr(getattr(runtime,'store',None),'candles',None)
        if callable(reader):
            history=reader(symbol, runtime.settings.interval, 200, closed_only=True)
            closes=[row.get('close') for row in history if row.get('close')]
    window=int(getattr(runtime.settings,'correlation_window_bars',120) or 120)
    # No history means no measurement, which the risk engine reads as "do not scale"
    # rather than as "volatility is zero".
    return realized_volatility(closes, window) if len(closes)>=20 else 0.0


def _correlations(runtime, symbols):
    """Pairwise correlation of closed-bar returns, cached per closed bar.

    Recomputed once per bar rather than once per candidate: the matrix costs a pass over
    every symbol's history, and the answer cannot change until a new bar closes.
    """
    from ..trading.concentration import correlation_matrix

    wanted=sorted(set(symbols))
    if len(wanted)<2:
        return {}
    key=(int(getattr(runtime.cache,'closed_bar_time',0) or 0), tuple(wanted))
    cached=getattr(runtime,'_correlation_cache',None)
    if cached and cached[0]==key:
        return cached[1]
    window=int(getattr(runtime.settings,'correlation_window_bars',120) or 120)
    series={}
    for item in wanted:
        rows=(getattr(runtime.cache,'closed_rows',None) or {}).get(item) or []
        closes=[row.get('close') for row in rows if row.get('close')]
        if len(closes)>=11:
            series[item]=closes
    matrix=correlation_matrix(series, window) if len(series)>1 else {}
    runtime._correlation_cache=(key, matrix)
    return matrix


def submit_entry(runtime, symbol, signal, candidate, feed, now_ms=None, expected_session=None):
    """Run one candidate through every gate and submit at most one order.

    Returns the order when it was created, otherwise None. The rejection reason is
    always recorded as a risk_decision event so a silent no-trade can be explained.
    """
    # The callers reach here after awaiting model inference, which yields the event loop.
    # An HTTP POST /api/simulation/start can complete inside that window and replace the
    # account, the session id and the initial equity. The order would then be sized from
    # the old book but booked against the new one, and because every audit row reads the
    # session id at write time the result would look entirely legitimate. The caller
    # therefore passes the session it made the decision for.
    if expected_session is not None and expected_session != runtime.session.session_id:
        runtime.errors.note('session_race', 'dropped %s decision for %s: session changed'
                            % (feed, symbol))
        return None
    if not candidate or not runtime.session.can_enter(symbol):
        return None
    now_ms = now_ms or _now_ms()
    # Geometry is validated before anything else touches it. A target on the wrong side of
    # the entry fires on the next tick and books the spread plus two fees, which is how a
    # single symbol once produced 37 round trips under ten seconds at a net loss.
    geometry = brackets.bracket_problem(candidate)
    if geometry is not None:
        runtime.store.record_event(Event('order_rejected', {
            'session_id': runtime.session.session_id, 'symbol': symbol,
            'reason': 'bracket_invalid:' + geometry, 'feed': feed,
            'entry': candidate.get('entry'), 'stop': candidate.get('stop'),
            'take_profit': candidate.get('take_profit')}).json())
        return None
    account = runtime.account
    risk = runtime.risk
    # One position and one working order per symbol. Without this check every bar added
    # another order for a symbol that already had one: only the first could ever be
    # filled, and the rest queued behind it to fill minutes or hours later at prices
    # their plan never described.
    if symbol in account.positions or runtime.broker.has_open_order(symbol):
        return None
    notional_by_symbol = {s: p.entry * p.qty for s, p in account.positions.items()}
    # Risk the open book has already committed, which is what bounds the next entry's
    # share of the portfolio budget.
    open_risk = sum(abs(p.entry - p.stop) * p.qty for p in account.positions.values() if p.stop)
    # The risk engine sizes the order and reports which cap bound it. Its notional is
    # also what the portfolio caps are checked against, instead of a second estimate
    # derived independently here that could disagree with the size actually submitted.
    # The model's own edge, carried into the size. Before this the candidate's `edge_bps`
    # and `probability_up` were computed, transported through the whole decision path, and
    # then read by nothing: a 0.5bp edge and a 50bp edge produced identical quantities.
    # Sizing is capped by max_risk either way, so this can only reduce exposure to weak
    # signals -- it never enlarges a position beyond what the caps already allowed.
    approved = risk.approve(candidate, account.equity, sum(notional_by_symbol.values()), now_ms,
                            open_risk=open_risk, symbol_notional=notional_by_symbol.get(symbol, 0.0),
                            realized_volatility=_realized_volatility(runtime, symbol),
                            edge_bps=candidate.get('edge_bps'),
                            probability_up=candidate.get('probability_up'))
    fresh, fresh_reason = runtime.guard.approve_new_entry(symbol)
    drift_ok, drift_reason = _drift_entry_check(runtime)
    # The book's correlations are only worth computing when the concentration cap is on.
    correlations=None
    if float(getattr(runtime.limits,'max_correlated_leverage',0.0) or 0.0)>0:
        correlations=_correlations(runtime, list(notional_by_symbol)+[symbol])
    # The correlation argument is only passed when the cap is actually on, so the call
    # keeps its original four-argument shape in every deployment that has not opted in.
    if correlations is None:
        portfolio_ok, portfolio_reason = runtime.limits.approve(
            symbol, approved.get('notional', 0.0), notional_by_symbol, account.equity)
    else:
        portfolio_ok, portfolio_reason = runtime.limits.approve(
            symbol, approved.get('notional', 0.0), notional_by_symbol, account.equity,
            correlations=correlations)
    executable, execution_reason = execution_ready(runtime.cache.latest.get(symbol, {}), now_ms)
    if not executable:
        fresh, fresh_reason = False, execution_reason
    # Size the order to the symbol's lot step before submitting. The risk engine sizes
    # by risk, not by lot step, so an unrounded quantity leaves an untradable residue:
    # the fill is rounded down, and the remainder rounds to zero and can never trade,
    # leaving the order stuck in PARTIALLY_FILLED forever.
    spec = runtime.broker.spec(symbol)
    quantity = spec.round_qty(approved.get('quantity', 0) or 0)
    tradable = quantity > 0 and quantity * candidate['entry'] >= spec.min_notional
    final_approved = (executable and fresh and drift_ok and portfolio_ok
                      and approved.get('approved', False) and tradable)
    # Ordered explicitly. The previous 'or' chain read approved.get('reason') first, and
    # for an approved plan that key is absent while portfolio_reason is the constant
    # 'portfolio_ok' -- truthy -- so the final fallback was unreachable and an order
    # rejected for being under the lot/notional step was logged as 'portfolio_ok'.
    if final_approved:
        reason = None
    elif not approved.get('approved', False):
        reason = approved.get('reason') or 'risk_rejected'
    elif not fresh:
        reason = fresh_reason
    elif not drift_ok:
        reason = drift_reason
    elif not portfolio_ok:
        reason = portfolio_reason
    elif not tradable:
        reason = 'below_min_notional'
    else:
        reason = 'not_executable'
    runtime.store.record_event(Event('risk_decision', {
        'session_id': runtime.session.session_id, 'symbol': symbol, 'approved': final_approved,
        'risk_cash': approved.get('risk_cash', 0), 'quantity': quantity,
        'notional': approved.get('notional', 0), 'binding': approved.get('binding'),
        'reason_codes': [] if final_approved else [reason],
        'feed': feed, 'strategy_decision_at': int(runtime.session.step),
    }).json())
    # Which cap decided the size rides along on the live symbol row, so an unexpectedly
    # small order can be explained from the dashboard instead of by reading the engine.
    latest = runtime.cache.latest.get(symbol)
    if latest is not None:
        latest['size_binding'] = approved.get('binding')
        latest['planned_notional'] = round(float(approved.get('notional', 0) or 0), 8)
        latest['planned_risk_cash'] = round(float(approved.get('risk_cash', 0) or 0), 8)
    order_id = uuid.uuid4().hex
    side = 'BUY' if candidate['side'] == 'LONG' else 'SELL'
    # Rest the entry on the book instead of crossing it when configured to. The broker
    # already prices a resting limit at the maker rate and refuses to fill one the market
    # has not traded to, so the only missing piece was that nothing ever submitted one.
    intent = OrderIntent(symbol, side, quantity, order_id=order_id)
    if str(getattr(runtime.settings, 'entry_order_type', 'market') or 'market').lower() == 'limit':
        passive = passive_entry_price(runtime.cache.latest.get(symbol, {}), side,
                                      getattr(runtime.settings, 'entry_passive_offset_bps', 0.0))
        if passive:
            intent = OrderIntent(symbol, side, quantity, order_type='LIMIT',
                                 limit_price=passive, order_id=order_id)
            plan_price = float(candidate.get('price') or 0) or passive
            candidate = {**candidate, 'price': plan_price}
    if not final_approved:
        runtime.store.record_event(Event('order_rejected', {
            'session_id': runtime.session.session_id, 'order_id': intent.order_id,
            'symbol': symbol, 'reason': reason, 'feed': feed}).json())
        return None
    # The order carries its own deadline. A plan priced one bar, not the rest of the
    # day, so the entry is only good for a bounded window after the decision.
    plan = dict(candidate)
    ttl = int(getattr(runtime.settings, 'order_ttl_ms', 0) or 0)
    if ttl > 0:
        plan['expires_at'] = now_ms + ttl
    order, fill_reason = runtime.broker.submit(intent, now_ms, plan)
    if order is None:
        runtime.store.record_event(Event('order_rejected', {
            'session_id': runtime.session.session_id, 'order_id': intent.order_id,
            'symbol': symbol, 'reason': fill_reason, 'feed': feed}).json())
        return None
    runtime.pending_plans[order.order_id] = plan
    # Counted here rather than in approve(): approve() also runs for plans the portfolio
    # caps or the lot step then refuse, and a budget spent on orders that never existed
    # would tighten itself the more the rest of the system says no.
    risk.record_entry()
    # The order now holds portfolio budget until it fills or stops working, so the next
    # submit_entry sees it even though no position exists yet.
    reserve_entry(runtime, order, plan)
    settle_order(runtime, order.order_id, volatility_bps=_bar_volatility_bps(runtime, order.symbol))
    runtime.store.set_runtime('risk_state', risk.snapshot())
    persist_symbol_edge(runtime)
    return order


def persist_symbol_edge(runtime):
    """Write the edge tracker's evidence back to the runtime store.

    Rescored on every entry, so a restart resumes from measured history rather than
    relearning the gate from scratch.
    """
    tracker = getattr(getattr(runtime, 'decisions', None), 'symbol_edge', None)
    if tracker is None:
        return
    try:
        runtime.store.set_runtime('symbol_edge', tracker.snapshot())
    except (TypeError, ValueError):
        # A tracker that cannot be serialized must not take the entry path down with it.
        pass


def flush_account_events(runtime):
    """Persist equity and stream newly closed trades/liquidations to the audit log."""
    account = runtime.account
    state = runtime.decision_state
    before_trades, before_liq = len(account.trades), len(account.liquidations)
    runtime.store.record_equity(account.cash, account.equity, account.unrealized_pnl(), len(account.positions))
    for trade in account.trades[before_trades:]:
        trade.setdefault('session_id', runtime.session.session_id)
        # A closed trade is the only place a losing streak can be observed, and the streak
        # is what scales the risk budget down while a model is misbehaving.
        runtime.risk.record_close(trade.get('pnl'))
        runtime.risk.observe_equity(account.equity)
        runtime.store.record_event(Event('fill', trade).json())
    for liquidation in account.liquidations[before_liq:]:
        runtime.store.record_event(Event('liquidation', liquidation).json())
    while state.stored_trades < len(account.trades):
        account.trades[state.stored_trades].setdefault('session_id', runtime.session.session_id)
        runtime.store.record_trade(account.trades[state.stored_trades])
        state.stored_trades += 1


def manage_open_position(runtime, symbol, price, now_ms=None):
    """Apply the exit policy to a position that is already open.

    Levels are set once when the trade is planned and nothing used to touch them again,
    so a winner could round-trip all the way back to its original stop. This runs on the
    live tick, which is the only path fast enough to matter: a stop that only moves when
    a 5-minute bar closes is not a trailing stop.

    Only the stop and target move. Entry, size and direction belong to the position and
    are never rewritten here, because the risk engine sized the trade against them.
    """
    account = runtime.account
    position = account.positions.get(symbol)
    if position is None:
        return None
    policy = getattr(runtime.decisions, 'exit_policy', None) or exit_policy.get_policy()
    atr = None
    rows = runtime.cache.closed_rows.get(symbol)
    if rows:
        try:
            atr = features_snapshot(rows).get('atr')
        except (ValueError, TypeError, KeyError):
            atr = None
    state = runtime.decision_state
    opened = int(state.position_opened_at.get(symbol) or 0)
    now_ms = now_ms or _now_ms()
    bars_held = 0
    # Settings.interval_ms, not a getattr default: the getattr form silently used five
    # minutes for every interval, so a 1m session held positions for 5x its configured
    # time stop -- exactly cancelling the measured-horizon alignment this policy exists for.
    interval_ms = int(getattr(runtime.settings, 'interval_ms', 0) or 0) or interval_to_ms(
        getattr(runtime.settings, 'interval', '1m'))
    if opened and interval_ms > 0:
        bars_held = int((now_ms - opened) // interval_ms)
    decision = exit_policy.manage_position(policy, position, price, atr, bars_held)
    if not decision:
        return None
    if decision.get('exit_now'):
        runtime.store.record_event(Event('exit_policy_triggered', {
            'symbol': symbol, 'reason': decision['reason'],
            'policy': policy.name, 'price': price, 'bars_held': bars_held}).json())
        return close_position(runtime, symbol, decision['reason'], price)
    moved = False
    if decision['stop'] != position.stop:
        position.stop = decision['stop']
        moved = True
    if decision['target'] != position.target:
        position.target = decision['target']
        moved = True
    if not moved:
        return None
    runtime.store.set_runtime('paper_account', account.snapshot())
    runtime.store.record_event(Event('exit_levels_adjusted', {
        'symbol': symbol, 'reason': decision['reason'], 'policy': policy.name,
        'stop': position.stop, 'target': position.target,
        'entry': position.entry, 'price': price}).json())
    return {'symbol': symbol, 'stop': position.stop, 'target': position.target,
            'reason': decision['reason']}


async def evaluate_tick(runtime, symbol, price):
    """Per-trade decision path; throttled per symbol."""
    settings = runtime.settings
    session = runtime.session
    if not settings.tick_strategy_enabled or session.status != 'running':
        return
    now_ms = _now_ms()
    # Managing open exposure comes first. A position that needs its stop tightened should
    # not wait behind a model inference that may take longer than the move itself.
    manage_open_position(runtime, symbol, price, now_ms)
    state = runtime.decision_state
    if now_ms - state.last_tick_strategy.get(symbol, 0) < settings.strategy_min_interval_ms:
        return
    rows = runtime.cache.closed_rows.get(symbol)
    if rows is None:
        rows = runtime.store.candles(symbol, settings.interval, settings.lookback, closed_only=True)
        runtime.cache.closed_rows[symbol] = rows
    live = runtime.cache.live_candles.get(symbol)
    if live and (not rows or live.get('open_time') != rows[-1].get('open_time')):
        rows = [*rows, live]
    if len(rows) < MIN_ROWS:
        return
    session_id = session.session_id
    signal = await runtime.decisions.signal(symbol, rows, price)
    observe_features(runtime, signal)
    candidate = runtime.decisions.plan(signal)
    state.last_tick_strategy[symbol] = now_ms
    record_signal(runtime, symbol, signal, candidate, 'tick', {'strategy_feed': 'tick'})
    previous = state.per_symbol.get(symbol, {})
    if not candidate or not runtime.session.can_enter(symbol):
        return
    if not execution_ready(runtime.cache.latest.get(symbol, {}), now_ms)[0]:
        return
    # The retry floor is what stops a structurally rejected entry from being retried on
    # every book event. The previous version only wrote the state back when an order was
    # actually created, so a symbol refused for max_positions retried at the tick interval
    # and appended two audit rows each time. One bar is the natural re-evaluation period;
    # ENTRY_REJECT_RETRY_MS overrides it.
    retry_ms = int(getattr(settings, 'entry_reject_retry_ms', 0) or 0) or int(
        getattr(settings, 'interval_ms', 0) or 0) or 300000
    if previous.get('side') == signal['side'] and previous.get('candidate'):
        if now_ms - int(previous.get('at', 0)) < retry_ms:
            return
    order = submit_entry(runtime, symbol, signal, candidate, 'tick', now_ms,
                         expected_session=session_id)
    # Written back whether or not the order was created: an attempt is an attempt, and
    # forgetting the refusals is what produced the event storm.
    state.per_symbol[symbol] = {**previous, 'at': now_ms, 'side': signal['side'],
                                'candidate': True, 'submitted': bool(order)}


async def evaluate_bar(runtime, symbol, rows):
    """Closed-bar decision path; at most one decision per bar per symbol."""
    settings = runtime.settings
    session = runtime.session
    decision_rows = [row for row in rows if row.get('is_closed')]
    if len(decision_rows) < MIN_ROWS:
        return
    runtime.shadow.submit(symbol, decision_rows)
    bar_time = decision_rows[-1].get('open_time', 0)
    now_ms = _now_ms()
    state = runtime.decision_state
    prior = state.per_symbol.get(symbol, {})
    if prior.get('bar_time') == bar_time and now_ms - int(prior.get('at', 0)) < settings.strategy_min_interval_ms:
        return
    # Score any earlier forecast for this symbol whose horizon has now elapsed. Doing it
    # on the closed-bar path keeps the tracker a function of completed bars rather than
    # of tick arrival order.
    tracker = getattr(runtime.decisions, 'symbol_edge', None)
    if tracker is not None:
        times = [int(row.get('open_time') or 0) for row in decision_rows]
        closes = [float(row.get('close') or 0) for row in decision_rows]
        tracker.resolve(symbol, times, closes)
    session_id = session.session_id
    signal = await runtime.decisions.signal(symbol, decision_rows)
    observe_features(runtime, signal)
    candidate = runtime.decisions.plan(signal)
    state.per_symbol[symbol] = {'bar_time': bar_time, 'at': now_ms,
                                'side': signal['side'], 'confidence': signal['confidence'],
                                'candidate': bool(candidate)}
    record_signal(runtime, symbol, signal, candidate, 'bar',
                  {'forming_candle': not bool(rows[-1].get('is_closed'))})
    if runtime.session.status == 'running':
        submit_entry(runtime, symbol, signal, candidate, 'bar', now_ms,
                     expected_session=session_id)
    flush_account_events(runtime)
    persist_symbol_edge(runtime)


async def prune_events(runtime):
    """Move aged audit rows out of the hot events table.

    Runs on the database thread, never on the event loop. The sweep is incremental, so
    after the first pass it usually finds nothing and costs one indexed count.
    """
    settings = runtime.settings
    keep_rows = int(getattr(settings, 'event_retention_rows', 0) or 0)
    if keep_rows <= 0:
        return
    if runtime.database_worker is None:
        # Loud rather than silent. The attribute existed on neither Runtime nor its
        # defaults, so every sweep raised AttributeError into a bare except and the audit
        # tables grew without bound while the dashboard reported a clean system.
        raise RuntimeError('database_worker_not_attached')
    durable = bool(getattr(settings, 'event_retention_durable', True))
    moved = await runtime.database_worker.call('prune_events', keep_rows, 5000, durable)
    # And bound the archive the pruned rows went into. It had no retention and no reader,
    # so it accumulated every row retention ever moved -- 1.02M and rising -- while the
    # rows in it were unreachable.
    archive_rows = int(getattr(runtime.settings, 'event_archive_rows', 0) or 0)
    if archive_rows > 0:
        await runtime.database_worker.call('prune_archive', archive_rows)
    if moved:
        runtime.store.record_event(Event('retention_pruned', {'moved': moved}).json())


async def run_decision_loop(runtime):
    """Periodic bar loop: refresh market, resubscribe, decide, then persist."""
    settings = runtime.settings
    session = runtime.session
    store = runtime.store
    cache = runtime.cache
    pending_plans = runtime.pending_plans
    try:
        await prune_events(runtime)
    except Exception as exc:  # retention must never stop trading
        runtime.errors.note('retention', exc)
    expire_entries(runtime)
    now_seconds = time.time()
    if not cache.market or now_seconds - cache.last_refresh >= settings.market_refresh_seconds:
        try:
            snapshot = await runtime.api.market_snapshot()
        except Exception as exc:
            # An unhandled failure here used to abort the entire iteration: no klines were
            # written, no decision was taken and retention never ran, and the exception was
            # swallowed one level up. A rate-limited venue is the common case, not a bug.
            runtime.errors.note('market_snapshot', exc)
            snapshot = None
        if snapshot is None:
            snapshot = dict(cache.market) or {}
        # Contract specs belong to the broker, not to the cached market rows that get
        # spread into the dashboard payload.
        specs = snapshot.pop('specs', None)
        if specs:
            runtime.broker.specs.update(specs)
        cache.market.clear()
        cache.market.update(snapshot)
        cache.last_refresh = now_seconds
    runtime.feeds.last_event = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cache.market['trained'] = runtime.models.trained_market(cache.market)
    mark_rows = {x['symbol']: x.get('mark_price', x.get('price', 0)) for x in cache.market.get('all', []) if x.get('symbol')}
    funding_rows = {x['symbol']: x.get('funding_rate', 0) for x in cache.market.get('all', []) if x.get('symbol')}
    if session.status == 'running':
        # Per-symbol settlement windows. The venue runs 8h funding on most contracts and
        # 4h or 1h on others; a single account-wide window mis-charged both directions.
        windows = dict(getattr(runtime.api, '_funding_intervals', None) or {})
        runtime.account.apply_funding(funding_rows, _now_ms(),
                                      interval_ms={s: windows[s] for s in runtime.account.positions
                                                   if s in windows} or None)
        # The meta-labeling gate is a property of a symbol, and the ranking is a property
        # of the market. Left independent they disagree, and the failure mode is silent:
        # a session selected five symbols that all failed the gate and never traded once.
        tier = _edge_tier(runtime)
        explore = int(getattr(settings, 'symbol_edge_explore_slots', 1) or 0)
        # Widen the pool once per refresh rather than per call, so choose and reselect see
        # the same candidate set.
        pool = {**cache.market, session.source: _candidates(cache.market, session.source, tier)}
        if not session.selected_symbols:
            session.choose(pool, runtime.limits.max_positions,
                           tier=tier, explore_slots=explore)
        else:
            # Replace symbols whose book never arrives. Every one of them costs a model
            # evaluation per tick and can never fill, and the session has a fixed number
            # of position slots, so a dead symbol is a slot doing nothing.
            #
            # Guarded on the transport being up, because the two failures are
            # indistinguishable from the session's side and only one of them is the
            # symbol's fault. With the socket down, every symbol looks dead at once and
            # the session re-picked its whole selection every ~70 seconds -- 54 times in
            # 24 hours -- discarding the pending forecasts the meta-labeling gate needs,
            # since those are resolved only for symbols still held. The gate then refused
            # every symbol forever, which read as a modelling verdict rather than the
            # transport outage it was.
            dead = session.without_market_data(_now_ms(), feed_live=_book_feed_live(runtime))
            if dead:
                session.reselect(pool, runtime.limits.max_positions, dead,
                                 tier=tier, explore_slots=explore)
            elif tier is not None and not any(tier(s) >= 2 for s in session.selected_symbols):
                # Nothing the session holds can be traded. Re-pick, throttled so a market
                # with no eligible symbol at all does not re-rank on every refresh.
                now_ms = _now_ms()
                if now_ms - int(session.last_selection_at or 0) >= RESELECT_COOLDOWN_MS:
                    session.choose(pool, runtime.limits.max_positions,
                                   tier=tier, explore_slots=explore)
    account = runtime.account
    active = session.selected_symbols or list(settings.symbols) or [x['symbol'] for x in cache.market.get('hot', [])[:10]]
    subscribed = set(active) | set(account.positions) | {order.symbol for order in runtime.broker.open_orders()}
    await runtime.feeds.ws.set_symbols(subscribed)
    await runtime.feeds.public.set_symbols(subscribed)
    raw_results = await asyncio.gather(*(runtime.api.klines(symbol, settings.interval, settings.lookback) for symbol in active),
                                       return_exceptions=True)
    raw_by_symbol = {symbol: raw for symbol, raw in zip(active, raw_results) if not isinstance(raw, Exception)}
    for symbol in active:
        try:
            raw = raw_by_symbol.get(symbol, [])
            if not raw:
                # A failed or rate-limited fetch arrives here as an empty list because the
                # gather was run with return_exceptions=True. Assigning it unconditionally
                # replaced a good local history with nothing, so one timeout blinded the
                # symbol until the next successful poll.
                runtime.errors.note('klines_unavailable', symbol)
                continue
            runtime.errors.ok('klines_unavailable')
            runtime.candle_writer.submit_nowait((symbol, settings.interval, raw, _now_ms()))
            cache.closed_rows[symbol] = [row for row in raw if row.get('is_closed')][-settings.lookback:]
            if raw:
                cache.live_candles[symbol] = raw[-1]
            # Exits are evaluated against the closed bar's high and low as well as against
            # the latest price. The mark pump only ever sees a single price, so a bar that
            # pierced the stop and closed back above it left the position open -- the paper
            # account survived every wick that would have taken the real one out. This is
            # the same conservative rule the backtest applies, so the two now agree.
            if symbol in runtime.account.positions:
                closed_bar = raw[-1] if raw and raw[-1].get('is_closed') else None
                close_price = float(closed_bar.get('close') if closed_bar else 0) or None
                if close_price:
                    # This is the bar boundary, and the only caller that records an
                    # equity sample, so the curve is one point per closed bar.
                    runtime.account.mark({symbol: close_price},
                                         bars={symbol: closed_bar} if closed_bar else None,
                                         record=True)
            await evaluate_bar(runtime, symbol, raw)
        except Exception as exc:  # keep one bad symbol from stopping the loop
            runtime.errors.note('bar_loop', '%s: %r' % (symbol, exc))
    # Expiring derivative statistics. Separate from the retention sweep because this is the
    # one piece of data whose loss is permanent.
    collector = getattr(runtime, 'derivatives_collector', None)
    if collector is not None and collector.due(_now_ms()):
        try:
            universe = sorted(set(session.selected_symbols) | set(account.positions)
                              | set(settings.symbols))[:max(1, int(settings.derivatives_collect_symbols))]
            await collector.run(universe)
            runtime.errors.ok('derivatives_collect')
        except Exception as exc:
            runtime.errors.note('derivatives_collect', exc)
    reconcile_interval = float(getattr(settings, 'reconcile_interval_seconds', 0) or 0)
    if getattr(settings, 'reconcile_enabled', False) and reconcile_interval > 0:
        if now_seconds - float(getattr(cache, 'last_reconcile', 0) or 0) >= reconcile_interval:
            cache.last_reconcile = now_seconds
            try:
                from ..backtest.account_reconcile import reconcile

                report = reconcile(runtime)
                runtime.reconciliation = report
                if report['consistent']:
                    runtime.errors.ok('reconcile')
                else:
                    runtime.errors.note('reconcile', ','.join(report['findings']))
                    runtime.store.record_event(Event('reconciliation', {
                        'session_id': session.session_id,
                        'findings': report['findings'],
                        'positions': report.get('positions'),
                        'orders': report.get('orders')}).json())
            except Exception as exc:
                runtime.errors.note('reconcile', exc)
    flow = getattr(runtime, 'flow', None)
    if flow is not None and flow.drain_due(_now_ms()):
        try:
            rows = flow.drain(_now_ms())
            if rows:
                runtime.store.record_flow(rows)
                runtime.errors.ok('flow_drain')
        except Exception as exc:
            runtime.errors.note('flow_drain', exc)
    monitor = getattr(runtime, 'drift', None)
    if monitor is not None and monitor.due(_now_ms()):
        try:
            report = monitor.check(_now_ms())
            if report.get('status') == 'drift':
                runtime.store.record_event(Event('feature_drift', {
                    'session_id': session.session_id,
                    'drifted': report.get('drifted'),
                    'threshold': report.get('threshold'),
                    'observations': report.get('observations'),
                    'policy': report.get('policy')}).json())
                runtime.errors.note('feature_drift', 'drifted=%s' % (report.get('drifted'),))
            else:
                runtime.errors.ok('feature_drift')
        except Exception as exc:
            runtime.errors.note('feature_drift', exc)
    interval = float(getattr(settings, 'event_retention_interval_seconds', 0) or 0)
    if interval > 0 and now_seconds - cache.last_retention >= interval:
        cache.last_retention = now_seconds
        try:
            await prune_events(runtime)
        except Exception as exc:
            runtime.errors.note('retention', exc)
    store.set_runtime('paper_account', account.snapshot())
    store.set_runtime('risk_state', runtime.risk.snapshot())
    store.set_runtime('processed_bars', runtime.decision_state.processed_bars)
    store.set_runtime('decision_state', runtime.decision_state.per_symbol)
