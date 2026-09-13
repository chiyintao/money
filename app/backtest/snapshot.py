"""State projection for the HTTP layer.

Before this module, position math, PnL aggregation and margin figures were written
out longhand in three places (state(), realtime_state() and the account audit), and
they had drifted: state() aggregated realized PnL from every historical session while
equity came from the live account. Everything now flows through the helpers here so a
figure is computed once and reused.
"""
import datetime
import time

from ..core.metrics import summary


def _feature_space(runtime):
    """Whether the serving gate's bounds can actually refuse anything.

    A gate that reports green because it is unreachable is worse than no gate, so the
    distinction is surfaced rather than left in the model loader's internals.
    """
    space = getattr(getattr(runtime, 'models', None), 'feature_space', None) or {}
    quality = space.get('bounds_quality') or {}
    return {'status': space.get('status'), 'verified': bool(space.get('verified')),
            'rows': space.get('rows'), 'sha256': space.get('sha256'),
            'bounds_method': space.get('bounds_method'),
            'ood_gate_informative': quality.get('informative'),
            'ood_degenerate_features': quality.get('degenerate') or []}


def _flow(runtime):
    collector = getattr(runtime, 'flow', None)
    if collector is None:
        return {'enabled': False}
    return dict(collector.health(), enabled=True)


def _drift(runtime):
    monitor = getattr(runtime, 'drift', None)
    if monitor is None:
        return {'enabled': False}
    return dict(monitor.health(), enabled=True)


def _derivatives(runtime):
    collector = getattr(runtime, 'derivatives_collector', None)
    if collector is None:
        return {'enabled': False}
    return dict(collector.health(), enabled=True)


def position_view(account, symbol, position):
    """One open position, with every derived figure computed once."""
    mark = account.marks.get(symbol, position.entry)
    notional = mark * position.qty
    direction = 1 if position.side == 'LONG' else -1
    return {
        'symbol': symbol,
        'side': position.side,
        'qty': position.qty,
        'entry': position.entry,
        'mark': mark,
        'notional': notional,
        'margin': account.margin.initial_margin(notional),
        'liquidation_price': account.margin.liquidation_price(position),
        'unrealized_pnl': (mark - position.entry) * position.qty * direction,
        'stop': position.stop,
        'target': position.target,
        'funding_paid': position.funding_paid,
        'order_id': position.order_id,
        'opened_at': position.opened_at,
        'leverage': account.margin.leverage,
    }


def position_views(account):
    return [position_view(account, symbol, position) for symbol, position in account.positions.items()]


def position_totals(account, positions):
    equity = account.equity
    total_notional = sum(item['notional'] for item in positions)
    used_margin = sum(item['margin'] for item in positions)
    return {
        'total_notional': total_notional,
        'used_margin': used_margin,
        'maintenance_margin': sum(account.margin.maintenance_margin(item['notional']) for item in positions),
        'margin_usage_pct': used_margin / equity * 100 if equity else 0.0,
        'gross_leverage': total_notional / equity if equity else 0.0,
    }


def lifetime_return_pct(trades, capitals):
    """Lifetime PnL as a percentage of each session's own starting capital.

    A dollar sum across sessions is not a quantity. The session form lets the operator
    pick the starting capital, so the trade table holds rounds made against 100 and
    against 10,000 side by side; adding their PnL reported -697.29 over a history where
    the single largest account contributed most of the absolute loss while the small
    accounts were down a far larger fraction. Returns are comparable across accounts and
    dollars are not.

    Trades whose session has no recorded capital are attributed to the median of the
    capitals that do, rather than dropped or counted at face value: dropping them hides
    real losses, and counting them as dollars silently reintroduces the mixing.
    """
    if not trades:
        return None, 0
    known = sorted(capitals.values())
    fallback = known[len(known) // 2] if known else None
    attributable = 0
    total = 0.0
    for trade in trades:
        capital = capitals.get(trade.get('session_id')) or fallback
        if not capital:
            continue
        total += float(trade.get('pnl', 0)) / capital
        attributable += 1
    if not attributable:
        return None, 0
    return total * 100.0, attributable


def session_accounting(store, session):
    """Realized figures scoped to the live session, with lifetime totals kept apart.

    The account balance only moves inside the current session, so mixing in trades
    from earlier sessions made realized PnL disagree with equity.
    """
    session_trades = store.trades_for_session(session.session_id, 500)
    all_time = store.recent_trades(500)
    # An open position has already paid its entry fee and any funding, both deducted
    # from cash, but neither appears in a closed trade yet. Counting only closed
    # trades made total_fees/total_funding disagree with the balance the UI shows.
    open_positions = list(getattr(getattr(session, 'account', None), 'positions', {}).values())
    open_fees = sum(float(getattr(p, 'entry_fee', 0) or 0) for p in open_positions)
    open_funding = sum(float(getattr(p, 'funding_paid', 0) or 0) for p in open_positions)
    try:
        capitals = store.session_capitals()
    except Exception:
        # A store too old to have the method, or a read failure, must not take the whole
        # snapshot down: callers get None and the UI shows the lifetime figure as unknown
        # rather than as a confidently wrong dollar amount.
        capitals = {}
    lifetime_pct, lifetime_n = lifetime_return_pct(all_time, capitals)
    return {
        'session_trade_rows': session_trades,
        'all_time_trade_rows': all_time,
        'realized_pnl': sum(float(t.get('pnl', 0)) for t in session_trades),
        'total_fees': sum(float(t.get('fees', 0)) for t in session_trades) + open_fees,
        'total_funding': sum(float(t.get('funding', 0)) for t in session_trades) + open_funding,
        'win_rate_pct': (sum(1 for t in session_trades if float(t.get('pnl', 0)) > 0) / len(session_trades) * 100) if session_trades else 0.0,
        # Kept for the existing chart and API shape, but it is no longer what the UI
        # reports as the lifetime figure: see lifetime_return_pct for why the sum is not
        # a quantity. It stays because removing a field the frontend reads is a larger
        # change than this fix needs.
        'all_time_realized_pnl': sum(float(t.get('pnl', 0)) for t in all_time),
        'lifetime_return_pct': lifetime_pct,
        'lifetime_trades': lifetime_n,
    }


def drawdown_curve(rows):
    """Attach running-peak drawdown to an equity series."""
    peak = None
    for item in rows:
        total = float(item['total'])
        peak = total if peak is None else max(peak, total)
        item['drawdown_pct'] = ((peak - total) / peak * 100) if peak else 0.0
    return rows


def account_block(account):
    return {
        'equity': account.equity,
        'cash': account.cash,
        'available_margin': account.available_margin(),
        'unrealized_pnl': account.unrealized_pnl(),
        'return_pct': (account.equity - account.initial_cash) / account.initial_cash * 100 if account.initial_cash else 0.0,
    }


def session_block(session):
    return {
        'session_id': session.session_id,
        'status': session.status,
        'source': session.source,
        'leverage': session.leverage,
        'selected_symbols': session.selected_symbols,
        'initial_cash': session.initial_cash,
        'started_at': session.started_at,
        'ended_at': session.ended_at,
        'end_reason': session.end_reason,
        'step': session.step,
    }


def mark_ages(last_mark_event):
    now = int(time.time() * 1000)
    return {symbol: now - int(stamp) for symbol, stamp in last_mark_event.items()}


def build_state(runtime, connectors):
    # connectors is spread into the payload rather than nested: the dashboard and
    # existing consumers read the flat keys (connector_health, connector_routes,
    # feed_normalizer, ws_reconnects).
    """Full state payload for the dashboard."""
    account = runtime.account
    session = runtime.session
    positions = position_views(account)
    totals = position_totals(account, positions)
    books = session_accounting(runtime.store, session)
    decision_counts = runtime.decisions.counts
    return {
        'schema_version': 4,
        'mode': 'paper',
        **account_block(account),
        **totals,
        'realized_pnl': books['realized_pnl'],
        'total_pnl': books['realized_pnl'] + account.unrealized_pnl(),
        'total_fees': books['total_fees'],
        'total_funding': books['total_funding'],
        'session_trades': len(books['session_trade_rows']),
        'all_time_realized_pnl': books['all_time_realized_pnl'],
        'lifetime_return_pct': books['lifetime_return_pct'],
        'lifetime_trades': books['lifetime_trades'],
        'win_rate_pct': books['win_rate_pct'],
        'positions_detail': positions,
        'positions': len(account.positions),
        'liquidations': account.liquidations[-20:],
        'trades': len(account.trades),
        'last_error': runtime.last_error,
        'errors': runtime.errors.snapshot(),
        'feature_space': _feature_space(runtime),
        'derivatives': _derivatives(runtime),
        'venue': dict(getattr(runtime, 'venue', None) or {}),
        'drift': _drift(runtime),
        'reconciliation': dict(getattr(runtime, 'reconciliation', None) or {}),
        'flow': _flow(runtime),
        'retrain': dict(getattr(getattr(runtime, 'retrainer', None), 'snapshot', dict)() or {}),
        'last_market_event': runtime.feeds.last_event,
        'ws_events': runtime.feeds.ws_events,
        **connectors,
        'persistence': {
            'events': runtime.event_writer.stats() if runtime.event_writer else {},
            'candles': runtime.candle_writer.stats() if runtime.candle_writer else {},
        },
        'updated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'symbols': list(runtime.cache.latest.values()),
        'metrics': summary(account.equity_curve, account.trades,
                           initial_equity=account.initial_cash,
                           interval_ms=runtime.settings.interval_ms),
        'equity_curve': runtime.store.equity_curve(),
        'recent_trades': books['session_trade_rows'],
        'all_time_trades': books['all_time_trade_rows'],
        'simulation_history': runtime.store.simulation_history(),
        'events': runtime.store.recent_events(100),
        'derivatives_history': runtime.store.derivatives(limit=100),
        'model_status': {
            'status': runtime.model_status.get('status'),
            'rows': runtime.model_status.get('rows'),
            'test_accuracy': runtime.model_status.get('test', {}).get('directional_accuracy_pct'),
        },
        'decision_model': runtime.decisions.brief(),
        'decision_counts': decision_counts,
        'simulation': runtime.session.snapshot(),
        'risk_state': runtime.risk.snapshot(),
        'account_audit': account.audit(),
        'open_orders': [order.__dict__ for order in runtime.broker.open_orders()],
        'order_history': [order.__dict__ for order in runtime.broker.orders.values()],
        'order_status_counts': runtime.broker.order_counts(),
        # The assumptions behind every fill price and fee in the history above. They were
        # constructor defaults with no way to see or change them, so a stored session could
        # not be explained by the costs it assumed.
        'cost_model': runtime.broker.cost_model(),
        'processed_bars': runtime.decision_state.processed_bars,
        'mark_event_age_ms': mark_ages(runtime.feeds.last_mark_event),
        **runtime.cache.market,
    }


def build_realtime_state(runtime):
    """Lean slice polled at 10Hz; only figures that move with price."""
    account = runtime.account
    positions = position_views(account)
    totals = position_totals(account, positions)
    books = session_accounting(runtime.store, runtime.session)
    return {
        'realtime': True,
        'schema_version': 4,
        **account_block(account),
        **totals,
        'realized_pnl': books['realized_pnl'],
        'total_pnl': books['realized_pnl'] + account.unrealized_pnl(),
        'total_fees': books['total_fees'],
        'session_trades': len(books['session_trade_rows']),
        'all_time_realized_pnl': books['all_time_realized_pnl'],
        'lifetime_return_pct': books['lifetime_return_pct'],
        'lifetime_trades': books['lifetime_trades'],
        'win_rate_pct': books['win_rate_pct'],
        'risk_state': runtime.risk.snapshot(),
        'positions_detail': positions,
        'positions': len(positions),
        'simulation': session_block(runtime.session),
        'symbols': list(runtime.cache.latest.values()),
        'mark_event_age_ms': mark_ages(runtime.feeds.last_mark_event),
        'decision_model': runtime.decisions.brief(),
        'updated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'ws_events': runtime.feeds.ws_events,
    }


def build_chart_data(runtime, symbol):
    store = runtime.store
    interval = runtime.settings.interval
    lookback = runtime.settings.lookback
    candles = store.candles(symbol, interval, lookback, closed_only=True)
    equity = drawdown_curve(store.equity_curve(limit=500))
    trades = [trade for trade in store.recent_trades(500) if trade.get('symbol') == symbol]
    return {'symbol': symbol, 'candles': candles, 'trades': trades, 'equity_curve': equity}
