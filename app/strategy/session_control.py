"""Session lifecycle handlers: manual close, risk reset, session start."""
import time

from ..trading import exit_policy, risk_profiles
from ..core.domain import Event


def set_exit_policy(runtime, name=None, overrides=None):
    """Switch the exit policy: stop width, reward:risk, trailing and time stop.

    Applies to the next plan immediately and to already-open positions on their next
    tick, so a live trade can be re-managed without a restart.
    """
    policy = exit_policy.resolve(name, overrides)
    runtime.decisions.exit_policy = policy
    runtime.store.set_runtime('exit_policy', policy.snapshot())
    runtime.store.set_runtime('exit_policy_name', policy.name)
    runtime.store.record_event(Event('exit_policy_changed', {
        'session_id': runtime.session.session_id,
        'policy': policy.snapshot()}).json())
    return policy


def set_profile(runtime, name=None, overrides=None):
    """Switch the active risk profile, optionally with per-session overrides.

    Applies to the live engine and limits immediately, so a running session can be
    re-tuned without a restart. The resolved values are persisted and recorded, because
    a trade that cannot be explained by the parameters in force when it happened is not
    reviewable after the fact.
    """
    profile = risk_profiles.resolve(name, overrides)
    runtime.risk.apply_profile(profile)
    runtime.limits.apply_profile(profile)
    runtime.store.set_runtime('risk_profile', profile.snapshot())
    runtime.store.set_runtime('risk_profile_name', profile.name)
    runtime.store.set_runtime('risk_state', runtime.risk.snapshot())
    runtime.store.record_event(Event('risk_profile_changed', {
        'session_id': runtime.session.session_id,
        'profile': profile.snapshot()}).json())
    return profile


def on_session_start(runtime, profile_name=None, overrides=None):
    runtime.decision_state.stored_trades = 0
    runtime.decision_state.processed_bars.clear()
    if profile_name or overrides:
        set_profile(runtime, profile_name, overrides)
    runtime.risk.reset_for_session(runtime.session.initial_cash)
    runtime.store.set_runtime('processed_bars', runtime.decision_state.processed_bars)
    runtime.store.set_runtime('risk_state', runtime.risk.snapshot())


def close_position(runtime, symbol, reason='manual', price=None):
    """Flatten one position at the given price (default: the current mark).

    An explicit price lets the exit policy close at the level it decided on rather than
    at whatever the mark has drifted to by the time the close runs.
    """
    symbol = symbol.upper()
    account = runtime.account
    position = account.positions.get(symbol)
    if position is None:
        return None
    if price is None:
        quote = runtime.cache.latest.get(symbol, {})
        price = float(account.marks.get(symbol) or quote.get('mark_price') or quote.get('price') or position.entry)
    pnl = account.close(symbol, price, reason, int(time.time() * 1000))
    runtime.decision_state.position_opened_at.pop(symbol, None)
    trade = account.trades[-1]
    trade['session_id'] = runtime.session.session_id
    runtime.store.record_trade(trade)
    runtime.store.record_event(Event('manual_position_close', {**trade, 'close_price': price}).json())
    runtime.store.record_equity(account.cash, account.equity, account.unrealized_pnl(), len(account.positions))
    runtime.store.set_runtime('paper_account', account.snapshot())
    runtime.decision_state.stored_trades = len(account.trades)
    return {'symbol': symbol, 'price': price, 'pnl': pnl, 'trade': trade}


def reset_risk(runtime):
    runtime.risk.reset_for_session(runtime.account.equity)
    snapshot = runtime.risk.snapshot()
    runtime.store.set_runtime('risk_state', snapshot)
    runtime.store.record_event(Event('risk_manual_reset', {
        'session_id': runtime.session.session_id,
        'day_start_equity': runtime.risk.day_start_equity}).json())
    return snapshot
