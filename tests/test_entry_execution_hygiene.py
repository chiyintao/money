"""Entry execution hygiene: deadline, drift limits and fill re-anchoring.

Each test here encodes a failure observed in the stored paper sessions, where 501
orders were submitted, 441 of them never reached a terminal status, and the fills that
did happen landed up to 4.3 hours after their decision at prices the plan's stop and
target were never computed for.
"""
import time

import pytest

from app.trading.broker import PaperBroker
from app.core.domain import OrderIntent
from app.trading.execution import cancel_entry, entry_drift_fraction, reanchor_plan, settle_order
from app.trading.risk import RiskEngine
from app.trading.simulation import PaperAccount


class Store:
    def __init__(self):
        self.events = []

    def record_event(self, payload):
        self.events.append(payload)

    def set_runtime(self, key, value):
        pass

    def reasons(self):
        return [event['payload'].get('reason') for event in self.events if event['type'] == 'order_canceled']


class Session:
    session_id = 's-test'
    step = 0

    def __init__(self):
        self.cooldowns = []

    def set_cooldown(self, symbol, minutes=5):
        self.cooldowns.append((symbol, minutes))

    def can_enter(self, symbol, now_ms=None):
        return True


class Settings:
    order_ttl_ms = 90000
    max_entry_adverse_fraction = .5
    max_entry_chase_fraction = .35
    entry_cooldown_minutes = 5


class Limits:
    def approve(self, symbol, notional, open_notionals, equity=None):
        return True, 'portfolio_ok'


class Guard:
    def approve_new_entry(self, symbol, now_ms=None):
        return True, 'fresh_data'


class Runtime:
    def __init__(self, quote, symbol='X'):
        self.settings = Settings()
        self.broker = PaperBroker(order_ttl_ms=Settings.order_ttl_ms)
        self.account = PaperAccount(cash=10000.0)
        self.store = Store()
        self.session = Session()
        self.risk = RiskEngine(max_risk=.005, max_daily_loss=.2, max_gross_leverage=2, day_start_equity=10000)
        self.limits = Limits()
        self.guard = Guard()
        self.pending_plans = {}
        self.cache = type('Cache', (), {'latest': {symbol: quote}})()


def quote(bid=99.5, ask=99.6, mark=99.55, bid_qty=100.0, ask_qty=100.0):
    now = int(time.time() * 1000)
    return {'book_time': now, 'mark_time': now, 'bid': bid, 'ask': ask, 'mark_price': mark,
            'bid_qty': bid_qty, 'ask_qty': ask_qty,
            'bids': [[bid, bid_qty]], 'asks': [[ask, ask_qty]]}


def short_plan(order_id='p1', entry=100.0, stop=110.0, target=80.0):
    return {'symbol': 'X', 'side': 'SHORT', 'entry': entry, 'stop': stop, 'take_profit': target,
            'stop_distance': abs(stop - entry), 'target_distance': abs(entry - target),
            'order_id': order_id}


def submit(runtime, quantity=1.0, side='SELL', order_id='p1', plan=None):
    plan = plan or short_plan(order_id)
    order, reason = runtime.broker.submit(OrderIntent('X', side, quantity, order_id=order_id),
                                          plan=plan)
    assert reason == 'accepted'
    runtime.pending_plans[order_id] = plan
    return order


# --------------------------------------------------------------- deadline

def test_expired_entry_is_cancelled_instead_of_filled():
    runtime = Runtime(quote())
    # Submitted two minutes ago with a ninety-second deadline: the order outlived its
    # own validity window while it rested, which is how fills landed hours after the bar
    # that produced them.
    past = int(time.time() * 1000) - 120000
    runtime.broker.submit(OrderIntent('X', 'SELL', 1.0, order_id='old'),
                          timestamp=past, plan=short_plan('old'))
    runtime.pending_plans['old'] = short_plan('old')

    settle_order(runtime, 'old')

    assert runtime.broker.orders['old'].status == 'CANCELED'
    assert runtime.account.positions == {}
    assert runtime.account.trades == []
    assert runtime.pending_plans == {}
    assert runtime.store.reasons() == ['entry_expired']


def test_order_without_explicit_deadline_still_gets_one():
    broker = PaperBroker(order_ttl_ms=1000)
    order, _ = broker.submit(OrderIntent('X', 'BUY', 1), timestamp=5000)
    assert order.expires_at == 6000
    assert [o.order_id for o in broker.expire_orders(6000)] == [order.order_id]


# ------------------------------------------------------------ drift limits

def test_entry_abandoned_when_market_moved_against_plan():
    # LONG planned at 100 with a 10-point stop; the ask is already 6 points down.
    runtime = Runtime(quote(bid=93.9, ask=94.0, mark=93.95))
    plan = {'symbol': 'X', 'side': 'LONG', 'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0,
            'stop_distance': 10.0, 'target_distance': 20.0}
    submit(runtime, side='BUY', plan=plan)

    settle_order(runtime, 'p1')

    assert runtime.broker.orders['p1'].status == 'CANCELED'
    assert runtime.account.positions == {}
    assert runtime.store.reasons() == ['entry_price_moved_against_plan']


def test_entry_abandoned_when_the_move_already_happened():
    # The exact stored case: EGLDUSDT shorted at a plan entry of 5.117 whose target was
    # 4.982, filled at 4.980. The fill was already past the target, so the position was
    # closed on the next mark for nothing but fees -- eight times in seven seconds.
    runtime = Runtime(quote(bid=4.9795, ask=4.9805, mark=4.980))
    plan = {'symbol': 'X', 'side': 'SHORT', 'entry': 5.117, 'stop': 5.191892857142857,
            'take_profit': 4.9821928571428575, 'stop_distance': 0.07489285714285704,
            'target_distance': 0.13480714285714268}
    submit(runtime, quantity=2.67, plan=plan)

    settle_order(runtime, 'p1')

    assert runtime.broker.orders['p1'].status == 'CANCELED'
    assert runtime.account.positions == {}
    assert runtime.account.trades == []
    assert runtime.store.reasons() == ['entry_price_already_past_target']


def test_drift_is_signed_and_scaled_by_each_bracket():
    plan = {'side': 'LONG', 'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0,
            'stop_distance': 10.0, 'target_distance': 20.0}
    assert entry_drift_fraction(plan, 100.0) == 0.0
    assert entry_drift_fraction(plan, 95.0) == pytest.approx(.5)      # halfway to the stop
    assert entry_drift_fraction(plan, 110.0) == pytest.approx(-.5)    # halfway to the target
    short = {'side': 'SHORT', 'entry': 100.0, 'stop': 110.0, 'take_profit': 80.0,
             'stop_distance': 10.0, 'target_distance': 20.0}
    assert entry_drift_fraction(short, 105.0) == pytest.approx(.5)
    assert entry_drift_fraction(short, 90.0) == pytest.approx(-.5)


# ------------------------------------------------------------- re-anchor

def test_reanchor_translates_both_levels_by_the_same_delta():
    plan = {'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0}
    adjusted = reanchor_plan(plan, 102.0)
    assert adjusted['stop'] == pytest.approx(92.0)
    assert adjusted['take_profit'] == pytest.approx(122.0)
    assert adjusted['planned_entry'] == 100.0
    assert adjusted['entry_deviation_bps'] == pytest.approx(200.0)
    # A short keeps its geometry too: stop above, target below, same distances.
    short = reanchor_plan({'entry': 100.0, 'stop': 110.0, 'take_profit': 80.0}, 98.0)
    assert short['stop'] == pytest.approx(108.0)
    assert short['take_profit'] == pytest.approx(78.0)


def test_fill_anchors_stop_and_target_to_the_traded_price():
    runtime = Runtime(quote(bid=99.5, ask=99.6, mark=99.55))
    submit(runtime, quantity=1.0)

    settle_order(runtime, 'p1')

    position = runtime.account.positions['X']
    assert runtime.broker.orders['p1'].status == 'FILLED'
    assert position.entry == pytest.approx(99.5, abs=.01)
    # Without re-anchoring these stayed at 110 and 80, leaving the stop further away
    # than the risk engine paid for.
    assert position.stop == pytest.approx(109.5, abs=.02)
    assert position.target == pytest.approx(79.5, abs=.02)
    # Marking at the fill price must not close the position for a fee-only loss.
    runtime.account.mark({'X': 99.55})
    assert 'X' in runtime.account.positions
    assert runtime.account.trades == []


# -------------------------------------------------------------- cooldown

def test_delayed_fill_starts_the_cooldown():
    runtime = Runtime(quote())
    submit(runtime, quantity=1.0)
    assert runtime.session.cooldowns == []

    settle_order(runtime, 'p1')

    assert runtime.session.cooldowns == [('X', 5)]


def test_cancelled_entry_does_not_start_a_cooldown():
    runtime = Runtime(quote(bid=90.0, ask=90.1, mark=90.05))
    submit(runtime, quantity=1.0)
    settle_order(runtime, 'p1')
    assert runtime.broker.orders['p1'].status == 'CANCELED'
    assert runtime.session.cooldowns == []


# ------------------------------------------------------- order bookkeeping

def test_open_order_index_follows_status_changes():
    broker = PaperBroker(slippage_bps=0)
    broker.submit(OrderIntent('X', 'BUY', 1, order_id='a'))
    broker.submit(OrderIntent('Y', 'BUY', 1, order_id='b'))
    assert broker.has_open_order('X') and not broker.has_open_order('Z')
    assert [o.order_id for o in broker.open_orders('X')] == ['a']
    assert sorted(o.order_id for o in broker.open_orders()) == ['a', 'b']

    broker.cancel('a')

    assert not broker.has_open_order('X')
    assert broker.open_orders('X') == []
    assert [o.order_id for o in broker.open_orders()] == ['b']


def test_submit_entry_refuses_a_second_order_for_the_same_symbol():
    from app.strategy.decision_loop import submit_entry

    # Depth smaller than the order keeps it resting, which is how 441 orders piled up.
    runtime = Runtime(quote(ask_qty=.1, bid_qty=.1))
    signal = {'side': 'LONG', 'confidence': .9, 'features': {'price': 100.0},
              'reason_codes': ['ensemble_agrees'], 'source': 'real_models'}
    candidate = {'symbol': 'X', 'side': 'LONG', 'entry': 100.0, 'stop': 90.0,
                 'take_profit': 120.0, 'stop_distance': 10.0}

    first = submit_entry(runtime, 'X', signal, candidate, 'bar')
    assert first is not None
    assert runtime.broker.orders[first.order_id].status == 'OPEN'

    assert submit_entry(runtime, 'X', signal, candidate, 'bar') is None
    assert len(runtime.broker.open_orders('X')) == 1


def test_submit_entry_refuses_when_a_position_is_already_open():
    from app.strategy.decision_loop import submit_entry

    runtime = Runtime(quote())
    runtime.account.positions['X'] = type('P', (), {'entry': 99.5, 'qty': 1.0, 'side': 'LONG'})()
    signal = {'side': 'LONG', 'confidence': .9, 'features': {'price': 100.0},
              'reason_codes': ['ensemble_agrees'], 'source': 'real_models'}
    candidate = {'symbol': 'X', 'side': 'LONG', 'entry': 100.0, 'stop': 90.0, 'take_profit': 120.0}

    assert submit_entry(runtime, 'X', signal, candidate, 'bar') is None


def test_resting_order_is_cancelled_once_its_deadline_passes():
    from dataclasses import replace

    runtime = Runtime(quote(ask_qty=.1, bid_qty=.1))
    order = submit(runtime, quantity=1.0)
    assert runtime.broker.orders['p1'].status == 'OPEN'

    # Age the resting order past its deadline, the way hours of rest used to.
    runtime.broker.track_order(replace(order, expires_at=int(time.time() * 1000) - 1))
    settle_order(runtime, 'p1')

    assert runtime.broker.orders['p1'].status == 'CANCELED'
    assert runtime.account.positions == {}
    assert runtime.pending_plans == {}
    assert runtime.store.reasons() == ['entry_expired']


def test_cancel_entry_is_idempotent_for_an_already_terminal_order():
    runtime = Runtime(quote())
    order = submit(runtime, quantity=1.0)
    cancel_entry(runtime, order, 'first')
    cancel_entry(runtime, order, 'second')
    assert runtime.broker.orders['p1'].status == 'CANCELED'
    assert runtime.store.reasons() == ['first', 'second']
