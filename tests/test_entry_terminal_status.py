"""End-to-end checks that a submitted entry reaches a terminal order status.

These drive the real settle_order path with a minimal runtime so the regression that
left every order stuck in PARTIALLY_FILLED cannot come back.
"""
import time

from app.trading.broker import PaperBroker
from app.core.domain import OrderIntent
from app.trading.execution import settle_order
from app.trading.simulation import PaperAccount


class Store:
    def __init__(self):
        self.events = []

    def record_event(self, payload):
        self.events.append(payload)


class Session:
    session_id = "s-test"


class Runtime:
    def __init__(self, quote):
        self.broker = PaperBroker()
        self.account = PaperAccount(cash=10000.0)
        self.store = Store()
        self.session = Session()
        self.pending_plans = {}
        self.cache = type("Cache", (), {"latest": {"BTCUSDT": quote}})()


def fresh_quote():
    now = int(time.time() * 1000)
    return {
        "book_time": now, "mark_time": now, "bid": 78145.0, "ask": 78147.0,
        "mark_price": 78146.0, "bid_qty": 50.0, "ask_qty": 50.0,
        "bids": [[78145.0, 50.0]], "asks": [[78147.0, 50.0]],
    }


def plan_for(order_id):
    return {"symbol": "BTCUSDT", "side": "LONG", "entry": 78146.0, "stop": 77833.0,
            "take_profit": 78700.0, "order_id": order_id}


def test_rounded_entry_reaches_filled():
    runtime = Runtime(fresh_quote())
    spec = runtime.broker.spec("BTCUSDT")
    quantity = spec.round_qty(0.1599561848018561)
    order, _ = runtime.broker.submit(
        OrderIntent("BTCUSDT", "BUY", quantity, order_id="ok-1"))
    runtime.pending_plans["ok-1"] = plan_for("ok-1")

    settle_order(runtime, "ok-1")

    assert runtime.broker.orders["ok-1"].status == "FILLED"
    assert runtime.broker.orders["ok-1"].filled_quantity == quantity
    assert "BTCUSDT" in runtime.account.positions
    assert runtime.pending_plans == {}


def test_an_off_grid_entry_is_normalised_at_submit():
    # This used to hang in PARTIALLY_FILLED forever: the raw quantity filled down to the lot
    # step and the residue rounded to zero, so the order could never reach a terminal status
    # and rested until its deadline while holding a position slot and a risk reservation.
    # The size is now snapped to the grid at submit, so it fills completely.
    runtime = Runtime(fresh_quote())
    order, status = runtime.broker.submit(
        OrderIntent("BTCUSDT", "BUY", 0.1599561848018561, order_id="bug-1"))
    assert status == "accepted"
    assert order.quantity == 0.159
    # The dropped fraction is recorded rather than rounded away silently: asking for a size
    # the venue cannot express is a sizing bug, and it should be visible.
    assert order.plan["quantity_reduced_from"] == 0.1599561848018561
    runtime.pending_plans["bug-1"] = plan_for("bug-1")

    settle_order(runtime, "bug-1")
    assert runtime.broker.orders["bug-1"].status == "FILLED"
    assert runtime.broker.orders["bug-1"].filled_quantity == 0.159
    assert runtime.broker.open_orders() == []


def test_a_size_below_one_step_is_refused_rather_than_rested():
    runtime = Runtime(fresh_quote())
    spec = runtime.broker.spec("BTCUSDT")
    order, status = runtime.broker.submit(
        OrderIntent("BTCUSDT", "BUY", spec.step_size / 10, order_id="tiny-1"))
    assert order is None
    assert status == "below_step_size"
    assert runtime.broker.open_orders() == []


def test_filled_entry_leaves_no_open_orders():
    runtime = Runtime(fresh_quote())
    quantity = runtime.broker.spec("BTCUSDT").round_qty(0.05)
    runtime.broker.submit(OrderIntent("BTCUSDT", "BUY", quantity, order_id="ok-2"))
    runtime.pending_plans["ok-2"] = plan_for("ok-2")
    settle_order(runtime, "ok-2")
    assert runtime.broker.open_orders() == []
