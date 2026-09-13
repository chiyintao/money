"""Order state machine and account reconciliation tests."""
import time

from app.core import order_state as os_
from app.backtest.account_reconcile import compare_cash, compare_exposure, compare_open_orders, compare_positions, position_quantities, reconcile, signed_fills
from app.trading.broker import PaperBroker
from app.core.domain import OrderIntent


class Position:
    def __init__(self, symbol, quantity, mark_price=100.0, leverage=10.0):
        self.symbol = symbol
        self.quantity = quantity
        self.mark_price = mark_price
        self.leverage = leverage


class Account:
    def __init__(self, positions=None, cash=1000.0, initial_cash=None, margin_used=None,
                 trades=None):
        self.positions = positions or {}
        self.cash = cash
        if initial_cash is not None:
            self.initial_cash = initial_cash
        if margin_used is not None:
            self.margin_used = margin_used
        self.trades = trades or []


class Order:
    def __init__(self, status):
        self.status = status


def test_every_status_is_classified_exactly_once():
    groups = [os_.OPEN_STATUSES, os_.TERMINAL_STATUSES]
    seen = [status for group in groups for status in group]
    assert sorted(seen) == sorted(os_.ALL_STATUSES)
    assert len(seen) == len(set(seen))
    assert not set(os_.OPEN_STATUSES) & set(os_.TERMINAL_STATUSES)


def test_terminal_states_do_not_transition():
    for status in os_.TERMINAL_STATUSES:
        for target in os_.ALL_STATUSES:
            assert os_.can_transition(status, target) is False


def test_an_expiry_and_a_cancel_are_different_states():
    # The whole point: both used to be written as CANCELED, so the audit trail could not
    # say whether a deadline passed or somebody decided to stop.
    assert os_.EXPIRED != os_.CANCELED
    assert os_.outcome(os_.EXPIRED) == "expired"
    assert os_.outcome(os_.CANCELED) == "canceled"
    assert os_.outcome(os_.FILLED) == "filled"


def test_transition_refuses_an_illegal_move_without_raising():
    order = Order(os_.FILLED)
    updated, reason = os_.transition(order, os_.CANCELED)
    assert updated.status == os_.FILLED
    assert reason.startswith("illegal_transition")
    working = Order(os_.OPEN)
    assert os_.transition(working, os_.EXPIRED)[1] == "ok"
    assert working.status == os_.EXPIRED
    assert os_.transition(working, os_.OPEN)[1].startswith("illegal_transition")


def test_a_repeat_transition_is_reported_rather_than_rejected():
    # A book event arriving after the order was already resolved is a race, not a bug.
    order = Order(os_.OPEN)
    assert os_.transition(order, os_.OPEN)[1] == "already_open"


def test_a_partial_fill_is_not_silently_completed():
    assert os_.status_for_fill(1.0, 0.4) == os_.PARTIALLY_FILLED
    assert os_.status_for_fill(1.0, 1.0) == os_.FILLED
    assert os_.status_for_fill(1.0, 1.0 + 1e-15) == os_.FILLED


def test_summary_separates_working_from_finished():
    summary = os_.summarise([Order(os_.FILLED), Order(os_.EXPIRED), Order(os_.CANCELED),
                             Order(os_.OPEN)])
    assert summary["working"] == 1
    assert summary["terminal"] == 3
    assert summary["filled_or_partial"] == 1
    assert abs(summary["fill_rate"] - 1 / 3) < 1e-9
    assert summary["EXPIRED"] == 1 and summary["CANCELED"] == 1


def test_an_unknown_status_is_not_counted_as_anything():
    summary = os_.summarise([Order("WAT")])
    assert summary["working"] == 0 and summary["terminal"] == 0


def test_submit_snaps_the_quantity_to_the_contract_grid():
    broker = PaperBroker()
    order, status = broker.submit(OrderIntent("X", "BUY", 1.2345))
    assert status == "accepted"
    assert order.quantity == 1.234
    assert order.plan["quantity_reduced_from"] == 1.2345
    assert order.status == os_.OPEN


def test_submit_refuses_a_size_the_venue_cannot_express():
    broker = PaperBroker()
    spec = broker.spec("X")
    assert broker.submit(OrderIntent("X", "BUY", spec.step_size / 100))[1] == "below_step_size"
    assert broker.orders == {}


def test_submit_snaps_the_limit_price_to_the_tick_grid():
    broker = PaperBroker()
    order, _ = broker.submit(OrderIntent("X", "BUY", 1.0, order_type="LIMIT",
                                         limit_price=100.123456))
    assert order.limit_price == 100.12


def test_expiry_marks_expired_and_cancel_marks_cancelled():
    broker = PaperBroker()
    broker.submit(OrderIntent("X", "BUY", 1.0, order_id="a"), timestamp=0,
                  plan={"expires_at": 10})
    broker.submit(OrderIntent("X", "BUY", 1.0, order_id="b"), timestamp=0,
                  plan={"expires_at": 10_000})
    assert [order.order_id for order in broker.expire_orders(100)] == ["a"]
    assert broker.orders["a"].status == os_.EXPIRED
    assert broker.cancel("b")[1] == "canceled"
    assert broker.orders["b"].status == os_.CANCELED
    counts = broker.order_counts()
    assert counts["EXPIRED"] == 1 and counts["CANCELED"] == 1 and counts["working"] == 0


def test_a_terminal_order_cannot_be_cancelled_again():
    broker = PaperBroker()
    broker.submit(OrderIntent("X", "BUY", 1.0, order_id="a"), timestamp=0,
                  plan={"expires_at": 10})
    broker.expire_orders(100)
    order, reason = broker.cancel("a")
    assert reason == "order_not_open"
    assert order.status == os_.EXPIRED


def test_signed_fills_net_buys_against_sells():
    assert signed_fills([{"symbol": "X", "side": "BUY", "quantity": 2.0},
                         {"symbol": "X", "side": "SELL", "quantity": 0.5}]) == {"X": 1.5}
    assert signed_fills([{"symbol": "X", "side": "SHORT", "quantity": 3.0}]) == {"X": -3.0}
    assert signed_fills([{"side": "BUY", "quantity": 1.0}]) == {}
    assert signed_fills([]) == {}


def test_position_quantities_reads_either_shape():
    held = {"X": Position("X", 2.0)}
    assert position_quantities(Account(held)) == {"X": 2.0}
    assert position_quantities(Account({"Y": {"quantity": 3.0}})) == {"Y": 3.0}
    assert position_quantities(Account({})) == {}


def test_a_position_with_no_fills_behind_it_is_a_finding():
    # The account holds something the ledger cannot account for.
    differences = compare_positions({"X": 0.0}, {"X": 2.0})
    assert differences == [{"symbol": "X", "ledger": 0.0, "account": 2.0, "difference": 2.0}]
    assert compare_positions({"X": 2.0}, {"X": 2.0}) == []


def test_cash_is_unavailable_without_a_starting_balance():
    account = Account(cash=500.0)
    assert compare_cash(account, [])["status"] == "unavailable"


def test_cash_reconciles_realised_pnl_less_costs():
    account = Account(cash=1010.0, initial_cash=1000.0)
    fills = [{"realized_pnl": 20.0, "fees": 8.0, "funding": 2.0}]
    report = compare_cash(account, fills)
    assert report["expected_cash"] == 1010.0
    assert report["within_tolerance"] is True
    assert compare_cash(Account(cash=1200.0, initial_cash=1000.0),
                        fills)["within_tolerance"] is False


def test_open_orders_are_compared_against_storage():
    broker = PaperBroker()
    broker.submit(OrderIntent("X", "BUY", 1.0, order_id="live-1"))
    # Storage believes an order is open that the broker has never heard of.
    report = compare_open_orders(broker, [{"order_id": "ghost-1", "status": "OPEN"}])
    assert report["only_stored"] == ["ghost-1"]
    assert report["only_live"] == ["live-1"]
    assert report["consistent"] is False
    # Terminal rows in storage are not open orders and must not be reported as missing.
    settled = compare_open_orders(broker, [{"order_id": "live-1", "status": "FILLED"}])
    assert settled["only_stored"] == []
    assert settled["only_live"] == ["live-1"]


def test_a_partial_fill_difference_is_reported():
    broker = PaperBroker()
    broker.submit(OrderIntent("X", "BUY", 2.0, order_id="a"))
    report = compare_open_orders(broker, [{"order_id": "a", "status": "OPEN",
                                           "filled_quantity": 1.0}])
    assert report["quantity_mismatch"] == [{"order_id": "a", "broker_filled": 0.0,
                                            "stored_filled": 1.0}]


def test_margin_is_compared_against_the_positions_it_covers():
    account = Account({"X": Position("X", 2.0, mark_price=100.0, leverage=10.0)},
                      margin_used=20.0)
    assert compare_exposure(account)["within_tolerance"] is True
    wrong = Account({"X": Position("X", 2.0, mark_price=100.0, leverage=10.0)},
                    margin_used=99.0)
    assert compare_exposure(wrong)["within_tolerance"] is False


def test_exposure_skips_a_position_with_no_price():
    account = Account({"X": Position("X", 2.0, mark_price=0.0)}, margin_used=5.0)
    report = compare_exposure(account)
    assert report["positions_covered"] == 0
    assert report["positions_held"] == 1


def test_an_account_with_no_margin_field_says_so():
    class Bare:
        positions = {}

    assert compare_exposure(Bare())["status"] == "unavailable"


def test_reconcile_reports_a_consistent_account_as_clean():
    class Store:
        def orders(self, active_only=False):
            return []

    class Runtime:
        pass

    runtime = Runtime()
    runtime.broker = PaperBroker()
    runtime.account = Account({}, cash=1000.0, initial_cash=1000.0, margin_used=0.0)
    runtime.store = Store()
    report = reconcile(runtime)
    assert report["consistent"] is True
    assert report["findings"] == []
    assert report["order_outcomes"]["working"] == 0


def test_reconcile_never_raises_on_a_broken_runtime():
    # It runs inside the decision loop; a monitoring step that can stop trading is worse
    # than the discrepancy it was watching for.
    report = reconcile(object())
    assert report["consistent"] is False
    assert "reconcile_failed" in report["findings"]
    assert "error" in report
