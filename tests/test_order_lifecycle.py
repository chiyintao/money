"""The order lifecycle table, and the broker that has to apply it.

The table in order_state described the lifecycle and nothing consulted it: every status
change built a fresh PaperOrder with whatever status the caller passed, so an illegal move
was written to the book and to the audit trail exactly like a legal one.
"""
from app.core import order_state as os_
from app.trading.broker import ContractSpec, PaperBroker
from app.core.domain import OrderIntent
from app.ops.prometheus import collect, render


def _broker():
    return PaperBroker(specs={"BTCUSDT": ContractSpec("BTCUSDT", tick_size=0.1,
                                                      step_size=0.001, min_notional=5.0)})


def test_a_pending_order_may_fill():
    """The old table allowed PENDING -> OPEN, REJECTED, CANCELED and nothing else.

    PENDING is in OPEN_STATUSES, so the broker treats such an order as actable and
    process() fills it -- and status_for_fill returns FILLED or PARTIALLY_FILLED. A
    marketable order fills immediately, so the most common transition in the system was
    the one the table called illegal. Wiring the table in without this fix would have
    turned the first fill of every pending order into a refusal.
    """
    assert os_.status_for_fill(1.0, 1.0) == os_.FILLED
    for status in (os_.FILLED, os_.PARTIALLY_FILLED):
        assert os_.can_transition(os_.PENDING, status)
        assert os_.can_transition(os_.OPEN, status)
    assert os_.can_transition(os_.PARTIALLY_FILLED, os_.FILLED)
    # A further partial fill is the same status with a larger quantity.
    assert os_.can_transition(os_.PARTIALLY_FILLED, os_.PARTIALLY_FILLED)


def test_nothing_leaves_a_terminal_state():
    for terminal in os_.TERMINAL_STATUSES:
        assert os_.TRANSITIONS[terminal] == ()
        for target in os_.ALL_STATUSES:
            assert os_.can_transition(terminal, target) is False


def test_the_table_only_names_states_that_exist():
    for source, targets in os_.TRANSITIONS.items():
        assert source in os_.ALL_STATUSES, source
        for target in targets:
            assert target in os_.ALL_STATUSES, (source, target)
    # Every open state can end without filling; every one can be filled.
    for status in os_.OPEN_STATUSES:
        for target in (os_.CANCELED, os_.EXPIRED, os_.FILLED, os_.PARTIALLY_FILLED):
            assert os_.can_transition(status, target), (status, target)


def test_the_broker_refuses_an_illegal_transition_instead_of_writing_it():
    broker = _broker()
    order, reason = broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET",
                                              order_id="o1"), timestamp=1_000, plan={})
    assert reason == "accepted" and order.status == os_.OPEN

    # Fill it, then try to cancel: a terminal order cannot move.
    fills, status = broker.process("o1", 50_000.0, bid=49_999.0, ask=50_001.0,
                                   timestamp=2_000)
    assert status == os_.FILLED and fills
    assert broker.orders["o1"].status == os_.FILLED

    # cancel() reports the existing contract: the order is returned unchanged with the
    # reason, rather than being replaced by something in a different state.
    unchanged, why = broker.cancel("o1", timestamp=3_000)
    assert why == "order_not_open" and unchanged.status == os_.FILLED
    assert broker.orders["o1"].status == os_.FILLED, "the book was not touched"


def test_the_depth_path_and_the_quote_path_agree():
    """Both fill paths must leave the same state, through the same code.

    _fill_from_depth built the updated order itself and called track_order directly -- a
    second implementation of the lifecycle that the table could not see.
    """
    import inspect
    from app.trading import execution

    source = inspect.getsource(execution._fill_from_depth)
    assert "broker.advance(" in source
    assert "PaperOrder(" not in source and "type(order)(" not in source
    # And the broker has exactly one place that constructs an order: submit, which creates
    # it. Every move goes through order_state.transition, so a second construction here
    # would be another way around the table.
    broker_source = inspect.getsource(PaperBroker)
    assert broker_source.count("PaperOrder(") == 1


def test_a_refused_transition_is_counted_not_swallowed():
    broker = _broker()
    order, _ = broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="o2"),
                             timestamp=1_000, plan={})
    filled_order, _ = broker.advance(order, os_.FILLED, timestamp=2_000, filled_quantity=0.01)
    assert filled_order.status == os_.FILLED
    refused, why = broker.advance(filled_order, os_.OPEN, timestamp=3_000)
    assert refused is None and why == "illegal_transition:FILLED->OPEN"
    # A silent refusal is how a book drifts away from its own audit trail.
    assert broker.transition_refusals.get("FILLED->OPEN") == 1


def test_an_unknown_status_is_refused_rather_than_replaced():
    broker = _broker()
    order, _ = broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="o3"),
                             timestamp=1_000, plan={})
    # PaperOrder is frozen, so an unrecognised status is made the only way one could
    # arrive: by rebuilding the order.
    import dataclasses
    broken = dataclasses.replace(order, status="SOMETHING_ELSE")
    refused, why = broker.advance(broken, os_.FILLED, timestamp=2_000)
    assert refused is None and why == "unknown_status:SOMETHING_ELSE"
    assert broker.transition_refusals.get("SOMETHING_ELSE") == 1


def test_a_repeated_partial_fill_is_still_written():
    """Same status, larger filled quantity: refusing it would drop the fill.

    PARTIALLY_FILLED -> PARTIALLY_FILLED is the only same-status move that carries new
    information, so the guard cannot treat every same-status move as a no-op.
    """
    broker = _broker()
    order, _ = broker.submit(OrderIntent("BTCUSDT", "BUY", 0.02, "MARKET", order_id="o4"),
                             timestamp=1_000, plan={})
    first, _ = broker.advance(order, os_.PARTIALLY_FILLED, timestamp=2_000, filled_quantity=0.01)
    assert first.filled_quantity == 0.01
    second, why = broker.advance(first, os_.PARTIALLY_FILLED, timestamp=3_000, filled_quantity=0.015)
    assert why == "ok" and second.filled_quantity == 0.015
    # The same status with the same quantity is a genuine no-op and says so.
    third, why = broker.advance(second, os_.PARTIALLY_FILLED, timestamp=4_000, filled_quantity=0.015)
    assert third is second and why == "already_partially_filled"


def test_fill_rate_is_computed_over_finished_orders_only():
    broker = _broker()
    for index in range(2):
        broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="f%d" % index),
                      timestamp=1_000 + index, plan={})
    broker.process("f0", 50_000.0, bid=49_999.0, ask=50_001.0, timestamp=2_000)
    broker.cancel("f1", timestamp=2_000)
    # f2 is still working and must not count against the rate.
    broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="f2"),
                  timestamp=3_000, plan={})
    counts = broker.order_counts()
    assert counts["working"] == 1 and counts["terminal"] == 2
    assert counts["fill_rate"] == 0.5

def test_reconciliation_reports_a_refused_transition():
    """A refusal is a divergence between the venue and the book, not a statistic.

    It was neither refused nor counted: the status was written regardless, so nothing
    downstream could tell that the order had been moved somewhere the lifecycle forbids.
    """
    from app.backtest.account_reconcile import reconcile

    class Store:
        def orders(self, active_only=False):
            return []

    class Runtime:
        pass

    runtime = Runtime()
    runtime.broker = _broker()
    order, _ = runtime.broker.submit(
        OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="r1"), timestamp=1_000, plan={})
    filled, _ = runtime.broker.advance(order, os_.FILLED, timestamp=2_000, filled_quantity=0.01)
    runtime.broker.advance(filled, os_.OPEN, timestamp=3_000)

    class Account:
        positions = {}
        cash = 1000.0
        initial_cash = 1000.0
        margin_used = 0.0

    runtime.account = Account()
    runtime.store = Store()
    report = reconcile(runtime)
    assert report["order_transition_refusals"] == {"FILLED->OPEN": 1}
    assert "order_transition_refused" in report["findings"]
    assert report["consistent"] is False

    body = render(collect({"equity": 1000.0, "account": {},
                           "reconciliation": report}))
    assert 'paper_order_transition_refused{reason="FILLED->OPEN"} 1' in body


def test_a_clean_book_reports_no_refusals():
    from app.backtest.account_reconcile import reconcile

    class Store:
        def orders(self, active_only=False):
            return []

    class Runtime:
        pass

    runtime = Runtime()
    runtime.broker = _broker()

    class Account:
        positions = {}
        cash = 1000.0
        initial_cash = 1000.0
        margin_used = 0.0

    runtime.account = Account()
    runtime.store = Store()
    report = reconcile(runtime)
    assert report["order_transition_refusals"] == {}
    assert "order_transition_refused" not in report["findings"]
