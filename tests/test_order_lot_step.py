from app.trading.broker import PaperBroker
from app.core.domain import OrderIntent


def test_unrounded_quantity_leaves_an_untradable_residue():
    # Regression: the risk engine sizes by risk, not by lot step. A fill for such a
    # quantity is rounded down, so the order is left PARTIALLY_FILLED, and the residue
    # itself rounds to zero -- it can never fill. The order stayed open forever.
    broker = PaperBroker()
    spec = broker.spec("BTCUSDT")
    desired = 0.1599561848018561
    depth = [(78146.9, 10.0)]

    fill, reason = broker.execute_depth(OrderIntent("BTCUSDT", "BUY", desired, order_id="o1"),
                                        asks=depth)
    assert reason == "filled"
    assert fill.quantity == 0.159
    assert fill.quantity < desired

    residual = desired - fill.quantity
    assert residual > 0
    assert spec.round_qty(residual) == 0.0

    late, late_reason = broker.execute_depth(
        OrderIntent("BTCUSDT", "BUY", residual, order_id="o1"), asks=depth)
    assert late is None
    assert late_reason in ("below_min_notional", "invalid_quantity")


def test_quantity_rounded_up_front_fills_completely():
    # The fix: round to the lot step before submitting, so the fill equals the order
    # quantity and the order reaches FILLED instead of hanging in PARTIALLY_FILLED.
    broker = PaperBroker()
    spec = broker.spec("BTCUSDT")
    quantity = spec.round_qty(0.1599561848018561)
    assert quantity == 0.159

    fill, reason = broker.execute_depth(OrderIntent("BTCUSDT", "BUY", quantity, order_id="o2"),
                                        asks=[(78146.9, 10.0)])
    assert reason == "filled"
    assert fill.quantity == quantity


def test_rounded_quantity_respects_the_lot_step_for_any_symbol():
    broker = PaperBroker()
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        spec = broker.spec(symbol)
        quantity = spec.round_qty(0.1599561848018561)
        assert abs(quantity / spec.step_size - round(quantity / spec.step_size)) < 1e-9
