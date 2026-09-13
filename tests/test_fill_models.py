"""The matching layer: fees by liquidity, and a fill price that knows the order size.

Two assumptions in execute() were wrong and each was invisible while they shared one
function. Slippage was a fixed basis-point count, so a 50 USDT order and a 500,000,000
USDT order filled at the same price; and Fill.liquidity was hardcoded to taker, so a
resting limit order was charged the taker rate and recorded as having crossed the spread.
"""
from app.trading.broker import ContractSpec, PaperBroker
from app.core.domain import OrderIntent
from app.trading.fill_models import MAKER, TAKER, FeeModel, FillModel, participation

SPEC = ContractSpec("BTCUSDT", tick_size=0.1, step_size=0.001, min_notional=5.0)


def _broker(**kwargs):
    return PaperBroker(specs={"BTCUSDT": SPEC}, **kwargs)


def test_the_fee_depends_on_the_liquidity_the_fill_took():
    fee = FeeModel(maker_rate=0.0002, taker_rate=0.0004)
    assert fee.fee(1000.0, MAKER) == 0.2
    assert fee.fee(1000.0, TAKER) == 0.4
    # An unlabelled liquidity class is charged as a taker rather than as a maker.
    assert fee.fee(1000.0, None) == 0.4
    assert fee.fee(1000.0, "something_new") == 0.4
    assert fee.rate(MAKER) == 0.0002 and fee.rate(TAKER) == 0.0004


def test_a_resting_limit_fill_is_a_maker_fill():
    broker = _broker()
    broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "LIMIT", limit_price=50000.0,
                              order_id="L1"), timestamp=1000, plan={})
    fills, status = broker.process("L1", 49999.0, bid=49998.0, ask=49999.0, timestamp=2000)
    fill = fills[0]
    assert fill.liquidity == MAKER
    assert abs(fill.fee - fill.price * fill.quantity * 0.0002) < 1e-12
    # Half the taker fee: the old code charged the taker rate on every fill.
    assert abs(fill.fee - fill.price * fill.quantity * 0.0004) > 1e-9
    # A maker fill is not crossed: it fills at the touch, not through it.
    assert fill.price == 49999.0


def test_a_marketable_limit_order_is_a_taker_fill():
    """Only knowable when the caller says the order never rested."""
    broker = _broker()
    broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "LIMIT", limit_price=50000.0,
                              order_id="L2"), timestamp=1000, plan={})
    fills, _ = broker.process("L2", 49999.0, bid=49998.0, ask=49999.0, timestamp=1000)
    fill = fills[0]
    assert fill.liquidity == TAKER
    assert abs(fill.fee - fill.price * fill.quantity * 0.0004) < 1e-12


def test_a_market_order_is_a_taker_fill():
    broker = _broker()
    order, _ = broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="M1"),
                             timestamp=1000, plan={})
    fills, _ = broker.process("M1", 50000.0, bid=49999.0, ask=50001.0, timestamp=2000)
    assert fills[0].liquidity == TAKER


def test_the_fill_label_travels_with_the_fee_that_was_charged():
    """A record whose liquidity says taker while its fee says maker is worse than either.

    The label was computed, used to pick the rate, and then dropped at the Fill
    construction, so every fill in the audit trail read taker while half of them had been
    charged the maker rate.
    """
    broker = _broker()
    broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "LIMIT", limit_price=50000.0,
                              order_id="L3"), timestamp=1_000, plan={})
    maker_fills, _ = broker.process("L3", 49999.0, bid=49998.0, ask=49999.0, timestamp=2_000)
    broker.submit(OrderIntent("BTCUSDT", "BUY", 0.01, "MARKET", order_id="M3"),
                  timestamp=3_000, plan={})
    taker_fills, _ = broker.process("M3", 50000.0, bid=49999.0, ask=50001.0, timestamp=4_000)

    maker, taker = maker_fills[0], taker_fills[0]
    assert maker.liquidity == MAKER and taker.liquidity == TAKER
    # Each label must match the rate its own fee was computed from, or the audit trail
    # contradicts the number next to it.
    assert abs(maker.fee - maker.price * maker.quantity * 0.0002) < 1e-12
    assert abs(taker.fee - taker.price * taker.quantity * 0.0004) < 1e-12
    assert maker.fee < taker.fee


def test_the_fill_price_responds_to_the_order_size():
    """The defect: size did not enter the price at all.

    Measured on the old code, 0.001 and 10,000 BTCUSDT both filled at 50,011.00 against a
    50,001.00 ask -- a 10.00 slippage each -- on notionals of 50 and 500,110,000 USDT.
    """
    model = FillModel(spread_bps=2.0, impact_coefficient_bps=10.0)
    small = model.price(50_001.0, "BUY", TAKER, participation=0.0001)
    large = model.price(50_001.0, "BUY", TAKER, participation=1.0)
    assert large > small, "a larger share of the touch must fill worse"
    assert abs(model.slip_bps(TAKER, participation=1.0) - 12.0) < 1e-9
    # Square root: a quarter of the touch costs half the coefficient.
    assert abs(model.slip_bps(TAKER, participation=0.25) - 7.0) < 1e-9


def test_an_unknown_size_is_not_treated_as_a_small_one():
    model = FillModel(spread_bps=2.0, impact_coefficient_bps=10.0)
    # No participation and no reference notional: the model charges the coefficient in
    # full rather than assuming the order was tiny.
    assert model.impact_bps() == 10.0
    assert model.slip_bps(TAKER) == 12.0
    assert model.impact_bps(participation=0.0) == 0.0
    # A reference notional is the fallback when participation is unavailable but size is not.
    scaled = FillModel(spread_bps=2.0, impact_coefficient_bps=10.0, reference_notional=100.0)
    assert abs(scaled.impact_bps(notional=100.0) - 10.0) < 1e-9
    assert abs(scaled.impact_bps(notional=25.0) - 5.0) < 1e-9


def test_a_maker_fill_pays_neither_spread_nor_impact():
    model = FillModel(spread_bps=2.0, impact_coefficient_bps=10.0)
    assert model.slip_bps(MAKER, participation=1.0) == 0.0
    assert model.price(100.0, "BUY", MAKER, participation=1.0) == 100.0
    # A sell is worse in the other direction.
    assert model.price(100.0, "SELL", TAKER) < 100.0
    assert model.price(100.0, "BUY", TAKER) > 100.0


def test_zero_impact_keeps_the_previous_behaviour():
    """The knob has to be able to reproduce the old numbers exactly."""
    broker = _broker()
    fill, reason = broker.execute(OrderIntent("BTCUSDT", "BUY", 1.0, "MARKET"), 50_000.0,
                                  bid=49_999.0, ask=50_001.0, timestamp=1_000)
    assert reason == "filled"
    assert fill.price == 50_011.0, "ask plus the 2 bp spread, as before"
    assert abs(fill.fee - 50_011.0 * 1.0 * 0.0004) < 1e-9


def test_the_broker_mirrors_the_models_it_built():
    broker = _broker(fee_rate=0.001, maker_fee_rate=0.0005, slippage_bps=7.0)
    assert broker.fee_model.taker_rate == 0.001
    assert broker.fee_model.maker_rate == 0.0005
    assert broker.fill_model.spread_bps == 7.0
    # fee_rate and slippage_bps are read by callers that price a round trip; they must
    # report what the models actually charge, not what was passed in and then overwritten.
    assert broker.fee_rate == 0.001 and broker.slippage_bps == 7.0
    described = broker.cost_model()
    assert described["fee"]["taker_bps"] == 10.0 and described["fill"]["spread_bps"] == 7.0


def test_participation_is_unmeasurable_rather_than_zero():
    assert participation(0.5, 2.0) == 0.25
    # A zero would read as "this order consumed none of the book", which is the optimistic
    # answer. None reads as "not measured", which is the true one.
    assert participation(1.0, 0) is None
    assert participation(1.0, None) is None
    assert participation(0, 5.0) is None


def test_the_depth_path_reports_the_share_of_the_ladder_it_consumed():
    broker = _broker()
    fill, reason = broker.execute_depth(OrderIntent("BTCUSDT", "BUY", 3.0),
                                        asks=[(100.0, 1.0), (102.0, 5.0)])
    assert reason == "filled"
    assert fill.liquidity == TAKER
    assert abs(fill.participation - 0.5) < 1e-9, "3 of the 6 on offer"
    assert abs(fill.fee - fill.price * fill.quantity * 0.0004) < 1e-9
