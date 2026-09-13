"""Passive (maker) entry: the price rule, the fill rule, and the fee it pays.

Every entry used to be a market order, so every round trip paid the taker rate plus the
spread -- 12 bp against a model whose measured edge is a few basis points. The broker could
already rest and fill a limit order and already charged a resting fill the maker rate; what
was missing was any caller that submitted one. These tests pin the three things that decide
whether the option is honest: where the order rests, that it does not fill until the market
trades to it, and that a fill is charged the maker rate rather than the taker rate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.domain import OrderIntent  # noqa: E402
from app.strategy.decision_loop import passive_entry_price  # noqa: E402
from app.trading.broker import PaperBroker  # noqa: E402
from app.trading.fill_models import MAKER, TAKER  # noqa: E402


def broker():
    return PaperBroker(fee_rate=0.0004, slippage_bps=2.0, maker_fee_rate=0.0002, specs={})


def test_a_buy_joins_the_bid_and_a_sell_the_ask():
    quote = {'bid': 100.0, 'ask': 100.2}
    assert passive_entry_price(quote, 'BUY') == 100.0
    assert passive_entry_price(quote, 'SELL') == 100.2


def test_an_offset_improves_on_the_touch_without_crossing():
    quote = {'bid': 100.0, 'ask': 100.2}
    # A buy below the bid, a sell above the ask: both are still resting prices.
    assert passive_entry_price(quote, 'BUY', 5.0) < quote['bid']
    assert passive_entry_price(quote, 'SELL', 5.0) > quote['ask']


def test_an_unreadable_quote_yields_no_price():
    # Guessing a level from the mid would submit an order nothing quoted.
    assert passive_entry_price({}, 'BUY') == 0.0
    assert passive_entry_price({'bid': 0, 'ask': 0}, 'BUY') == 0.0
    assert passive_entry_price({'bid': None, 'ask': None}, 'SELL') == 0.0


def test_a_resting_limit_does_not_fill_until_the_market_reaches_it():
    b = broker()
    b.submit(OrderIntent('BTCUSDT', 'BUY', 0.1, order_type='LIMIT', limit_price=99.50,
                         order_id='l1'), timestamp=1000, plan={'price': 99.50})
    fills, status = b.process('l1', market_price=100.0, bid=99.99, ask=100.01,
                              available_qty=1.0, timestamp=2000)
    assert fills == []
    assert status == 'not_marketable'


def test_a_resting_fill_is_charged_the_maker_rate():
    b = broker()
    b.submit(OrderIntent('BTCUSDT', 'BUY', 0.1, order_type='LIMIT', limit_price=99.50,
                         order_id='l1'), timestamp=1000, plan={'price': 99.50})
    fills, status = b.process('l1', market_price=99.40, bid=99.40, ask=99.41,
                              available_qty=1.0, timestamp=3000)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.liquidity == MAKER
    # 2 bp against the 10 bp notional, which is the maker rate and not the taker rate.
    assert abs(fill.fee - fill.price * fill.quantity * 0.0002) < 1e-9


def test_a_market_order_still_crosses_and_pays_the_taker_rate():
    b = broker()
    b.submit(OrderIntent('BTCUSDT', 'BUY', 0.1, order_id='m1'), timestamp=1000,
             plan={'price': 100.0})
    fill, _ = b.execute(OrderIntent('BTCUSDT', 'BUY', 0.1, order_id='m1'), market_price=100.0,
                        bid=99.99, ask=100.01, timestamp=1000)
    assert fill.liquidity == TAKER
    assert abs(fill.fee - fill.price * fill.quantity * 0.0004) < 1e-9


def test_the_maker_route_costs_less_per_side_than_the_taker_route():
    # The whole point of the option: the fee differential plus the spread it does not pay.
    b = broker()
    maker_fee = b.fee_model.fee(100.0, MAKER)
    taker_fee = b.fee_model.fee(100.0, TAKER)
    assert maker_fee < taker_fee


def test_only_the_exact_word_limit_enables_maker_pricing():
    # The setting is read from .env at import, so asserting the ambient value would test
    # the deployment rather than the code. What matters is narrower and not environment
    # dependent: anything that is not the word limit must leave taker pricing in place,
    # because a typo that silently halved every modelled cost would be worse than a typo
    # that did nothing.
    from app.strategy.live_models import ModelDecision

    decision = ModelDecision({}, entry_order_type='LIMIT ')
    assert decision.maker_entry is True
    for value in ('market', 'Market', '', None, 'limit-order', 'maker'):
        assert ModelDecision({}, entry_order_type=value).maker_entry is False


def test_a_maker_round_trip_is_charged_the_maker_rate_on_both_legs():
    # This is the number the entry gate is measured against, so it has to follow the
    # execution mode: 2 x taker + 2 x half-spread when crossing, 2 x maker when resting.
    from app.strategy.live_models import ModelDecision

    taker = ModelDecision({}, fee_rate=0.0004, maker_fee_rate=0.0002, slippage_bps=2.0,
                          entry_order_type='market')
    assert abs(taker.round_trip_cost_pct - 0.0012) < 1e-12

    maker = ModelDecision({}, fee_rate=0.0004, maker_fee_rate=0.0002, slippage_bps=2.0,
                          entry_order_type='limit')
    assert abs(maker.round_trip_cost_pct - 0.0004) < 1e-12
    assert maker.round_trip_cost_pct < taker.round_trip_cost_pct
