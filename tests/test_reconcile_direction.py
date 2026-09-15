"""A calibration check must not silently rewrite its own input.

\`account_reconcile\` compares what the account believes holds against what the order ledger
says was filled. Both readers resolved a fill's direction with a two-branch expression whose
else branch meant "sell":

    direction = 1.0 if side in ('BUY', 'LONG') else -1.0

That is not a default, it is a reversal. The stored fills are exactly the shape that
triggers it -- 188 \`order_filled\` events carry \`"side": null\` -- so every quantity the ledger
totalled was negated, and the reconciliation reported \`position_order_mismatch\` on positions
that were correct. Those findings fired every five minutes for as long as anyone looked and
were never actionable, which is worse than no check at all: a permanent alarm trains its
reader to ignore the one that matters.

The fix is to return None for an unreadable side and let the caller report that as the
finding, so a missing field produces "the side is missing" rather than a wrong number.
"""
import pytest

from app.ops.account_reconcile import (
    _direction,
    order_position_mismatches,
    signed_fills,
)


class FakeOrder:
    def __init__(self, symbol, side, filled):
        self.symbol = symbol
        self.side = side
        self.filled_quantity = filled


class FakeAccount:
    def __init__(self, positions):
        self.positions = positions


class FakePosition:
    def __init__(self, qty):
        self.qty = qty


# --- _direction refuses to guess -------------------------------------------------------

def test_buy_sides_are_positive():
    for side in ('BUY', 'buy', 'LONG', 'long', ' Buy '):
        assert _direction(side) == 1.0


def test_sell_sides_are_negative():
    for side in ('SELL', 'sell', 'SHORT', 'short'):
        assert _direction(side) == -1.0


def test_a_missing_side_is_not_a_sell():
    """The regression: null used to return -1.0 and invert the whole ledger."""
    for side in (None, '', '  ', 'UNKNOWN'):
        assert _direction(side) is None


# --- signed_fills ----------------------------------------------------------------------

def test_a_buy_adds_and_a_sell_subtracts():
    fills = [{'symbol': 'X', 'side': 'BUY', 'quantity': 3},
             {'symbol': 'X', 'side': 'SELL', 'quantity': 1}]
    assert signed_fills(fills)['X'] == 2.0


def test_a_fill_with_no_side_is_skipped_not_negated():
    fills = [{'symbol': 'X', 'side': 'BUY', 'quantity': 5},
             {'symbol': 'X', 'side': None, 'quantity': 5}]
    # The null fill contributes nothing. Before the fix this returned 0.0 -- the buy and
    # the phantom sell cancelled exactly, which is why every position looked wrong.
    assert signed_fills(fills)['X'] == 5.0


def test_a_lone_null_side_fill_does_not_create_a_negative_position():
    assert signed_fills([{'symbol': 'X', 'side': None, 'quantity': 9}]) == {}


# --- order_position_mismatches ---------------------------------------------------------

def test_a_correct_position_is_not_reported_as_a_mismatch():
    account = FakeAccount({'X': FakePosition(10.0)})
    orders = [FakeOrder('X', 'BUY', 10.0)]
    assert order_position_mismatches(account, orders) == []


def test_a_genuine_mismatch_is_still_reported():
    account = FakeAccount({'X': FakePosition(586.0)})
    orders = [FakeOrder('X', 'BUY', 26.0)]
    found = order_position_mismatches(account, orders)
    assert len(found) == 1
    assert found[0]['symbol'] == 'X'
    assert found[0]['difference'] == pytest.approx(560.0)


def test_an_order_with_no_side_is_reported_as_such():
    account = FakeAccount({'X': FakePosition(10.0)})
    orders = [FakeOrder('X', None, 10.0)]
    found = order_position_mismatches(account, orders)
    assert len(found) == 1
    # Named for what it is. Reporting a number here would be inventing one.
    assert found[0]['reason'] == 'order_side_missing'
    assert found[0]['from_orders'] is None


def test_the_real_ena_shape_is_no_longer_a_phantom_mismatch():
    """586 held against a 26 buy: a real difference, and it still reports one."""
    account = FakeAccount({'ENAUSDT': FakePosition(586.0)})
    orders = [FakeOrder('ENAUSDT', 'BUY', 612.0), FakeOrder('ENAUSDT', 'BUY', 26.0)]
    found = order_position_mismatches(account, orders)
    assert found and found[0]['symbol'] == 'ENAUSDT'
    assert found[0]['from_orders'] == 638.0


def test_a_sell_ledger_against_a_short_position_agrees():
    account = FakeAccount({'X': FakePosition(-10.0)})
    orders = [FakeOrder('X', 'SELL', 10.0)]
    assert order_position_mismatches(account, orders) == []
