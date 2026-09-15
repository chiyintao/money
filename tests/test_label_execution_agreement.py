"""The label and the trade must resolve an ambiguous bar the same way.

A five-minute bar carries four prices. When its high clears the target and its low clears the
stop, the data cannot say which happened first -- the true path between them is unobserved.
That makes the tie-break a modelling decision rather than a fact, and the two places that
make it have to agree, because one of them trains the model and the other one runs the money.

They did not agree:

    app/models/labels.py    tested the upper barrier first  -> the row is a WINNER
    app/trading/execution_rules.py  took the stop           -> the trade is a LOSER

Every bar that spanned both levels was therefore a winner in the training set and a loser in
the account. The model was not merely noisy about these rows, it was taught the opposite of
what the broker does, and it learned to expect a fill it can never get.

How big is it? Measured by re-labelling 30k real BTCUSDT bars under both policies: 5 rows in
5998 flip, 0.08%, and the reported win rate moves from 32.29% to 32.13%. That is small, and
it is stated here rather than left to be imagined, because the first estimate of this was
wrong by two orders of magnitude.

It is not uniformly small, though. The rate tracks the barrier width against bar volatility:
at the live geometry the barriers span 78 bp and a 5-minute bar rarely covers that, so few
rows are ambiguous; at a 10 bp/16 bp barrier -- the regime a short-stop policy operates near
-- the same measurement gives 3.23%, forty times higher. The rule matters most exactly where
a tight-stop strategy lives.

The default now follows execution, and the optimistic reading is opt-in so the bias can be
measured rather than suffered.
"""
import pytest

from app.models.labels import (
    AMBIGUOUS_STOP,
    AMBIGUOUS_TARGET,
    LOWER,
    UPPER,
    triple_barrier,
)
from app.trading.execution_rules import evaluate_bar_exit


def bars_with(high, low, close=None):
    return [
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
         "open_time": 0, "close_time": 1},
        {"open": 100.0, "high": high, "low": low,
         "close": close if close is not None else 100.0,
         "open_time": 1, "close_time": 2},
    ]


class Position:
    side = "LONG"
    stop = 97.0
    target = 103.0


def test_a_bar_spanning_both_barriers_is_labelled_the_way_it_is_traded():
    """The regression. upper 103, lower 97, one bar through both."""
    outcome = triple_barrier(bars_with(104.0, 96.0), 0, 3.0, 3.0, 1)
    traded = evaluate_bar_exit(Position(), {"high": 104.0, "low": 96.0})
    assert outcome["barrier"] == LOWER
    assert traded["reason"] == "stop_loss"
    assert outcome["future_return"] < 0


def test_the_ambiguous_row_is_marked_ambiguous():
    outcome = triple_barrier(bars_with(104.0, 96.0), 0, 3.0, 3.0, 1)
    assert outcome["ambiguous"] is True


def test_the_optimistic_reading_is_available_but_not_default():
    outcome = triple_barrier(bars_with(104.0, 96.0), 0, 3.0, 3.0, 1,
                             ambiguous_policy=AMBIGUOUS_TARGET)
    assert outcome["barrier"] == UPPER
    assert outcome["ambiguous"] is True


def test_the_default_policy_is_the_one_that_matches_execution():
    assert AMBIGUOUS_STOP == "stop"
    assert AMBIGUOUS_STOP != AMBIGUOUS_TARGET


def test_a_target_only_touch_is_unchanged():
    outcome = triple_barrier(bars_with(103.5, 99.8), 0, 3.0, 3.0, 1)
    assert outcome["barrier"] == UPPER
    assert outcome["ambiguous"] is False
    assert evaluate_bar_exit(Position(), {"high": 103.5, "low": 99.8})["reason"] == "take_profit"


def test_a_stop_only_touch_is_unchanged():
    outcome = triple_barrier(bars_with(100.2, 96.5), 0, 3.0, 3.0, 1)
    assert outcome["barrier"] == LOWER
    assert outcome["ambiguous"] is False
    assert evaluate_bar_exit(Position(), {"high": 100.2, "low": 96.5})["reason"] == "stop_loss"


def test_a_bar_that_touches_nothing_keeps_walking():
    bars = bars_with(100.4, 99.6) + bars_with(103.5, 99.8)[1:]
    outcome = triple_barrier(bars, 0, 3.0, 3.0, 3)
    assert outcome["barrier"] == UPPER
    assert outcome["bars_held"] == 2


def test_the_vertical_barrier_is_still_the_fallback():
    outcome = triple_barrier(bars_with(100.4, 99.6), 0, 3.0, 3.0, 1)
    assert outcome["barrier"] == "vertical"
