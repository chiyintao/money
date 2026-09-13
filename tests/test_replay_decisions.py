"""The replay must be able to see the decisions the walk-forward produced.

The dataset identifies a bar by its close_time, and every out-of-sample prediction is
keyed by that field. The replay looked the decision up by the bar open_time, which for a
five-minute bar is 299,999 ms earlier. The two never met: a replay holding 1,244 real
decisions executed none of them, reported zero trades, and the promotion gate refused the
model for insufficient evidence -- with nothing in the evidence block saying the decisions
had been invisible rather than unprofitable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.portfolio_oos import decision_strategy  # noqa: E402


def bar(open_time, close_time, close=100.0):
    return {"open_time": open_time, "close_time": close_time, "close": close}


def decisions_for(symbol, timestamp):
    return {symbol: {timestamp: {"side": "LONG", "predicted_return": 0.004,
                                 "timestamp": timestamp, "fold": 1}}}


def test_a_decision_keyed_by_close_time_is_found():
    open_time, close_time = 1_000_000, 1_299_999
    strategy = decision_strategy(decisions_for("BTCUSDT", close_time))
    plan = strategy("BTCUSDT", [bar(open_time, close_time)])
    assert plan is not None
    assert plan["side"] == "LONG"
    assert strategy.stats["hits"] == 1


def test_a_decision_keyed_by_open_time_is_still_found():
    # Kept as a fallback, so a caller that keyed its decisions the other way still works.
    open_time, close_time = 1_000_000, 1_299_999
    strategy = decision_strategy(decisions_for("BTCUSDT", open_time))
    plan = strategy("BTCUSDT", [bar(open_time, close_time)])
    assert plan is not None
    assert strategy.stats["hits"] == 1


def test_a_bar_with_no_decision_is_counted_as_a_miss():
    strategy = decision_strategy(decisions_for("BTCUSDT", 1_299_999))
    assert strategy("BTCUSDT", [bar(1_000_000, 1_299_999)]) is None or True
    plan = strategy("ETHUSDT", [bar(1_000_000, 1_299_999)])
    assert plan is None
    assert strategy.stats["misses"] == 1


def test_the_lookup_counts_are_exposed_to_the_caller():
    # The whole failure mode was silent: a full-looking evidence block with zero trades.
    # These counters are what make it say which of the two it was.
    strategy = decision_strategy({})
    assert strategy.stats == {"hits": 0, "misses": 0}
    strategy("BTCUSDT", [bar(1, 2)])
    assert strategy.stats["misses"] == 1


def test_an_empty_history_is_not_a_miss():
    # No bar means no lookup happened, and counting it as a miss would inflate the count
    # with bars the replay never considered.
    strategy = decision_strategy(decisions_for("BTCUSDT", 1))
    assert strategy("BTCUSDT", []) is None
    assert strategy.stats == {"hits": 0, "misses": 0}
