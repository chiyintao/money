import pytest

from app.backtest.snapshot import lifetime_return_pct, session_accounting
from app.trading.simulation import PaperAccount, Position


class FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def trades_for_session(self, session_id, limit=500):
        return list(self._rows)

    def recent_trades(self, limit=500):
        return list(self._rows)


class FakeSession:
    session_id = "s1"

    def __init__(self, account):
        self.account = account


def test_open_position_entry_fee_counts_toward_total_fees():
    # Regression: the entry fee is deducted from cash at open, so it must appear in
    # total_fees even though no trade has closed yet. Otherwise the UI reports 0 fees
    # while the balance is already lower.
    account = PaperAccount(cash=9995.0)
    account.positions["BTCUSDT"] = Position("BTCUSDT", "LONG", 0.159, 78146.9, 77833.8, 78709.0,
                                            entry_fee=4.97)
    books = session_accounting(FakeStore([]), FakeSession(account))
    assert books["total_fees"] == 4.97
    assert len(books["session_trade_rows"]) == 0
    assert books["realized_pnl"] == 0


def test_open_position_funding_counts_toward_total_funding():
    account = PaperAccount(cash=9990.0)
    position = Position("BTCUSDT", "LONG", 0.1, 78000.0, 77000.0, 79000.0, entry_fee=3.0)
    position.funding_paid = 1.25
    account.positions["BTCUSDT"] = position
    books = session_accounting(FakeStore([]), FakeSession(account))
    assert books["total_funding"] == 1.25
    assert books["total_fees"] == 3.0


def test_closed_trade_fees_are_still_counted_once():
    account = PaperAccount(cash=10000.0)
    rows = [{"pnl": 12.5, "fees": 4.0, "funding": 0.5}]
    books = session_accounting(FakeStore(rows), FakeSession(account))
    assert books["total_fees"] == 4.0
    assert books["total_funding"] == 0.5
    assert books["realized_pnl"] == 12.5
    assert books["all_time_realized_pnl"] == 12.5


def test_closed_and_open_fees_are_summed_together():
    account = PaperAccount(cash=10000.0)
    account.positions["ETHUSDT"] = Position("ETHUSDT", "SHORT", 1.0, 2470.0, 2500.0, 2400.0,
                                            entry_fee=1.0)
    rows = [{"pnl": 5.0, "fees": 4.0, "funding": 0.0}]
    books = session_accounting(FakeStore(rows), FakeSession(account))
    assert books["total_fees"] == 5.0


def test_session_without_an_account_does_not_crash():
    # A session restored before its account exists must still produce totals.
    books = session_accounting(FakeStore([]), FakeSession(None))
    assert books["total_fees"] == 0
    assert books["total_funding"] == 0


# ------------------------------------------------- lifetime PnL across accounts
# The stored history holds sessions started against both 100 and 10,000, because the
# session form lets the operator choose. Adding their dollar PnL produced -697.29, a sum
# of quantities in different units: the largest account contributed most of the absolute
# loss while the small accounts were down a far larger fraction. A return against each
# session's own capital is the only form in which those rounds compare.
def test_lifetime_return_is_a_percentage_of_each_sessions_own_capital():
    trades = [
        {"session_id": "big", "pnl": -700.0},
        {"session_id": "small", "pnl": -7.0},
    ]
    capitals = {"big": 10_000.0, "small": 100.0}
    total, n = lifetime_return_pct(trades, capitals)
    assert n == 2
    # Both are -7%. The dollar sum says the big account was hurt a hundred times more.
    assert total == pytest.approx(-14.0)


def test_lifetime_return_is_none_without_any_known_capital():
    # No denominator means no answer, not a confident zero.
    assert lifetime_return_pct([{"session_id": "x", "pnl": -5.0}], {}) == (None, 0)


def test_trades_from_unknown_sessions_use_the_median_capital():
    # Dropping them hides real losses; counting them as dollars reintroduces the mixing.
    trades = [
        {"session_id": "known", "pnl": -100.0},
        {"session_id": "forgotten", "pnl": -100.0},
    ]
    capitals = {"known": 10_000.0, "a": 1_000.0, "b": 1_000.0}
    total, n = lifetime_return_pct(trades, capitals)
    assert n == 2
    assert total == pytest.approx(-1.0 - 10.0)      # 1,000 is the median fallback


def test_lifetime_return_handles_an_empty_history():
    assert lifetime_return_pct([], {"a": 100.0}) == (None, 0)


def test_accounting_survives_a_store_without_session_capitals():
    # A store too old to have the method must not take the snapshot down; the lifetime
    # figure degrades to unknown rather than to a wrong number.
    books = session_accounting(FakeStore([{"pnl": 3.0}]), FakeSession(None))
    assert books["lifetime_return_pct"] is None
    assert books["lifetime_trades"] == 0
