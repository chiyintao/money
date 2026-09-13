"""A position must survive a restart and still be closable.

The account snapshot is written with `asdict()`, which is recursive: it turns each Lot
inside a Position into a plain dict as well as the Position itself. `restore` rebuilt only
the outer object, so after any restart `position.lots` was a list of dicts while every
method on Position reads `lot.qty`. `take_lots` raised AttributeError on the first read,
and every exit a position could take goes through `close()` and therefore through
`take_lots` -- the manual close button, the stop, the target and the exit policy all at
once. The operator saw a close button answering HTTP 500 and a position that survived
every attempt to flatten it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.trading.simulation import Lot, PaperAccount, Position  # noqa: E402


def account_with_two_lots():
    account = PaperAccount(10000.0, 0.0004, 2.0)
    position = Position("ZECUSDT", "SHORT", 0.0, 0.0, 0.0, 0.0)
    position.add_lot(1113.42, 0.04, 0.0089, "f1", 1)
    position.add_lot(1115.00, 0.03, 0.0067, "f2", 2)
    position.stop, position.target = 1117.82, 1104.69
    account.positions["ZECUSDT"] = position
    account.marks["ZECUSDT"] = 1110.0
    return account


def test_the_snapshot_stores_lots_as_plain_dicts():
    # Not a bug on its own -- it is what asdict does -- but it is the reason restore has
    # to put them back, so it is pinned here rather than left implicit.
    snapshot = account_with_two_lots().snapshot()
    assert isinstance(snapshot["positions"][0]["lots"][0], dict)


def test_restore_rebuilds_the_lots():
    restored = PaperAccount.restore(account_with_two_lots().snapshot())
    lots = restored.positions["ZECUSDT"].lots
    assert len(lots) == 2
    assert all(isinstance(lot, Lot) for lot in lots)
    assert [round(lot.qty, 4) for lot in lots] == [0.04, 0.03]


def test_a_restored_position_can_be_closed():
    # The whole point: before the fix this raised
    # AttributeError: 'dict' object has no attribute 'qty'.
    account = PaperAccount.restore(account_with_two_lots().snapshot())
    pnl = account.close("ZECUSDT", 1110.0, "manual", 123456)
    assert isinstance(pnl, float)
    assert account.positions == {}
    assert len(account.trades) == 1


def test_a_restored_position_can_be_closed_partially():
    account = PaperAccount.restore(account_with_two_lots().snapshot())
    account.close("ZECUSDT", 1110.0, "manual", 123456, quantity=0.04)
    remaining = account.positions["ZECUSDT"]
    assert round(remaining.qty, 6) == 0.03
    assert all(isinstance(lot, Lot) for lot in remaining.lots)


def test_the_restored_position_keeps_its_geometry():
    restored = PaperAccount.restore(account_with_two_lots().snapshot())
    position = restored.positions["ZECUSDT"]
    assert position.side == "SHORT"
    assert position.stop == 1117.82
    assert position.target == 1104.69
    assert round(position.qty, 6) == 0.07
    assert round(restored.marks["ZECUSDT"], 6) == 1110.0


def test_a_position_without_lots_still_closes():
    # A snapshot written before lots existed. take_lots synthesises one from the stored
    # average, and that path has to keep working.
    account = PaperAccount(10000.0, 0.0004, 2.0)
    account.positions["BTCUSDT"] = Position("BTCUSDT", "LONG", 0.5, 100.0, 90.0, 120.0)
    restored = PaperAccount.restore(account.snapshot())
    assert restored.positions["BTCUSDT"].lots == []
    assert isinstance(restored.close("BTCUSDT", 110.0, "manual", 1), float)


def test_an_unexpected_field_in_a_snapshot_does_not_break_restore():
    # A snapshot written by a newer build must not make every stored position
    # unloadable; the known fields are taken and the rest ignored.
    snapshot = account_with_two_lots().snapshot()
    snapshot["positions"][0]["field_from_the_future"] = 1
    restored = PaperAccount.restore(snapshot)
    assert "ZECUSDT" in restored.positions
