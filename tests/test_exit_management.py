"""Exit management must not disable itself the first time it acts.

`manage_position` read the risk unit off the position CURRENT stop. Breakeven sets that
stop to the entry, so on the very next evaluation the risk was zero, the function returned
None, and none of the rules below it ever ran again: no trailing stop, no further
breakeven, nothing. A trade that had been moved to breakeven was frozen there for the rest
of its life, which is why the dashboard showed a stop price identical to the entry price
on trade after trade and never once showed a trailing stop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.trading.exit_policy import PRESETS, manage_position  # noqa: E402
from app.trading.simulation import Position  # noqa: E402


def position(side="LONG", entry=100.0, stop=98.0, target=104.0):
    p = Position("TESTUSDT", side, 1.0, entry, stop, target)
    p.initial_stop = stop
    p.initial_target = target
    return p


def test_breakeven_moves_the_stop_to_the_entry():
    policy = PRESETS["scalp"]
    p = position()
    # 0.6R of a 2.0 risk unit is 1.2, so 101.2 is the first price that qualifies. atr=0
    # keeps the trailing rule out of the way so this asserts breakeven alone; with an ATR
    # the trailing stop is tighter and wins, which the next test covers.
    out = manage_position(policy, p, 101.3, atr=0.0)
    assert out is not None
    assert out["stop"] == 100.0
    assert "breakeven" in out["reason"]


def test_a_position_already_at_breakeven_still_trails():
    # The regression. Before the fix this returned None forever, because risk was
    # measured from the current stop and the current stop was the entry.
    policy = PRESETS["scalp"]
    p = position()
    p.stop = 100.0  # already moved to breakeven by an earlier evaluation
    out = manage_position(policy, p, 103.0, atr=1.0)
    assert out is not None, "a breakeven position must still be managed"
    assert out["stop"] > 100.0, "the trailing stop should have moved above breakeven"


def test_the_risk_unit_is_the_planned_one_not_the_current_stop():
    # R is a property of the plan, so two positions opened on the same plan must agree
    # about how far 0.6R is regardless of where their stops have since moved to. This is
    # what makes breakeven fire at the same price for both instead of never again.
    policy = PRESETS["scalp"]
    fresh = position(stop=98.0)
    half = position(stop=98.0)
    half.stop = 99.5  # a stop that has already been tightened, but not to breakeven
    a = manage_position(policy, fresh, 101.3, atr=0.0)
    b = manage_position(policy, half, 101.3, atr=0.0)
    assert a is not None and b is not None
    # Both reach the same 0.6R threshold and both end with the stop at the entry.
    assert a["stop"] == 100.0
    assert b["stop"] == 100.0


def test_risk_is_reconstructed_from_the_target_when_no_initial_stop_was_recorded():
    # A snapshot written before the field existed carries initial_stop 0.0. The planned
    # risk is still recoverable: the target sits target_rr risk units from the entry.
    policy = PRESETS["scalp"]
    p = position(entry=100.0, stop=98.0, target=103.2)  # 2.0 risk x 1.6 = 3.2
    p.initial_stop = 0.0
    # 0.6R of 2.0 is 1.2, so 101.2 triggers breakeven.
    out = manage_position(policy, p, 101.3, atr=0.0)
    assert out is not None
    assert out["stop"] == 100.0


def test_the_reconstruction_uses_the_policy_ratio_not_the_stored_stop():
    # With the stop already at the entry there is nothing left to measure from, so the
    # target is the only remaining record of how wide the trade was planned to be.
    policy = PRESETS["scalp"]
    p = position(entry=100.0, stop=98.0, target=103.2)
    p.initial_stop = 0.0
    p.stop = 100.0  # moved to breakeven before the field existed
    # Breakeven is already satisfied and would be a no-op, so the only rule that can move
    # the stop is the trailing one -- and it needs an ATR to compute a distance from.
    out = manage_position(policy, p, 101.3, atr=1.0)
    assert out is not None, "a legacy breakeven position must still be trailed"
    assert out["reason_codes"] == ["trail"]
    assert out["stop"] == 100.1  # 101.3 - 1.2 x 1.0 ATR


def test_the_stop_never_moves_backwards():
    policy = PRESETS["scalp"]
    p = position()
    p.stop = 101.0  # already far above the entry, from trailing
    p.initial_stop = 98.0
    out = manage_position(policy, p, 101.1, atr=1.0)
    if out is not None:
        assert out["stop"] >= 101.0, "a stop may only ever tighten"


def test_a_time_stop_still_fires_once_the_stop_is_at_breakeven():
    policy = PRESETS["scalp"]
    p = position()
    p.stop = 100.0
    out = manage_position(policy, p, 100.0, atr=1.0, bars_held=policy.time_stop_bars)
    assert out is not None
    assert out.get("exit_now") is True
    assert out["reason"] == "time_stop"
