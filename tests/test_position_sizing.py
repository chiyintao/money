"""Position sizing: what decides the size of an order.

The stored sessions sized every entry at roughly 11 USDT of notional on a 100 USDT
account, because sizing reduced to "risk max_risk of equity over the stop distance"
and the stops on those symbols were around 1.8% wide. Everything here pins down the
constraints that replaced it and, just as importantly, which one is reported as having
bound.
"""
import pytest

from app.trading.risk import RiskEngine


def engine(**overrides):
    values = {"max_risk": .005, "max_portfolio_risk": .02, "target_exposure": 1.0,
              "max_symbol_leverage": 3.0, "max_gross_leverage": 5.0, "day_start_equity": 100.0}
    values.update(overrides)
    return RiskEngine(**values)


def plan(entry=100.0, stop=98.0, side='LONG'):
    return {'side': side, 'entry': entry, 'stop': stop}


def test_risk_cap_alone_decides_a_tight_stop():
    sized = engine().size(plan(), equity=100.0)
    assert sized['binding'] == 'risk'
    assert sized['risk_cash'] == pytest.approx(.5)
    assert sized['notional'] == pytest.approx(25.0)


def test_target_exposure_decides_when_risk_would_be_generous():
    # A 5% per-trade risk on a 2% stop would size 250 of notional; the exposure target
    # is what stops it from deploying that much by accident.
    sized = engine(max_risk=.05, target_exposure=.2).size(plan(), equity=100.0)
    assert sized['binding'] == 'target'
    assert sized['notional'] == pytest.approx(20.0)


def test_notional_room_decides_when_the_caps_bind():
    sized = engine(max_symbol_leverage=.1, max_gross_leverage=.1).size(plan(), equity=100.0)
    assert sized['binding'] == 'room'
    assert sized['notional'] == pytest.approx(10.0)


def test_zero_target_exposure_disables_the_target_instead_of_blocking():
    sized = engine(target_exposure=0).size(plan(), equity=100.0)
    assert sized['binding'] == 'risk'
    assert sized['quantity'] > 0


def test_a_wide_stop_still_deploys_real_capital():
    # The stored case: a 1.8% stop on a 100 account. Risk-based sizing alone gave about
    # 11 of notional at max_risk 0.002; the current default deploys roughly 2.5x that,
    # and the exposure target shows what the risk cap is holding back.
    sized = engine().size(plan(entry=100.0, stop=98.2), equity=100.0)
    assert sized['notional'] == pytest.approx(.5 / .018)
    assert sized['notional'] > 11.0
    assert sized['target_notional'] == pytest.approx(100.0)


def test_portfolio_budget_shrinks_the_next_entry():
    sized_solo = engine().size(plan(), equity=100.0, open_risk=0.0)
    sized_shared = engine().size(plan(), equity=100.0, open_risk=1.8)
    assert sized_solo['risk_cash'] == pytest.approx(.5)
    assert sized_shared['risk_cash'] == pytest.approx(.2)
    assert sized_shared['notional'] < sized_solo['notional']


def test_exhausted_portfolio_budget_refuses_the_entry():
    decision = engine().approve(plan(), 100.0, 0, open_risk=2.0)
    assert decision['approved'] is False
    assert decision['reason'] == 'risk_budget_exhausted'


def test_budget_is_capped_by_the_per_trade_risk_not_the_portfolio_total():
    # Room to spare in the portfolio budget must not license a larger single trade.
    sized = engine(max_portfolio_risk=.10).size(plan(), equity=100.0, open_risk=0.0)
    assert sized['risk_cash'] == pytest.approx(.5)


def test_no_room_refuses_with_a_notional_reason():
    # Open notional already at the gross cap, so there is nothing left to allocate.
    decision = engine().approve(plan(), 100.0, open_notional=500.0)
    assert decision['approved'] is False
    assert decision['reason'] == 'notional_limit'


def test_a_partial_room_shrinks_the_entry_and_says_so():
    decision = engine(max_symbol_leverage=.1).approve(plan(), 100.0, open_notional=0.0)
    assert decision['approved'] is True
    assert decision['binding'] == 'room'
    assert decision['notional'] == pytest.approx(10.0)


def test_approve_reports_the_binding_constraint_with_the_size():
    decision = engine().approve(plan(), 100.0, 0)
    assert decision['approved'] is True
    assert decision['binding'] == 'risk'
    assert decision['quantity'] == pytest.approx(.25)
    assert decision['notional'] == pytest.approx(25.0)
    assert decision['risk_cash'] == pytest.approx(.5)


def test_daily_breaker_still_overrides_sizing():
    risk = engine()
    assert risk.approve(plan(), 100.0, 0)['approved'] is True
    risk.day_start_equity = 100.0
    blocked = risk.approve(plan(), 79.0, 0)
    assert blocked['approved'] is False
    assert blocked['reason'] == 'daily_loss_circuit_breaker'


def test_sizing_state_round_trips_through_snapshot():
    risk = engine(max_risk=.011, target_exposure=.7, max_portfolio_risk=.033)
    restored = RiskEngine.restore(risk.snapshot(), 100.0)
    assert restored.max_risk == pytest.approx(.011)
    assert restored.target_exposure == pytest.approx(.7)
    assert restored.max_portfolio_risk == pytest.approx(.033)


def test_plan_without_usable_brackets_is_rejected():
    assert engine().size({'side': 'LONG', 'entry': 100.0, 'stop': 100.0}, 100.0) is None
    assert engine().approve({'side': 'LONG', 'entry': 100.0, 'stop': 100.0}, 100.0)['reason'] == 'invalid_plan'
