"""The daily trade budget: the one cap that counts entries instead of sizing them.

Every other limit in the risk engine answers "how much may this trade lose". None of them
answers "how many times may this fire". The entry gate has produced 37 round trips in
under ten seconds on a single symbol at a net loss, and the daily loss breaker is
deliberately forgiving at the date boundary, so a strategy that takes many small losing
trades is invisible to every cap the engine had.
"""
from app.ops.prometheus import collect, render
from app.trading.risk import RiskEngine

# 2026-01-01T00:00:00Z. Event times, because the live path stamps the day from the
# exchange timestamp rather than from the wall clock.
DAY = 1_767_225_600_000
DAY_KEY = "2026-01-01"


def _plan(entry=100.0, stop=99.0, side="LONG"):
    return {"side": side, "entry": entry, "stop": stop, "take_profit": entry + 2.0,
            "symbol": "BTCUSDT"}


def _engine(limit=3, equity=10_000.0):
    engine = RiskEngine(day_start_equity=equity, max_risk=0.005, max_daily_loss=0.99,
                        max_gross_leverage=5.0, max_symbol_leverage=3.0,
                        daily_trade_limit=limit)
    engine.day_key = DAY_KEY
    engine.trades_today = 0
    return engine


def test_the_cap_refuses_only_after_the_budget_is_spent():
    engine = _engine(limit=3)
    for index in range(3):
        verdict = engine.approve(_plan(), 10_000.0, 0.0, DAY + index * 60_000)
        assert verdict["approved"] is True, verdict.get("reason")
        engine.record_entry()
    refused = engine.approve(_plan(), 10_000.0, 0.0, DAY + 600_000)
    assert refused["approved"] is False
    assert refused["reason"] == "daily_trade_limit"
    # The refusal says what the budget was, so a cap can be told from a halt.
    assert refused["trades_today"] == 3 and refused["daily_trade_limit"] == 3


def test_zero_disables_the_cap_and_the_budget_rolls_with_the_date():
    unlimited = _engine(limit=0)
    for index in range(50):
        assert unlimited.approve(_plan(), 10_000.0, 0.0, DAY + index * 1_000)["approved"]
        unlimited.record_entry()
    assert unlimited.trades_remaining() is None

    engine = _engine(limit=2)
    engine.record_entry(); engine.record_entry()
    assert engine.approve(_plan(), 10_000.0, 0.0, DAY)["reason"] == "daily_trade_limit"
    # The drawdown halt is not a daily quantity: a new date must not clear it. The trade
    # count is, so it must. Both are decided in _roll_day now -- there used to be a second
    # copy of the roll for the event-time branch, which production always takes, and the
    # two had already drifted apart.
    engine.halt_scope = "high_water_mark"
    engine.halted = True
    engine.halt_reason = "drawdown_from_high_water_mark"
    verdict = engine.approve(_plan(), 10_000.0, 0.0, DAY + 86_400_000)
    assert engine.trades_today == 0
    assert engine.day_key == "2026-01-02"
    assert engine.halted is True and engine.halt_scope == "high_water_mark"
    assert verdict["approved"] is False and verdict["reason"] == "drawdown_from_high_water_mark"
    assert engine.trades_remaining() == 2


def test_the_count_survives_a_restart_and_reaches_the_dashboard():
    engine = _engine(limit=5)
    engine.record_entry(); engine.record_entry()
    restored = RiskEngine.restore(engine.snapshot(), default_equity=10_000.0)
    # A cap that forgets how much of the day has been spent restarts the budget on every
    # deploy, which is exactly when a runaway loop is most likely to be running.
    assert restored.daily_trade_limit == 5 and restored.trades_today == 2
    assert restored.trades_remaining() == 3

    body = render(collect({"equity": 10_000.0, "account": {}, "risk_state": engine.snapshot()}))
    assert "paper_risk_trades_today 2" in body
    assert "paper_risk_daily_trade_limit 5" in body
    assert "paper_risk_trades_remaining 3" in body


def test_a_refused_plan_does_not_spend_the_budget():
    """approve() also runs for plans something else then refuses.

    Counting inside approve() would spend the daily budget on orders that never existed,
    so a cap meant to bound runaway trading would tighten itself the more the rest of the
    system said no.
    """
    engine = _engine(limit=2)
    for _ in range(10):
        # An unusable plan: entry equal to stop. approve() is called, nothing is counted.
        assert engine.approve(_plan(stop=100.0), 10_000.0, 0.0, DAY)["approved"] is False
    assert engine.trades_today == 0
    assert engine.approve(_plan(), 10_000.0, 0.0, DAY)["approved"] is True


def test_a_new_session_starts_the_day_clean():
    engine = _engine(limit=1)
    engine.record_entry()
    assert engine.approve(_plan(), 10_000.0, 0.0, DAY)["approved"] is False
    engine.reset_for_session(10_000.0, day_key="2026-01-02")
    assert engine.trades_today == 0
    assert engine.approve(_plan(), 10_000.0, 0.0, DAY)["approved"] is True


def test_a_stale_halt_reason_does_not_outlive_the_halt():
    """The event-time branch left halt_reason from the previous day in place.

    The two copies of the day roll disagreed about which fields a new day resets, so after
    a daily breaker fired, the next day carried yesterday reason code into every refusal
    and into the dashboard, saying the account was halted when it was not.
    """
    engine = _engine(limit=5)
    engine.halted = True
    engine.halt_scope = "daily"
    engine.halt_reason = "daily_loss_circuit_breaker"
    engine.halt_equity = 5_000.0
    engine.halt_threshold = 8_000.0
    verdict = engine.approve(_plan(), 10_000.0, 0.0, DAY + 86_400_000)
    assert verdict["approved"] is True
    assert engine.halted is False and engine.halt_reason == ""
    assert engine.halt_equity == 0.0 and engine.halt_threshold == 0.0