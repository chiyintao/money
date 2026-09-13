from types import SimpleNamespace

from app.trading.guards import PortfolioLimits, portfolio_limits


def fake_settings(**overrides):
    base = {"max_positions": 5, "max_symbol_leverage": 3.0, "max_gross_leverage": 5.0}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_limits_come_from_settings():
    limits = portfolio_limits(fake_settings(max_positions=3, max_symbol_leverage=4.0, max_gross_leverage=6.0))
    assert limits.max_positions == 3
    assert limits.max_symbol_leverage == 4.0
    assert limits.max_gross_leverage == 6.0


def test_per_symbol_cap_is_clamped_to_the_total_cap():
    # A per-symbol cap above the total can never bind, so it must not be left higher.
    limits = portfolio_limits(fake_settings(max_symbol_leverage=9.0, max_gross_leverage=2.0))
    assert limits.max_symbol_leverage == 2.0


def test_caps_are_multiples_of_equity_not_absolute_amounts():
    # Regression: the caps were dollar amounts, so the same number meant "unreachable"
    # on a 100 account and "binding" on a 10000 one.
    limits = PortfolioLimits(max_symbol_leverage=3.0, max_gross_leverage=5.0)
    assert limits.approve("BTCUSDT", 299, {}, equity=100) == (True, "portfolio_ok")
    assert limits.approve("BTCUSDT", 301, {}, equity=100)[1] == "max_symbol_notional"
    assert limits.approve("BTCUSDT", 29900, {}, equity=10000) == (True, "portfolio_ok")
    assert limits.approve("BTCUSDT", 30100, {}, equity=10000)[1] == "max_symbol_notional"


def test_total_notional_cap_still_binds():
    limits = PortfolioLimits(max_symbol_leverage=3.0, max_gross_leverage=5.0)
    assert limits.approve("ETHUSDT", 300, {"BTCUSDT": 300}, equity=100)[1] == "max_total_notional"


def test_a_risk_sized_entry_is_accepted_at_the_default_caps():
    # Regression: risking 0.5% of a 10000 account over a 0.4% stop needs ~12500 of
    # notional. The old hardcoded 5000 cap rejected every such order, so no position
    # could ever open.
    limits = portfolio_limits(fake_settings())
    assert limits.approve("BTCUSDT", 12519, {}, equity=10000) == (True, "portfolio_ok")


def test_position_count_cap_still_binds():
    limits = PortfolioLimits(max_positions=1)
    assert limits.approve("ETHUSDT", 100, {"BTCUSDT": 100}, equity=1000)[1] == "max_positions"
    # Adding to an existing symbol is not a new position.
    assert limits.approve("BTCUSDT", 100, {"BTCUSDT": 100}, equity=1000)[0]


def test_without_equity_only_the_position_count_is_enforceable():
    limits = PortfolioLimits(max_positions=1)
    assert limits.approve("X", 10 ** 9, {})[0]
    assert limits.approve("Y", 1, {"X": 1})[1] == "max_positions"
