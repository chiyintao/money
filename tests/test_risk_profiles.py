"""Risk profiles make aggressiveness a setting rather than a startup constant.

Aggressiveness used to be fixed in .env: changing it meant editing a file and
restarting, and a past session could not be explained by the limits it traded under.
The stored sessions showed the cost -- a 100 USDT account at 10x deployed about 16 USDT
per entry, because MAX_RISK_PER_TRADE was 0.002 and MAX_POSITIONS was 4 against 7
selected symbols.
"""
import pytest

from app.trading import risk_profiles
from app.trading.guards import PortfolioLimits
from app.trading.risk import RiskEngine


def test_every_preset_is_complete_and_ordered_by_aggression():
    previous = None
    for name in risk_profiles.PROFILE_ORDER:
        profile = risk_profiles.PRESETS[name]
        assert profile.name == name
        assert profile.label and profile.description
        if previous is not None:
            # The presets double as documentation, so they must actually escalate.
            assert profile.max_risk_per_trade > previous.max_risk_per_trade
            assert profile.max_portfolio_risk > previous.max_portfolio_risk
            assert profile.target_exposure >= previous.target_exposure
            assert profile.max_positions >= previous.max_positions
        previous = profile


def test_portfolio_risk_covers_the_per_trade_risks_it_permits():
    # If every slot were stopped out at once the loss must be the portfolio budget, not
    # the per-trade budget multiplied by the slot count.
    for name in risk_profiles.PROFILE_ORDER:
        profile = risk_profiles.PRESETS[name]
        assert profile.max_portfolio_risk >= profile.max_risk_per_trade


def test_unknown_profile_falls_back_instead_of_raising():
    # A bad stored name must not stop the service from starting.
    assert risk_profiles.get_profile('nonsense') is risk_profiles.PRESETS[risk_profiles.DEFAULT_PROFILE]
    assert risk_profiles.get_profile(None) is risk_profiles.PRESETS[risk_profiles.DEFAULT_PROFILE]


def test_an_unknown_preset_name_resolves_to_the_default():
    assert risk_profiles.resolve('typo').name == risk_profiles.DEFAULT_PROFILE


def test_overrides_replace_named_fields_only():
    profile = risk_profiles.resolve('balanced', {'max_risk_per_trade': .03})
    balanced = risk_profiles.PRESETS['balanced']
    assert profile.max_risk_per_trade == .03
    assert profile.max_portfolio_risk == balanced.max_portfolio_risk
    assert profile.name == 'custom'


def test_overrides_are_validated_against_hard_bounds():
    with pytest.raises(ValueError, match='risk_field_out_of_range:max_risk_per_trade'):
        risk_profiles.resolve('balanced', {'max_risk_per_trade': 0.5})
    with pytest.raises(ValueError, match='risk_field_out_of_range:max_positions'):
        risk_profiles.resolve('balanced', {'max_positions': 0})
    with pytest.raises(ValueError, match='unknown_risk_field'):
        risk_profiles.resolve('balanced', {'not_a_field': 1})
    with pytest.raises(ValueError, match='invalid_risk_value'):
        risk_profiles.resolve('balanced', {'max_risk_per_trade': 'abc'})


def test_no_overrides_returns_the_named_preset_unchanged():
    assert risk_profiles.resolve('aggressive').name == 'aggressive'


def test_engine_takes_every_sizing_value_from_the_profile():
    engine = RiskEngine(day_start_equity=100)
    profile = risk_profiles.PRESETS['assertive']
    engine.apply_profile(profile)
    assert engine.max_risk == profile.max_risk_per_trade
    assert engine.max_portfolio_risk == profile.max_portfolio_risk
    assert engine.target_exposure == profile.target_exposure
    assert engine.max_symbol_leverage == profile.max_symbol_leverage
    assert engine.max_gross_leverage == profile.max_gross_leverage
    assert engine.max_daily_loss == profile.max_daily_loss


def test_applying_a_profile_keeps_the_daily_breaker_state():
    # day_start_equity and the halt flag are state, not policy. Clearing them would
    # silently disarm the circuit breaker that the profile is meant to work with.
    engine = RiskEngine(day_start_equity=100)
    engine.reset_for_session(100)
    engine.day_start_equity = 250
    engine.halted = True
    engine.apply_profile(risk_profiles.PRESETS['aggressive'])
    assert engine.day_start_equity == 250
    assert engine.halted is True


def test_limits_take_their_caps_from_the_profile():
    limits = PortfolioLimits()
    profile = risk_profiles.PRESETS['aggressive']
    limits.apply_profile(profile)
    assert limits.max_positions == profile.max_positions
    assert limits.max_gross_leverage == profile.max_gross_leverage


def test_a_per_symbol_cap_above_the_total_is_still_clamped():
    limits = PortfolioLimits()
    limits.apply_profile(risk_profiles.PRESETS['conservative'])
    assert limits.max_symbol_leverage <= limits.max_gross_leverage


def test_more_aggressive_profiles_size_up_on_the_same_plan():
    # The whole point of the setting: the same trade, sized by different profiles.
    plan = {'side': 'LONG', 'entry': 100.0, 'stop': 98.0}
    sizes = {}
    for name in risk_profiles.PROFILE_ORDER:
        engine = RiskEngine(day_start_equity=100)
        engine.apply_profile(risk_profiles.PRESETS[name])
        decision = engine.approve(plan, 100.0, 0)
        assert decision['approved'] is True
        sizes[name] = decision['notional']
    assert sizes['conservative'] < sizes['balanced'] < sizes['assertive'] < sizes['aggressive']
    # Balanced is the current default and should already beat the timid configuration
    # the stored sessions ran with.
    assert sizes['balanced'] > sizes['conservative']


def test_describe_exposes_the_catalogue_and_the_bounds():
    described = risk_profiles.describe()
    assert described['default'] == risk_profiles.DEFAULT_PROFILE
    assert described['order'] == list(risk_profiles.PROFILE_ORDER)
    assert len(described['profiles']) == len(risk_profiles.PROFILE_ORDER)
    assert 'max_risk_per_trade' in described['limits']
