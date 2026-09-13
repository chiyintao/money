"""Named risk profiles: the knob that decides how aggressive a session trades.

Before this, aggressiveness was fixed in .env at startup. Changing it meant editing a
file and restarting, and there was no record of which settings a past session traded
under. The stored sessions made that concrete: a 100 USDT account at 10x leverage
deployed about 16 USDT per entry and 60 USDT in total, because MAX_RISK_PER_TRADE was
0.002 and MAX_POSITIONS was 4 while the session had selected 7 symbols.

A profile names a coherent set of sizing parameters. The presets are ordered, so they
double as documentation of what each level of aggression actually implies. Any field can
still be overridden per session, and the resolved values are written into the session's
start event, so a trade can always be explained by the parameters in force when it
happened.

Sizing follows the constraint set in risk.py: the per-trade risk budget, the portfolio
risk budget spread across open positions, a target exposure, and the notional caps.
Kelly puts full-Kelly risk for a 45% win rate at 2:1 payoff near 17.5% of equity, with
practitioners using a half or quarter of that to absorb estimation error, so the presets
below are deliberately far inside even a quarter-Kelly bound.
"""
from dataclasses import asdict, dataclass, replace

# Ordering matters: index is the aggressiveness rank shown in the UI.
PROFILE_ORDER = ('conservative', 'balanced', 'assertive', 'aggressive', 'maximum')


@dataclass(frozen=True)
class RiskProfile:
    name: str
    label: str
    max_risk_per_trade: float
    max_portfolio_risk: float
    target_exposure: float
    max_positions: int
    max_symbol_leverage: float
    max_gross_leverage: float
    max_daily_loss: float
    description: str

    def snapshot(self):
        return asdict(self)


# Each level roughly doubles the per-trade risk of the one before it and widens the
# portfolio budget in step, so the numbers stay internally consistent: if every position
# were stopped out at once, the loss is max_portfolio_risk, never the sum of the
# per-trade budgets.
PRESETS = {
    'conservative': RiskProfile(
        name='conservative', label='稳健',
        max_risk_per_trade=.0025, max_portfolio_risk=.01, target_exposure=1.0,
        max_positions=4, max_symbol_leverage=2.0, max_gross_leverage=3.0, max_daily_loss=.15,
        description='单笔 0.25% 风险，组合 1%。回撤最小，适合验证模型是否真的有边际。'),
    'balanced': RiskProfile(
        name='balanced', label='均衡',
        max_risk_per_trade=.005, max_portfolio_risk=.02, target_exposure=2.0,
        max_positions=5, max_symbol_leverage=3.0, max_gross_leverage=5.0, max_daily_loss=.20,
        description='单笔 0.5% 风险，组合 2%，目标敞口 2 倍权益。当前默认档。'),
    'assertive': RiskProfile(
        name='assertive', label='进取',
        max_risk_per_trade=.010, max_portfolio_risk=.04, target_exposure=3.0,
        max_positions=6, max_symbol_leverage=4.0, max_gross_leverage=8.0, max_daily_loss=.25,
        description='单笔 1% 风险，组合 4%，目标敞口 3 倍。接近业界常用的 1-2% 单笔风险规则。'),
    'aggressive': RiskProfile(
        name='aggressive', label='激进',
        max_risk_per_trade=.020, max_portfolio_risk=.08, target_exposure=5.0,
        max_positions=8, max_symbol_leverage=6.0, max_gross_leverage=12.0, max_daily_loss=.30,
        description='单笔 2% 风险，组合 8%，目标敞口 5 倍。对应 10 倍杠杆下约 8% 的不利波动触及强平。'),
    'maximum': RiskProfile(
        name='maximum', label='极限',
        max_risk_per_trade=.030, max_portfolio_risk=.12, target_exposure=8.0,
        max_positions=10, max_symbol_leverage=10.0, max_gross_leverage=20.0, max_daily_loss=.40,
        description='单笔 3% 风险，组合 12%，目标敞口 8 倍。贴近强平边缘，仅用于短周期验证。'),
}

DEFAULT_PROFILE = 'balanced'

# Hard bounds. A profile outside these is a configuration mistake, not a preference, and
# the failure mode is a margin call rather than a bad backtest.
LIMITS = {
    'max_risk_per_trade': (0.0, 0.10),
    'max_portfolio_risk': (0.0, 0.50),
    'target_exposure': (0.0, 50.0),
    'max_positions': (1, 50),
    'max_symbol_leverage': (0.0, 50.0),
    'max_gross_leverage': (0.0, 100.0),
    'max_daily_loss': (0.0, 1.0),
}


def profile_names():
    return list(PROFILE_ORDER)


def get_profile(name, fallback=DEFAULT_PROFILE):
    """Preset by name, defaulting rather than raising so an unknown stored value cannot
    stop the service from starting."""
    return PRESETS.get(name or '', PRESETS[fallback])


def custom_profile(base, overrides):
    """Apply per-session overrides to a preset and validate the result.

    Raises ValueError with the offending field and bound, so the API returns a 400 that
    names the problem instead of silently clamping a risk limit the caller asked for.
    """
    base = get_profile(base) if isinstance(base, str) else base
    clean = {}
    for key, value in (overrides or {}).items():
        if key not in LIMITS:
            raise ValueError('unknown_risk_field:' + str(key))
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError('invalid_risk_value:' + str(key))
        low, high = LIMITS[key]
        integer = key == 'max_positions'
        if integer:
            number = int(number)
        if not low <= number <= high:
            raise ValueError('risk_field_out_of_range:%s:%s-%s' % (key, low, high))
        clean[key] = number
    if not clean:
        return base
    return replace(base, name='custom', label='自定义', description='会话自定义参数。', **clean)


def resolve(name=None, overrides=None):
    """A profile from a preset name plus optional overrides."""
    return custom_profile(get_profile(name), overrides)


def describe():
    """Every preset plus the bounds, for the UI and /api/risk/profiles."""
    return {'default': DEFAULT_PROFILE,
            'order': profile_names(),
            'limits': LIMITS,
            'profiles': [PRESETS[name].snapshot() for name in PROFILE_ORDER]}
