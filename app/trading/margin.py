"""Margin and liquidation, with the exchange's own risk tiers.

The previous model was three lines: maintenance margin was notional times a flat 0.5%,
liquidation happened when account equity fell to that number, and the position was closed
at the mark with the ordinary taker fee. Four things were missing and every one of them
made the paper account look safer than the contract it was imitating:

* The venue does not use one maintenance rate. It uses brackets -- 0.40% up to 50,000
  USDT of notional, then 0.50%, 1.00%, 2.50%, 5.00% on successively larger tiers -- so a
  large position is liquidated much sooner than a flat 0.5% implies. The bias is not
  symmetric: it only ever understates the risk of the positions that matter most.
* Liquidation carries its own fee, distinct from the taker fee. It was defined on the
  model and never read, so every liquidation was undercharged by half a percent of
  notional.
* Liquidation was judged on whole-account equity while the initial margin was computed
  from a leverage value that came from the session's UI input, not from the risk engine.
  The two never had to agree.
* Equity could go negative. When the gap through the maintenance level is larger than the
  remaining margin -- routine on a fast move -- the account simply carried a negative
  balance, which no venue permits and which silently inflates every subsequent metric.

This module now exposes a bracket schedule, a per-position liquidation price that uses it,
and a liquidation check that reports whether the account was actually made whole.
"""
from dataclasses import dataclass, field

from ..core.instruments import MaintenanceTier  # noqa: F401  (re-exported for callers)


# The venue's published USDT-M brackets, low leverage end. The upper tiers never bind for
# a research account, but the top rate is what decides the fate of a position that has
# grown into it, so the structure is carried rather than flattened.
DEFAULT_TIERS = (
    MaintenanceTier(50_000.0, 0.004, 0.0),
    MaintenanceTier(250_000.0, 0.005, 50.0),
    MaintenanceTier(1_000_000.0, 0.010, 1300.0),
    MaintenanceTier(5_000_000.0, 0.025, 16300.0),
    # 141,300 is not a round number on purpose: it is 5,000,000 * 5% minus the previous
    # tier's requirement of 108,700, which is what makes the top bracket continuous.
    MaintenanceTier(float('inf'), 0.050, 141_300.0),
)


class MaintenanceSchedule:
    """Per-symbol maintenance brackets, falling back to the published defaults."""

    def __init__(self, tiers=None, brackets=None):
        self.tiers = tuple(tiers or DEFAULT_TIERS)
        # symbol -> tuple of MaintenanceTier, as parsed from /fapi/v1/leverageBracket.
        self.brackets = dict(brackets or {})

    @classmethod
    def flat(cls, rate):
        """A single bracket at one rate, for venues without published risk tiers."""
        return cls(tiers=(MaintenanceTier(float('inf'), float(rate), 0.0),))

    def tiers_for(self, symbol):
        tiers = self.brackets.get(symbol)
        return tiers if tiers else self.tiers

    def tier_for(self, notional, symbol=None):
        notional = abs(float(notional or 0.0))
        chosen = None
        for tier in self.tiers_for(symbol):
            chosen = tier
            if notional <= tier.notional_cap:
                break
        return chosen if chosen is not None else self.tiers[-1]

    def maintenance_margin(self, notional, symbol=None):
        notional = abs(float(notional or 0.0))
        tier = self.tier_for(notional, symbol)
        # The floor of the chosen bracket, so the subtraction is exact at a boundary.
        return max(0.0, notional * tier.rate - tier.amount)

    def update(self, brackets):
        """Install parsed per-symbol brackets; a malformed entry is ignored, not fatal."""
        clean = {}
        for symbol, tiers in (brackets or {}).items():
            if tiers:
                clean[str(symbol)] = tuple(tiers)
        self.brackets.update(clean)
        return self


@dataclass
class MarginModel:
    leverage: float = 2.0
    # Kept for callers that pass it explicitly; the schedule is what actually decides,
    # and this is the rate of the first bracket when a schedule is absent.
    maintenance_rate: float = .004
    liquidation_fee_rate: float = .005
    schedule: MaintenanceSchedule = field(default_factory=MaintenanceSchedule)

    def initial_margin(self, notional):
        return abs(float(notional or 0.0)) / max(1e-9, float(self.leverage))

    def maintenance_margin(self, notional, symbol=None):
        if self.schedule is None:
            return abs(float(notional or 0.0)) * self.maintenance_rate
        return self.schedule.maintenance_margin(notional, symbol)

    def position_notional(self, position, price):
        return position.qty * price

    def maintenance_for(self, positions, marks):
        """Total maintenance requirement and the per-symbol breakdown behind it."""
        total = 0.0
        detail = {}
        for symbol, position in positions.items():
            notional = self.position_notional(position, marks.get(symbol, position.entry))
            value = self.maintenance_margin(notional, symbol)
            detail[symbol] = {'notional': notional, 'maintenance': value,
                              'rate': self.schedule.tier_for(notional, symbol).rate}
            total += value
        return total, detail

    def liquidation_price(self, position, symbol=None):
        """Price at which this position alone would breach its maintenance requirement.

        The bracket is chosen from the entry notional. Re-deriving it from the live
        notional would move the liquidation level as the position moves, which is not how
        the venue behaves: the bracket is fixed when the position is opened.
        """
        entry = float(position.entry)
        leverage = max(1e-9, float(self.leverage))
        notional = entry * float(position.qty)
        rate = self.maintenance_margin(notional, symbol or position.symbol) / max(1e-12, notional)
        buffer = 1.0 / leverage - rate
        if position.side in ('LONG', 'BUY'):
            return entry * (1 - buffer)
        return entry * (1 + buffer)

    def should_liquidate(self, equity, positions, marks):
        """Whether the account has fallen through its maintenance requirement.

        Returns (liquidate, maintenance_total, detail). The check is on account equity,
        which is what the venue enforces in cross margin mode -- the mode this account
        models -- so a single position's own distance to liquidation is not sufficient on
        its own.
        """
        maintenance, detail = self.maintenance_for(positions, marks)
        return float(equity) <= maintenance, maintenance, detail
