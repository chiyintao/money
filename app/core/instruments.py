"""Venue-shaped value objects: how a contract rounds, and how it liquidates.

These two definitions were the reason the venue layer imported the order layer. `market.py`
builds ContractSpec from the exchange's own filters and MaintenanceTier from its leverage
brackets, and both were defined next to the code that consumes them -- `PaperBroker` and
`MarginModel` respectively. That made the bottom of the stack depend on the top of it, and
the cycle was hidden behind a function-local import at both call sites.

They live in core because they are the shared vocabulary between the venue and the order
layer: a value object with no behaviour beyond parsing and rounding has no business
depending on the broker that uses it.
"""
from dataclasses import dataclass


@dataclass
class ContractSpec:
    """How one symbol rounds price and quantity, and what it costs to hold."""

    symbol: str
    tick_size: float = .01
    step_size: float = .001
    min_notional: float = 5.0
    leverage: float = 1.0
    maintenance_margin_rate: float = .005

    def round_price(self, value):
        return round(round(value / self.tick_size) * self.tick_size, 8)

    def round_qty(self, value):
        return round(int(value / self.step_size) * self.step_size, 8)


@dataclass(frozen=True)
class MaintenanceTier:
    """One bracket: the notional it covers up to, its rate, and its cumulative amount.

    The cumulative amount is what the venue publishes and what makes the tiers continuous:
    maintenance margin at a tier is notional * rate - amount, so the value does not jump
    when a position crosses a boundary.
    """

    notional_cap: float
    rate: float
    amount: float = 0.0
