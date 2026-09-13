"""The matching layer, decomposed into named models.

LEAN splits execution into a fill model, a fee model, a slippage model, a margin model and
a settlement model so that each assumption can be stated, replaced and tested on its own.
This repository had one: execute() computed a price, a slippage and a fee inline, and every
caller that needed a fill went through it or -- worse -- built its own.

Two assumptions were wrong in ways that only show up when you look at them separately.

**Size did not enter the fill price at all.** Slippage was price * slippage_bps / 10000, so
a 50 USDT order and a 5,000,000 USDT order filled at exactly the same price. Measured on
the old code: quantities 0.001 and 10,000 BTCUSDT both filled at 50,011.00 against a
50,001.00 ask -- a 10.00 slippage each -- on notionals of 50 and 500,110,000 USDT. A paper
account that absorbs any size at the touch price cannot report a capacity limit, because it
does not have one.

**Every fill was a taker fill.** Fill carries a liquidity field, execute_depth hardcoded
'taker' and execute() took the default, so a resting limit order that filled was recorded
as having crossed the spread and was charged the taker rate. On the venue standard tier
that is 0.0400% against a maker rate of 0.0200% -- the fill was charged double, and because
the field said 'taker' nothing downstream could tell.
"""
from dataclasses import dataclass

MAKER = "maker"
TAKER = "taker"

# The venue standard USDT-M tier: 0.0200% maker, 0.0400% taker. Both are per side.
DEFAULT_MAKER_RATE = 0.0002
DEFAULT_TAKER_RATE = 0.0004


@dataclass
class FeeModel:
    """What a fill costs in fees, by the liquidity it took or provided."""

    maker_rate: float = DEFAULT_MAKER_RATE
    taker_rate: float = DEFAULT_TAKER_RATE

    def rate(self, liquidity=None):
        """The rate for one liquidity class. An unknown class is charged as a taker."""
        return self.maker_rate if str(liquidity or "").lower() == MAKER else self.taker_rate

    def fee(self, notional, liquidity=None):
        return abs(float(notional or 0.0)) * self.rate(liquidity)

    def describe(self):
        return {"maker_rate": self.maker_rate, "taker_rate": self.taker_rate,
                "maker_bps": self.maker_rate * 10000, "taker_bps": self.taker_rate * 10000}


@dataclass
class FillModel:
    """The price a fill happens at, and why it is worse than the quote.

    Two additive terms, because they have different causes and different scaling.

    spread_bps is the cost of crossing: it applies to any aggressive order, does not
    depend on size, and is zero for an order that rests and is hit.

    impact_coefficient_bps is market impact. It is expressed against participation -- the
    share of the liquidity available at the touch that this order consumes -- because that
    is the quantity the model actually has. An order for the whole touch moves the price by
    the coefficient; an order for a quarter of it moves it by half (square-root, the
    standard empirical form). When participation is unknown the model falls back to the
    order notional against reference_notional, and when both are unknown it charges the
    coefficient in full rather than pretending the order is small -- an unknown size is not
    a small size.
    """

    spread_bps: float = 2.0
    impact_coefficient_bps: float = 0.0
    reference_notional: float = 0.0
    # How long the order takes to reach the venue and come back, in milliseconds. The
    # decision is made on a bar close; the fill does not happen at that price. A backtest
    # that fills at the close implicitly assumes zero latency and no adverse move during
    # it, which is the single most flattering assumption available -- and it is exactly
    # the assumption that separates a strategy that works on paper from one that works.
    entry_latency_ms: float = 0.0
    # What share of the move during the latency window goes against the order. 1.0 is the
    # conservative reading (the market moved entirely against us while we were in flight),
    # 0.0 restores the old zero-latency behaviour, and 0.5 is the neutral assumption that
    # the drift is as likely to help as hurt -- which still costs money in expectation,
    # because a fill only happens when the order is marketable.
    latency_adverse_share: float = 1.0

    def impact_bps(self, participation=None, notional=None):
        if self.impact_coefficient_bps <= 0:
            return 0.0
        share = None
        if participation is not None:
            try:
                share = max(0.0, float(participation))
            except (TypeError, ValueError):
                share = None
        elif notional is not None and self.reference_notional > 0:
            try:
                share = max(0.0, float(notional) / float(self.reference_notional))
            except (TypeError, ValueError, ZeroDivisionError):
                share = None
        if share is None:
            # No size information at all. Charging zero here would make every unmeasured
            # fill the cheapest possible fill, which is the assumption this model exists
            # to remove.
            return float(self.impact_coefficient_bps)
        if share <= 0:
            return 0.0
        return float(self.impact_coefficient_bps) * (share ** 0.5)

    def latency_bps(self, latency_ms=None, volatility_bps=None, bar_ms=300_000.0):
        """Expected adverse drift, in bps, over the order's flight time.

        Diffusion scaling: the standard deviation of the price over a window is the
        per-bar deviation scaled by the square root of the window's share of a bar. An
        order in flight for one tenth of a bar therefore sees about a third of the bar's
        move, which is the right order of magnitude and does not require any new data.

        Returns 0.0 when there is nothing to scale -- no latency configured, no volatility
        estimate, or no bar length. That keeps this strictly additive: a deployment that
        sets no latency fills exactly as it did before.
        """
        latency = float(self.entry_latency_ms if latency_ms is None else latency_ms or 0.0)
        if latency <= 0:
            return 0.0
        try:
            sigma = max(0.0, float(volatility_bps or 0.0))
        except (TypeError, ValueError):
            return 0.0
        if sigma <= 0:
            return 0.0
        try:
            bar = float(bar_ms or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if bar <= 0:
            return 0.0
        share = max(0.0, min(1.0, latency / bar))
        adverse = max(0.0, min(1.0, float(self.latency_adverse_share or 0.0)))
        return sigma * (share ** 0.5) * adverse

    def slip_bps(self, liquidity=None, participation=None, notional=None,
                 latency_ms=None, volatility_bps=None, bar_ms=300_000.0,
                 include_spread=True, include_impact=True):
        """Total adverse move in basis points for this fill.

        ``include_spread`` and ``include_impact`` exist for the depth-walk path. That path
        consumes real ladder levels, so the spread it pays and the impact it causes are
        already in the price it computed; charging the model's spread on top would count
        the same cost twice. Latency is the one term it cannot have observed, so it is
        still added.
        """
        if str(liquidity or TAKER).lower() != TAKER:
            # A resting order that is hit fills at its own price. It does not cross the
            # spread and it does not move the market; what it pays instead is adverse
            # selection, which this model cannot see and does not pretend to model.
            return 0.0
        total = 0.0
        if include_spread:
            total += max(0.0, float(self.spread_bps or 0.0))
        if include_impact:
            total += self.impact_bps(participation, notional)
        # Latency is charged like the other two: a real cost of trading that scales with
        # volatility rather than a fixed toll, so a quiet market is cheap to reach and a
        # violent one is not.
        return total + self.latency_bps(latency_ms, volatility_bps, bar_ms)

    def price(self, quote_price, side, liquidity=None, participation=None, notional=None,
              latency_ms=None, volatility_bps=None, bar_ms=300_000.0,
              include_spread=True, include_impact=True):
        """The fill price. Buys pay more, sells receive less; a maker fill is not crossed."""
        price = float(quote_price or 0.0)
        if price <= 0:
            return 0.0
        move = price * self.slip_bps(liquidity, participation, notional,
                                     latency_ms, volatility_bps, bar_ms,
                                     include_spread, include_impact) / 10000.0
        if str(side or "").upper() in ("BUY", "LONG"):
            return price + move
        return price - move

    def describe(self):
        return {"spread_bps": self.spread_bps,
                "impact_coefficient_bps": self.impact_coefficient_bps,
                "reference_notional": self.reference_notional,
                "entry_latency_ms": self.entry_latency_ms,
                "latency_adverse_share": self.latency_adverse_share}


def participation(quantity, available):
    """Share of the liquidity at the touch an order consumes, or None if unmeasurable."""
    try:
        available = float(available)
        quantity = float(quantity)
    except (TypeError, ValueError):
        return None
    if available <= 0 or quantity <= 0:
        return None
    return quantity / available
