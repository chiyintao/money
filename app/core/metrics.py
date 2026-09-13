"""Performance metrics.

The annualisation factor was the literal 365*24*60 -- one-minute bars -- regardless of the
interval the service was configured with. At 5m every volatility and every Sharpe ratio was
inflated by sqrt(5) = 2.24, and at 15m by 3.87, silently and in the direction that makes a
strategy look better. The factor now comes from the interval.

The set was also thin in a way that hid things. Win rate, profit factor and expectancy all
describe trade *outcomes*; none of them describes how much capital had to be committed to
get those outcomes, so a strategy that turns the whole account over every hour and one that
holds a single position for a month report the same numbers. Turnover, exposure and Calmar
are here for that reason.
"""
import math
from math import sqrt

# Bars per year, by the interval those bars are measured in.
_MINUTES_PER_YEAR = 365 * 24 * 60
# Annualising needs a year of observations to be a rate rather than an extrapolation, and
# the exponent has to stay finite. A two-point curve at a 1m interval fails both tests.
MIN_RETURNS_FOR_ANNUALISATION = 365 * 24 * 2
MAX_ANNUALISATION_EXPONENT = 100.0


def periods_per_year(interval=None, interval_ms=None):
    """Annualisation factor for bars of a given interval.

    Accepts either the interval string or its width in milliseconds, so a caller that only
    has the resolved millisecond value does not have to reconstruct the string.
    """
    minutes = None
    if interval_ms:
        try:
            minutes = float(interval_ms) / 60000.0
        except (TypeError, ValueError):
            minutes = None
    if not minutes and interval:
        from .config import interval_to_ms

        try:
            minutes = interval_to_ms(str(interval)) / 60000.0
        except (TypeError, ValueError, KeyError):
            minutes = None
    if not minutes or minutes <= 0:
        minutes = 1.0
    return _MINUTES_PER_YEAR / minutes


def _drawdown(series):
    peak = series[0] if series else 0.0
    max_dd = 0.0
    run = 0
    longest = 0
    for value in series:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak else 0.0
        max_dd = max(max_dd, dd)
        run = run + 1 if dd > 0 else 0
        longest = max(longest, run)
    # The recovery run, which is what a drawdown limit is actually expressed against.
    return max_dd, longest


def _returns(series):
    out = []
    for previous, current in zip(series, series[1:]):
        if previous:
            out.append(current / previous - 1)
    return out


def summary(equity_curve, trades, initial_equity=None, periods_per_year=365 * 24 * 60,
            interval=None, interval_ms=None):
    """Performance for one equity curve and its trades.

    The periods_per_year parameter is kept for callers that pass it explicitly; interval or
    interval_ms take precedence, because the hardcoded default was the bug.
    """
    if interval or interval_ms:
        periods_per_year = globals()['periods_per_year'](interval=interval, interval_ms=interval_ms)
    values = [float(value) for value in equity_curve if value is not None]
    start = float(initial_equity if initial_equity is not None else (values[0] if values else 0))
    end = values[-1] if values else start
    series = ([start] + values) if values and values[0] != start else (values or [start])
    max_dd, max_drawdown_period = _drawdown(series)
    returns = _returns(series)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns) if returns else 0.0
    volatility = sqrt(variance) * sqrt(periods_per_year) if returns else 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_dev = (sqrt(sum(value * value for value in downside) / len(downside)) * sqrt(periods_per_year)
                    if returns else 0.0)
    sharpe = (mean_return * periods_per_year) / volatility if volatility else None
    sortino = (mean_return * periods_per_year) / downside_dev if downside_dev else None
    # Annualising a handful of observations is meaningless and, at these ratios, does not
    # even fit in a float: two points and a yearly factor raised 262800/1 produced an
    # OverflowError. A year's worth of bars is the minimum at which the figure means
    # anything, and the exponent is computed in log space so nothing can overflow.
    annual_return = None
    if start > 0 and end > 0 and len(returns) >= MIN_RETURNS_FOR_ANNUALISATION:
        exponent = periods_per_year / len(returns)
        if exponent <= MAX_ANNUALISATION_EXPONENT:
            annual_return = math.exp(min(math.log(end / start) * exponent, 50.0)) - 1

    pnls = [float(trade.get('pnl', 0) or 0) for trade in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    expectancy = sum(pnls) / len(pnls) if pnls else 0.0
    total_fees = sum(float(trade.get('fees', 0) or 0) for trade in trades)
    total_funding = sum(float(trade.get('funding', 0) or 0) for trade in trades)
    traded_notional = sum(abs(float(trade.get('entry_notional', 0) or 0))
                          for trade in trades)
    # Turnover against the average equity the account actually had, not the starting
    # balance: a strategy that doubles the account and then trades the same size is not
    # turning over half as fast.
    average_equity = (sum(series) / len(series)) if series else 0.0
    turnover = traded_notional / average_equity if average_equity else 0.0
    holding = [float(trade.get('holding_ms', 0) or 0) for trade in trades
               if trade.get('holding_ms')]
    # Exposure: the fraction of the window during which the account held anything.
    exposure_pct = (sum(1 for trade in trades if trade.get('holding_ms')) / len(series) * 100
                    if series and trades else 0.0)
    return {
        'start_equity': start, 'equity': end,
        'return_pct': (end / start - 1) * 100 if start else 0.0,
        'annualized_return_pct': None if annual_return is None else annual_return * 100,
        'max_drawdown_pct': max_dd * 100,
        'max_drawdown_periods': max_drawdown_period,
        'annualized_volatility_pct': volatility * 100,
        'sharpe': sharpe, 'sortino': sortino,
        'calmar': (annual_return / max_dd) if (annual_return is not None and max_dd) else None,
        'periods_per_year': periods_per_year,
        'trades': len(trades),
        'win_rate_pct': len(wins) / len(pnls) * 100 if pnls else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss else None,
        'average_win': gross_profit / len(wins) if wins else 0.0,
        'average_loss': sum(losses) / len(losses) if losses else 0.0,
        'best_trade': max(pnls) if pnls else 0.0,
        'worst_trade': min(pnls) if pnls else 0.0,
        'expectancy': expectancy, 'gross_profit': gross_profit, 'gross_loss': gross_loss,
        'total_fees': total_fees, 'total_funding': total_funding,
        'net_of_costs': gross_profit - gross_loss - total_fees,
        'turnover': turnover, 'traded_notional': traded_notional,
        'average_equity': average_equity,
        'exposure_pct': exposure_pct,
        'average_holding_bars': (sum(holding) / len(holding)) if holding else 0.0,
    }


def cost_share(metrics):
    """How much of the gross result the costs consumed.

    A strategy whose edge is real but entirely paid to the venue shows up here and nowhere
    else: profit factor and expectancy both look healthy because both are computed after
    costs have already been subtracted from each trade.
    """
    gross = float(metrics.get('gross_profit', 0) or 0)
    costs = float(metrics.get('total_fees', 0) or 0) + abs(float(metrics.get('total_funding', 0) or 0))
    if gross <= 0:
        return None
    return {'gross_profit': gross, 'costs': costs, 'cost_share_pct': costs / gross * 100,
            'net': gross - costs}
