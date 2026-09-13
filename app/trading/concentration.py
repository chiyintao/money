"""Concentration and volatility, the two ways a sized-at-submit basket still went wrong.

Every entry was sized in isolation. Per-trade risk, per-symbol notional and gross notional
were all capped, and none of them can see that six "different" symbols are one trade: in a
risk-off move the whole basket is short beta and the losses arrive together, so a book that
passes every individual cap can lose several times the portfolio risk budget in one hour.
The measured session data shows exactly this -- nine symbols, one dominant factor.

Two corrections, both computed from bars already stored and neither needing a new data
source:

* correlation-aware exposure -- the notional held in symbols historically correlated with
  the new one is capped separately from gross notional;
* volatility targeting -- size is scaled by target volatility over realized, so a symbol
  whose stop distance suddenly covers three times the usual range does not silently take
  three times the risk.
"""
import math

DEFAULT_WINDOW = 120
DEFAULT_CORRELATION_THRESHOLD = 0.6


def returns(closes):
    """Simple close-to-close returns, skipping unusable pairs."""
    out = []
    for previous, current in zip(closes, closes[1:]):
        try:
            previous = float(previous)
            current = float(current)
        except (TypeError, ValueError):
            continue
        if previous <= 0 or current <= 0:
            continue
        out.append(current / previous - 1.0)
    return out


def realized_volatility(closes, window=DEFAULT_WINDOW):
    """Standard deviation of returns, per bar. Zero when there is not enough history."""
    series = returns([float(value) for value in (closes or [])][-int(window) - 1:])
    if len(series) < 5:
        return 0.0
    mean = sum(series) / len(series)
    variance = sum((item - mean) ** 2 for item in series) / (len(series) - 1)
    deviation = variance ** 0.5
    return deviation if math.isfinite(deviation) else 0.0


def volatility_scale(realized, target, floor=0.25, cap=1.0):
    """Risk-budget multiplier that aims realized risk at a target volatility.

    Capped at 1.0 by default: this shrinks exposure when a symbol is unusually volatile and
    does not lever up when it is unusually calm, because a quiet market is not evidence that
    the next bar will be quiet, and the cap is the difference between targeting volatility
    and writing an unhedged short-volatility position.
    """
    try:
        realized = float(realized)
        target = float(target)
    except (TypeError, ValueError):
        return 1.0
    if realized <= 0 or target <= 0 or not math.isfinite(realized):
        return 1.0
    scale = target / realized
    if not math.isfinite(scale):
        return 1.0
    return max(float(floor), min(float(cap), scale))


def correlation(left, right, window=DEFAULT_WINDOW):
    """Pearson correlation of two return series. None when it cannot be measured."""
    a = returns([float(value) for value in (left or [])][-int(window) - 1:])
    b = returns([float(value) for value in (right or [])][-int(window) - 1:])
    size = min(len(a), len(b))
    if size < 10:
        return None
    a, b = a[-size:], b[-size:]
    mean_a = sum(a) / size
    mean_b = sum(b) / size
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    value = cov / math.sqrt(var_a * var_b)
    return value if math.isfinite(value) else None


def correlation_matrix(series_by_symbol, window=DEFAULT_WINDOW):
    """Pairwise correlations for the symbols that have enough history."""
    symbols = sorted(series_by_symbol)
    matrix = {}
    for index, left in enumerate(symbols):
        for right in symbols[index + 1:]:
            value = correlation(series_by_symbol[left], series_by_symbol[right], window)
            if value is not None:
                matrix[(left, right)] = value
                matrix[(right, left)] = value
    return matrix


def correlated_notional(symbol, open_notionals, matrix, threshold=DEFAULT_CORRELATION_THRESHOLD):
    """Notional already held in symbols correlated with this one beyond a threshold.

    Unmeasurable pairs are excluded rather than assumed safe: a pair with no overlapping
    history is unknown, and treating unknown as uncorrelated is how a new listing gets to
    double an existing position in the same underlying.
    """
    total = 0.0
    for other, notional in (open_notionals or {}).items():
        if other == symbol:
            continue
        value = matrix.get((symbol, other))
        if value is None:
            continue
        if abs(value) >= float(threshold):
            total += abs(float(notional or 0.0))
    return total


def concentration_report(symbol, open_notionals, matrix, equity, threshold=DEFAULT_CORRELATION_THRESHOLD):
    """Human-readable concentration state, for the dashboard and the audit trail."""
    crowded = correlated_notional(symbol, open_notionals, matrix, threshold)
    gross = sum(abs(float(value or 0.0)) for value in (open_notionals or {}).values())
    return {'symbol': symbol, 'correlated_notional': crowded, 'gross_notional': gross,
            'equity': float(equity or 0.0),
            'correlated_share': (crowded / gross) if gross > 0 else 0.0,
            'threshold': float(threshold),
            'measured_pairs': sum(1 for other in (open_notionals or {})
                                  if (symbol, other) in matrix)}
