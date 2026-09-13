"""Prometheus text exposition.

The module held a two-method counter/gauge store that nothing ever called, while /metrics
served a single hand-written line reading paper_equity. Anything a monitor would want to
alert on -- decision throughput, gate rejections, feed staleness, funding settlements,
margin usage -- existed only inside the JSON dashboard payload, which means it could not be
alerted on at all.

The exposition here follows the text format: HELP and TYPE lines, a metric name per series,
and labels where a metric is naturally sliced by symbol, reason or outcome. Labels are
escaped and values are formatted as Prometheus expects (numbers, or +Inf).
"""
import time

# Metric name prefix for everything this service exports.
PREFIX = 'paper'

_ESCAPES = (('\\', '\\\\'), ('"', '\\"'), ('\n', '\\n'))


def escape_label(value):
    """Escape a label value per the exposition format."""
    text = str(value)
    for needle, replacement in _ESCAPES:
        text = text.replace(needle, replacement)
    return text


def format_value(value):
    """A float as Prometheus reads it: finite, or an explicit infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'NaN'
    if number != number:
        return 'NaN'
    if number == float('inf'):
        return '+Inf'
    if number == float('-inf'):
        return '-Inf'
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(round(number, 10))


def render(samples):
    """Render an iterable of sample dicts into the text exposition format.

    Each sample is {'name', 'type', 'help', 'value', 'labels'}. Ordering is stable so a
    diff between two scrapes is readable.
    """
    lines = []
    seen_headers = {}
    for sample in samples:
        name = sample['name']
        header = (sample.get('type', 'gauge'), sample.get('help'))
        if name not in seen_headers:
            seen_headers[name] = header
            if header[1]:
                lines.append('# HELP %s %s' % (name, header[1]))
            lines.append('# TYPE %s %s' % (name, header[0]))
        elif seen_headers[name] != header:
            # A metric whose type or help changes between samples in one scrape would
            # produce a file a Prometheus parser rejects.
            seen_headers[name] = header
        labels = sample.get('labels') or {}
        if labels:
            rendered = ','.join('%s="%s"' % (key, escape_label(value))
                                for key, value in sorted(labels.items()))
            lines.append('%s{%s} %s' % (name, rendered, format_value(sample['value'])))
        else:
            lines.append('%s %s' % (name, format_value(sample['value'])))
    return '\n'.join(lines) + ('\n' if lines else '')


def metric(name, value, kind='gauge', help=None, **labels):
    return {'name': '%s_%s' % (PREFIX, name), 'value': value, 'type': kind,
            'help': help, 'labels': labels}


# --------------------------------------------------------------- collection
# Reasons are enumerated so a rejection reason that is not on this list is still exported,
# under unknown; an alert is only useful if the series keeps existing.
def collect(state, runtime=None):
    """Project the assembled dashboard state into Prometheus samples.

    Reads the state dict the HTTP layer already builds rather than reaching into the
    runtime for each figure, so the two cannot disagree about what the account holds.
    """
    from ..core.metrics import cost_share

    account = state.get('account') or {}
    metrics = state.get('metrics') or {}
    risk = state.get('risk_state') or {}
    session = state.get('simulation') or {}
    health = state.get('health') or {}
    samples = []

    def add(name, value, kind='gauge', help=None, **labels):
        samples.append(metric(name, value, kind, help, **labels))

    # --- account
    add('equity', state.get('equity', 0), 'gauge', 'Account equity in quote currency')
    add('cash', account.get('cash', 0), 'gauge', 'Free cash')
    add('unrealized_pnl', account.get('unrealized_pnl', 0), 'gauge', 'Open position PnL')
    add('realized_pnl', account.get('realized_pnl', 0), 'gauge', 'Closed trade PnL')
    add('margin_used', account.get('margin_used', 0), 'gauge', 'Initial margin committed')
    add('maintenance_margin', account.get('maintenance_margin', 0), 'gauge',
        'Maintenance margin required by the schedule')
    add('margin_ratio', account.get('margin_ratio', 0), 'gauge',
        'Maintenance margin over equity; 1.0 is liquidation')
    add('open_positions', len(account.get('positions') or []))
    add('open_orders', len(state.get('open_orders') or []))

    # --- performance
    for key, help_text in (('return_pct', 'Return since session start, percent'),
                           ('max_drawdown_pct', 'Maximum drawdown, percent'),
                           ('sharpe', 'Annualised Sharpe ratio'),
                           ('sortino', 'Annualised Sortino ratio'),
                           ('profit_factor', 'Gross profit over gross loss'),
                           ('win_rate_pct', 'Winning trades, percent'),
                           ('turnover', 'Traded notional over average equity'),
                           ('exposure_pct', 'Share of bars holding a position')):
        value = metrics.get(key)
        if value is not None:
            add(key, value, 'gauge', help_text)
    add('trades_total', metrics.get('trades', 0), 'counter', 'Closed trades this session')
    add('fees_total', metrics.get('total_fees', 0), 'counter', 'Fees paid')
    add('funding_total', metrics.get('total_funding', 0), 'counter', 'Funding paid or received')
    costs = cost_share(metrics)
    if costs:
        add('cost_share_pct', costs['cost_share_pct'], 'gauge',
            'Costs as a share of gross profit')

    # --- risk
    add('risk_halted', 1 if risk.get('halted') else 0, 'gauge', 'Circuit breaker engaged')
    add('risk_peak_equity', risk.get('peak_equity', 0), 'gauge', 'High-water mark')
    add('risk_drawdown', (1 - (state.get('equity', 0) / risk['peak_equity']))
        if risk.get('peak_equity') else 0, 'gauge', 'Drawdown from the high-water mark')
    add('risk_loss_streak', risk.get('loss_streak', 0), 'gauge', 'Consecutive losing closes')
    # Order lifecycle moves the table refused. Non-zero means the venue produced a state
    # change the book could not record, which is a divergence rather than a statistic.
    refusals = ((state.get('reconciliation') or {}).get('order_transition_refusals')
                or state.get('order_transition_refusals') or {})
    if refusals:
        for reason, count in sorted(refusals.items()):
            add('order_transition_refused', count, 'counter',
                'Order status changes the lifecycle table refused', reason=reason)
    # Entries against the day's budget. A cap that is being approached looks exactly like a
    # quiet day on every other series here, and the difference matters before it binds.
    add('risk_trades_today', risk.get('trades_today', 0), 'gauge', 'Entries submitted today')
    if risk.get('daily_trade_limit'):
        add('risk_daily_trade_limit', risk.get('daily_trade_limit', 0), 'gauge',
            'Entries permitted per UTC day')
        add('risk_trades_remaining', risk.get('trades_remaining') or 0, 'gauge',
            'Entries left in the daily budget')
    add('risk_streak_scale', risk.get('streak_scale', 1), 'gauge',
        'Current size multiplier from the losing streak')

    # --- decisions and gates
    counts = state.get('decision_counts') or {}
    add('decisions_total', counts.get('total', 0), 'counter', 'Strategy decisions recorded')
    add('orders_total', counts.get('orders', 0), 'counter', 'Orders created')
    for reason, count in (counts.get('rejections') or {}).items():
        add('rejections_total', count, 'counter', 'Rejected entries by reason', reason=reason)

    # --- feeds
    add('ws_reconnects', health.get('ws_reconnects', 0), 'counter', 'WebSocket reconnects')
    for symbol_row in (state.get('symbols') or [])[:64]:
        symbol = symbol_row.get('symbol')
        if not symbol:
            continue
        age = symbol_row.get('age_ms')
        if age is not None:
            add('symbol_age_ms', age, 'gauge', 'Time since the last event for a symbol',
                symbol=symbol)
        price = symbol_row.get('price')
        if price:
            add('symbol_price', price, 'gauge', 'Last price', symbol=symbol)

    # --- derivatives
    derivatives = state.get('derivatives') or {}
    if derivatives.get('enabled'):
        add('derivatives_runs_total', derivatives.get('runs', 0), 'counter',
            'Derivative collection runs')
        add('derivatives_rows_total', derivatives.get('rows', 0), 'counter',
            'Derivative rows written')
        add('derivatives_failures_total', derivatives.get('failed', 0), 'counter',
            'Derivative collection failures')

    # --- venue
    venue = state.get('venue') or {}
    kind = venue.get('status')
    if kind:
        add('venue_ok', 1 if kind == 'ok' else 0, 'gauge',
            'Whether the stream and price venues agree')
        add('venue_checks_total', 1, 'counter', 'Venue checks performed',
            status=kind)

    # --- retention
    persistence = (state.get('persistence') or {}).get('events') or {}
    if persistence:
        add('events_rows', persistence.get('written', 0), 'counter', 'Audit rows written')

    add('scrape_timestamp_ms', int(time.time() * 1000), 'gauge', 'Time this scrape was built')
    return samples


def text(state, runtime=None):
    return render(collect(state, runtime))
