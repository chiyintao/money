"""One-page decision health report from the live dashboard API.

Kept as a script rather than a notebook cell because the shape of /api/state is wide
and mostly irrelevant here: this pulls out the decision funnel, the account, and the
recent realized round trips, and prints a compact summary.
"""
import json
import sys
import time
import urllib.request

URL = 'http://127.0.0.1:8101/api/state'


def fetch(url=URL, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _num(value, digits=2):
    try:
        return ('%.' + str(digits) + 'f') % float(value)
    except (TypeError, ValueError):
        return str(value)


def funnel(state):
    """Decision counts, ordered by how often the gate fired."""
    counts = state.get('decision_counts') or {}
    if not isinstance(counts, dict):
        return []
    return sorted(counts.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))


def trades(state):
    """Closed round trips.

    /api/state carries them under 'trades' in some builds and under 'all_time_trades'
    in others; both are the same closed-trade rows, and neither is a fill ledger.
    """
    for key in ('trades', 'all_time_trades'):
        rows = state.get(key)
        if isinstance(rows, list) and rows:
            return rows
    return []


def summarize(state, now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    lines = []
    lines.append('时间(本地)  : %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('')
    lines.append('=== 决策漏斗 ===')
    for name, value in funnel(state):
        lines.append('  %-30s %s' % (name, value))
    lines.append('')
    lines.append('=== 账户 ===')
    for key in ('equity', 'cash', 'available_margin', 'gross_leverage',
                'all_time_realized_pnl'):
        lines.append('  %-24s %s' % (key, state.get(key)))
    lines.append('  %-24s %s' % ('all_time_trades', len(trades(state))))
    lines.append('  %-24s %s' % ('open_positions', len(state.get('positions') or [])))
    lines.append('  %-24s %s' % ('open_orders', len(state.get('open_orders') or [])))
    errors = state.get('errors') or {}
    lines.append('  %-24s %s' % ('error_count', errors.get('count')))
    lines.append('  %-24s %s' % ('last_error', state.get('last_error')))
    lines.append('')

    rows = trades(state)
    lines.append('=== 已平仓回合（最近 20 笔）===')
    lines.append('  %-18s %-12s %-6s %10s %10s %12s  %s'
                 % ('exit_time', 'symbol', 'side', 'pnl', 'fees', 'holding_min', 'reason'))
    realized = 0.0
    fees = 0.0
    for row in rows:
        pnl = float(row.get('pnl') or 0.0)
        fee = float(row.get('fees') or 0.0)
        realized += pnl
        fees += fee
    for row in rows[:20]:
        held = float(row.get('holding_ms') or 0) / 60000.0
        stamp = time.strftime('%m-%d %H:%M:%S',
                              time.localtime((row.get('exit_time') or 0) / 1000.0))
        lines.append('  %-18s %-12s %-6s %10s %10s %12s  %s'
                     % (stamp, row.get('symbol'), row.get('side'),
                        _num(row.get('pnl'), 4), _num(row.get('fees'), 4),
                        _num(held, 1), row.get('reason')))
    lines.append('')
    lines.append('=== 汇总 ===')
    lines.append('  回合数            %d' % len(rows))
    lines.append('  净盈亏(含费)      %s' % _num(realized, 4))
    lines.append('  手续费合计        %s' % _num(fees, 4))
    if rows:
        wins = [r for r in rows if float(r.get('pnl') or 0) > 0]
        lines.append('  胜率              %s' % _num(100.0 * len(wins) / len(rows), 1) + '%')
    if now_ms:
        stamps = [r.get('exit_time') or 0 for r in rows]
        if stamps:
            lines.append('  最近一笔平仓距今  %s 分钟' % _num((now_ms - max(stamps)) / 60000.0, 1))
    return '\n'.join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    url = argv[0] if argv else URL
    try:
        state = fetch(url)
    except Exception as exc:
        print('cannot read %s: %r' % (url, exc))
        return 1
    print(summarize(state))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
