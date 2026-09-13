"""Reproducible per-symbol, per-horizon edge report.

Every conclusion behind the meta-labeling gate came from a single ad-hoc analysis over a
few days of data, with some symbols contributing barely more than a dozen independent
observations. This turns that analysis into a command that can be re-run at any time, so
a parameter change can be justified against the same measurement rather than against a
number someone remembered.

    python -m app.edge_report
    python -m app.edge_report --symbols TACUSDT,VTHOUSDT --min-samples 20
    python -m app.edge_report --json data/edge_report.json
"""
import argparse
import json
import sys

from ..trading.broker import PaperBroker
from ..core.config import Settings
from ..storage.storage import Store
from .symbol_edge import DEFAULT_HORIZONS, horizon_analysis


def _parse_symbols(value):
    if not value:
        return None
    return {token.strip().upper() for token in value.split(',') if token.strip()}


def render(analysis):
    """Human-readable tables. Pooled first, then the per-symbol grid."""
    lines = []
    pooled = analysis['pooled']
    lines.append('round-trip cost assumed: %.2f bps   interval: %s' % (
        analysis['cost'] * 10000, analysis['interval']))
    lines.append('')
    lines.append('PASS 1 - pooled across every symbol')
    lines.append('%-8s %8s %9s %14s' % ('bars', 'n', 'hit%', 'mean_net_bps'))
    for horizon in analysis['horizons']:
        stats = pooled.get(horizon)
        if not stats:
            continue
        lines.append('%-8d %8d %8.1f%% %14.2f' % (
            horizon, stats['n'], 100 * stats['hit_rate'], stats['mean_net_bps']))
    best = analysis.get('best_horizon')
    lines.append('')
    lines.append('best horizon by pooled net edge: %s' % (
        ('%d bars' % best) if best else 'none (no horizon is positive after costs)'))
    lines.append('')
    symbols = analysis['symbols']
    if not symbols:
        lines.append('PASS 2 - no symbol has enough resolved observations yet')
        return '\n'.join(lines)
    horizons = [h for h in analysis['horizons']
                if any(h in symbols[s] for s in symbols)]
    lines.append('PASS 2 - mean net bps per symbol per horizon')
    lines.append('%-20s' % 'symbol' + ''.join('%10s' % ('%db' % h) for h in horizons)
                 + '%10s' % 'verdict')
    # Best first: the table exists to show which symbols earn their costs.
    def score(symbol):
        values = [symbols[symbol][h]['mean_net_bps'] for h in horizons if h in symbols[symbol]]
        return max(values) if values else float('-inf')
    for symbol in sorted(symbols, key=score, reverse=True):
        cells = []
        for horizon in horizons:
            stats = symbols[symbol].get(horizon)
            cells.append('%10.1f' % stats['mean_net_bps'] if stats else '%10s' % '-')
        positive = sum(1 for h in horizons
                       if h in symbols[symbol] and symbols[symbol][h]['mean_net_bps'] > 0)
        total = sum(1 for h in horizons if h in symbols[symbol])
        verdict = 'tradeable' if total and positive > total / 2 else ('mixed' if positive else 'reject')
        lines.append('%-20s' % str(symbol)[:20] + ''.join(cells) + '%10s' % verdict)
    lines.append('')
    lines.append('verdict is a majority vote over horizons with enough samples; the gate itself')
    lines.append('uses the configured single horizon, so treat this as corroboration.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Per-symbol, per-horizon edge report')
    parser.add_argument('--symbols', help='comma separated subset to analyse')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='minimum resolved observations before a cell is reported')
    parser.add_argument('--horizons', default=','.join(str(h) for h in DEFAULT_HORIZONS),
                        help='comma separated horizon in bars')
    parser.add_argument('--fee-rate', type=float, default=None,
                        help='taker fee per side; defaults to the broker cost model')
    parser.add_argument('--slippage-bps', type=float, default=None,
                        help='slippage per side in bps; defaults to the broker cost model')
    parser.add_argument('--json', dest='json_path', help='also write the raw analysis here')
    args = parser.parse_args(argv)

    settings = Settings()
    store = Store(settings.data_dir)
    # The broker owns the cost model, so the report defaults to exactly what the paper
    # service charges rather than to a number that could drift away from it.
    broker = PaperBroker()
    fee = broker.fee_rate if args.fee_rate is None else args.fee_rate
    slippage = broker.slippage_bps if args.slippage_bps is None else args.slippage_bps
    cost = 2 * float(fee) + 2 * float(slippage) / 10000
    horizons = tuple(int(h) for h in args.horizons.split(',') if h.strip())
    analysis = horizon_analysis(store, settings.interval, horizons=horizons, cost=cost,
                                min_samples=args.min_samples,
                                symbols=_parse_symbols(args.symbols))
    print(render(analysis))
    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as handle:
            json.dump(analysis, handle, ensure_ascii=False, indent=2, default=str)
        print('\nwrote %s' % args.json_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
