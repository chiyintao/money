"""Startup check that the market data venue is the one the rest of the system assumes.

Why this exists. BINANCE_BASE_URL pointed at the production REST API while BINANCE_WS_URL
pointed at wss://fstream.binancefuture.com, which is not a production endpoint -- it is a
testnet host. The two were never compared, so the service built every feature, every closed
bar and every paper fill from a testnet order book while taking mark price, funding and
instrument filters from production. Nothing failed. The dashboard, the audit trail and the
model metrics were all self-consistent, because a wrong price is still a price.

Measured over twenty seconds of BTCUSDT bookTicker:

    endpoint                        msgs/20s   median spread   median bid qty   median ask qty
    fstream.binancefuture.com             129        0.54 bps        0.002 BTC        353.8 BTC
    fstream.binance.com                10,645        0.01 bps        2.156 BTC       10.864 BTC

Eighty-two times fewer updates, fifty-four times the spread, and a book whose best bid held
156 dollars of size against 27 million dollars on the offer. The depth-based fill model
reads exactly those two numbers to decide whether an order fills at the touch or walks the
book, so a one-sided testnet book made buys and sells fill under completely different rules.

The check is deliberately cheap (a few seconds, one symbol) and reports the raw numbers as
well as a verdict, so a disagreement is diagnosable rather than merely refused.
"""
import json
import statistics
import time

# Production hosts for USD-M futures. Anything else is either a testnet or a typo.
PRODUCTION_WS_HOSTS = ('fstream.binance.com', 'stream.binance.com')
PRODUCTION_REST_HOSTS = ('fapi.binance.com',)
TESTNET_MARKERS = ('binancefuture.com', 'demo-fstream', 'demo-fapi', 'testnet')

DEFAULT_SYMBOL = 'BTCUSDT'
DEFAULT_SECONDS = 6.0
# A production perpetual's spread is a fraction of a basis point and its deviation from the
# REST last price is a trade's worth of noise. Both thresholds sit well outside normal, so
# the check reports a venue mismatch rather than ordinary market movement.
MAX_MID_DEVIATION_BPS = 25.0
# A production major trades at about 0.01 bps at the touch. A full basis point is a
# hundred times that and still permissive for a thin alt, so it flags a genuinely impaired
# book without flagging an ordinary one.
MAX_SPREAD_BPS = 1.0
# A real book does not quote one side three orders of magnitude larger than the other at
# the touch. A testnet book does, because its liquidity is synthetic.
MAX_SIZE_IMBALANCE = 100.0


def classify_host(url):
    """Whether a URL looks like production, testnet, or neither."""
    text = str(url or '').strip().lower()
    host = text.split('://')[-1].split('/')[0]
    if any(marker in host for marker in TESTNET_MARKERS):
        return 'testnet'
    if host in PRODUCTION_WS_HOSTS or host in PRODUCTION_REST_HOSTS:
        return 'production'
    return 'unknown'


def host_mismatch(ws_url, rest_url):
    """A description of an inconsistent venue pairing, or None when they agree."""
    kind_ws = classify_host(ws_url)
    kind_rest = classify_host(rest_url)
    if 'testnet' in (kind_ws, kind_rest) and kind_ws != kind_rest:
        return {'reason': 'mixed_venue', 'ws': kind_ws, 'rest': kind_rest,
                'detail': 'market data from %s and prices from %s' % (kind_ws, kind_rest)}
    return None


def summarise(samples):
    """Message count, spread and book shape from a list of bookTicker messages."""
    rows = []
    for item in samples or []:
        try:
            bid, ask = float(item['b']), float(item['a'])
            bid_qty = float(item.get('B') or 0)
            ask_qty = float(item.get('A') or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        rows.append((bid, ask, bid_qty, ask_qty))
    if not rows:
        return None
    mids = [(bid + ask) / 2 for bid, ask, _, _ in rows]
    spreads = [(ask - bid) / bid * 10000 for bid, ask, _, _ in rows]
    bid_qty = statistics.median([row[2] for row in rows])
    ask_qty = statistics.median([row[3] for row in rows])
    sizes = [value for value in (bid_qty, ask_qty) if value > 0]
    imbalance = (max(sizes) / min(sizes)) if len(sizes) == 2 else 0.0
    return {'messages': len(rows), 'mid': statistics.median(mids),
            'spread_bps': statistics.median(spreads),
            'bid_qty': bid_qty, 'ask_qty': ask_qty,
            'size_imbalance': imbalance}


def verdict(sample, rest_last=None, elapsed_seconds=DEFAULT_SECONDS,
            max_deviation_bps=MAX_MID_DEVIATION_BPS, max_spread_bps=MAX_SPREAD_BPS,
            max_size_imbalance=MAX_SIZE_IMBALANCE):
    """Compare a sampled book against the REST price and against what a real book looks like.

    Returns a dict whose status is ok, suspect or unknown. Unknown is used when nothing
    could be sampled, and must not be reported as agreement: the previous arrangement looked
    fine precisely because nothing was compared.
    """
    if not sample:
        return {'status': 'unknown', 'findings': ['no_book_ticker_received'],
                'detail': 'the stream produced nothing to compare'}
    findings = []
    deviation = None
    if rest_last:
        try:
            rest_last = float(rest_last)
        except (TypeError, ValueError):
            rest_last = None
    if rest_last:
        deviation = (sample['mid'] - rest_last) / rest_last * 10000
        if abs(deviation) > max_deviation_bps:
            findings.append('mid_deviates_from_rest_price')
    if sample['spread_bps'] > max_spread_bps:
        findings.append('spread_wider_than_a_production_book')
    if sample['size_imbalance'] > max_size_imbalance:
        findings.append('one_sided_touch_size')
    seconds = max(0.5, float(elapsed_seconds or DEFAULT_SECONDS))
    return {'status': 'suspect' if findings else 'ok', 'findings': findings,
            'messages': sample['messages'],
            'messages_per_second': round(sample['messages'] / seconds, 2),
            'mid': sample['mid'], 'rest_last': rest_last,
            'deviation_bps': None if deviation is None else round(deviation, 3),
            'spread_bps': round(sample['spread_bps'], 4),
            'size_imbalance': round(sample['size_imbalance'], 3),
            'thresholds': {'max_deviation_bps': max_deviation_bps,
                           'max_spread_bps': max_spread_bps,
                           'max_size_imbalance': max_size_imbalance}}


async def sample_book_ticker(ws_url, symbol=DEFAULT_SYMBOL, seconds=DEFAULT_SECONDS,
                             max_messages=2000):
    """Collect best bid/ask for one symbol. Returns (samples, elapsed_seconds, error)."""
    import asyncio

    try:
        import websockets
    except ImportError:
        return [], 0.0, 'websockets_unavailable'
    url = '%s/ws/%s@bookTicker' % (str(ws_url).rstrip('/'), str(symbol).lower())
    samples = []
    started = time.time()
    try:
        async with websockets.connect(url, open_timeout=8) as socket:
            deadline = started + float(seconds)
            while time.time() < deadline and len(samples) < max_messages:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=3)
                except Exception:
                    break
                try:
                    samples.append(json.loads(raw))
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        return samples, time.time() - started, repr(exc)
    return samples, time.time() - started, None


async def check_venue(api, settings=None, symbol=DEFAULT_SYMBOL, seconds=DEFAULT_SECONDS):
    """Sample the configured stream and compare it with the configured REST price."""
    ws_url = getattr(settings, 'ws_url', None) or getattr(settings, 'binance_ws_url', '')
    rest_url = getattr(settings, 'base_url', None) or getattr(settings, 'binance_base_url', '')
    samples, elapsed, error = await sample_book_ticker(ws_url, symbol, seconds)
    report = verdict(summarise(samples), None, elapsed)
    report['symbol'] = symbol
    report['ws_url'] = str(ws_url)
    report['rest_url'] = str(rest_url)
    report['ws_kind'] = classify_host(ws_url)
    report['rest_kind'] = classify_host(rest_url)
    report['sampling_error'] = error
    mismatch = host_mismatch(ws_url, rest_url)
    if mismatch:
        report['findings'] = list(report['findings']) + ['mixed_venue']
        report['status'] = 'suspect'
        report['mismatch'] = mismatch
    try:
        prices = await api.price_snapshot()
        last = prices.get(symbol)
    except Exception as exc:
        last = None
        report['price_error'] = repr(exc)
    if last is not None:
        refined = verdict(summarise(samples), last, elapsed)
        report['deviation_bps'] = refined['deviation_bps']
        for finding in refined['findings']:
            if finding not in report['findings']:
                report['findings'].append(finding)
        if refined['status'] == 'suspect':
            report['status'] = 'suspect'
    report['checked_at'] = int(time.time() * 1000)
    return report
