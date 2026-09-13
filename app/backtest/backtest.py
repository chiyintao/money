"""Offline backtests that reuse the live sizing and execution rules.

A backtest is only useful if it predicts the service. These runs used to construct
their own RiskEngine with hardcoded values (including a 2% daily loss against the
service's 20%) and to fill entries without the re-anchoring that settlement applies, so
a result here said nothing about what the paper account would actually do. Both entry
paths now go through the same RiskEngine, PortfolioLimits and plan adjustment the live
service uses.
"""
import argparse, json
from bisect import bisect_right
from ..core.config import Settings
from ..core.metrics import summary
from ..trading.risk import RiskEngine
from ..trading.simulation import PaperAccount
from ..trading.strategy import predict, plan
from ..trading.broker import PaperBroker
from ..core.domain import OrderIntent
from ..trading.guards import portfolio_limits
from ..trading.execution import reanchor_plan
from ..features.quality import validate_ohlcv
from ..trading.execution_rules import evaluate_bar_exit
from ..core.engine import DeterministicEventEngine, candle_events


def rule_strategy(fee_rate=.0004, slippage_bps=2):
    """Legacy rule baseline, kept only for A/B comparison against the real models."""
    def decide(symbol, history):
        return plan(predict(symbol, history), fee_rate, slippage_bps)
    return decide


def model_strategy(decisions):
    """Offline decisions produced by the same real trained weights the paper service uses."""
    def decide(symbol, history):
        return decisions.plan(decisions.signal_sync(symbol, history))
    return decide


def live_risk(starting_equity, settings=None):
    """A RiskEngine configured exactly like the paper service's."""
    settings=settings or Settings()
    return RiskEngine(settings.max_risk_per_trade, settings.max_daily_loss, settings.max_gross_leverage,
                      starting_equity, max_portfolio_risk=settings.max_portfolio_risk,
                      target_exposure=settings.target_exposure,
                      max_symbol_leverage=settings.max_symbol_leverage,
                      sizing_reference_edge_bps=settings.sizing_reference_edge_bps)


def live_limits(settings=None):
    return portfolio_limits(settings or Settings())


def portfolio_state(account, symbol):
    """Open notional by symbol, the risk already committed, and this symbol's share."""
    notional_by_symbol={name:position.entry*position.qty for name,position in account.positions.items()}
    open_risk=sum(abs(position.entry-position.stop)*position.qty
                  for position in account.positions.values() if position.stop)
    return notional_by_symbol, open_risk, notional_by_symbol.get(symbol,0.0)




class HistoryView:
    """A read-only prefix of a bar series, built in constant time.

    The entry loop asks for the history up to the current bar once per (timestamp,
    symbol) pair. Slicing the list answered that with a fresh list of up to a hundred
    thousand pointers: 1.26 million allocations and roughly 650 GB of copying for one
    year across eleven symbols, which is where the twelve minutes of the portfolio
    replay went. This exposes the same sequence interface over the same underlying
    list, so every strategy keeps working unchanged, and building it costs nothing.
    """

    __slots__ = ("_rows", "_end")

    def __init__(self, rows, end):
        self._rows = rows
        self._end = end

    def __len__(self):
        return self._end

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._rows[:self._end][index]
        if index < 0:
            index += self._end
        if index < 0 or index >= self._end:
            raise IndexError(index)
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows[:self._end])

    def __repr__(self):
        return "HistoryView(%d of %d)" % (self._end, len(self._rows))

def fill_entry(account, broker, risk, limits, symbol, plan_row, entry_price, timestamp):
    """Size, gate and fill one pending entry the way the paper service does.

    Returns the fill on success, otherwise None. The stop and target are re-anchored to
    the price that actually traded, matching settle_order live, so a backtest and the
    service hold positions with the same geometry.
    """
    notional_by_symbol, open_risk, symbol_notional = portfolio_state(account, symbol)
    decision=risk.approve(plan_row, account.equity, sum(notional_by_symbol.values()), timestamp,
                          open_risk=open_risk, symbol_notional=symbol_notional)
    allowed,_=limits.approve(symbol, decision.get('notional',0.0), notional_by_symbol, account.equity)
    if not decision.get('approved') or not allowed:
        return None
    intent=OrderIntent(symbol,'BUY' if plan_row['side']=='LONG' else 'SELL',
                       decision.get('quantity',0), order_id=f'{symbol}-{timestamp}')
    fill,_=broker.execute(intent, entry_price, timestamp=timestamp)
    if not fill:
        return None
    account.open_fill(fill, reanchor_plan(plan_row, fill.price), timestamp)
    return fill


def run_backtest(rows, symbol='TEST', starting_equity=10000, fee_rate=.0004, slippage_bps=2, strategy=None,
                 risk=None, limits=None, specs=None, interval=None):
    strategy=strategy or rule_strategy(fee_rate, slippage_bps)
    quality=validate_ohlcv(rows)
    if not quality['ready']:
        raise ValueError({'reason':'invalid_ohlcv','quality':quality})
    account=PaperAccount(starting_equity, fee_rate, slippage_bps, initial_cash=starting_equity)
    account.equity_curve.append(float(starting_equity))
    broker=PaperBroker(fee_rate=fee_rate, slippage_bps=slippage_bps, specs=specs or {})
    risk=risk or live_risk(starting_equity)
    limits=limits or live_limits()
    pending=None
    for index in range(50, len(rows)):
        bar=rows[index]
        if pending and symbol not in account.positions:
            fill_bar=bar; entry=float(fill_bar.get('open',fill_bar['close']))
            fill_entry(account, broker, risk, limits, symbol, pending, entry, int(fill_bar.get('open_time',index)))
            pending=None
        if symbol in account.positions:
            position=account.positions[symbol]
            exit_decision=evaluate_bar_exit(position,bar)
            if exit_decision:
                account.close(symbol,exit_decision['price'],exit_decision['reason'],int(bar.get('close_time',bar.get('open_time',0))))
        account.mark({symbol:float(bar['close'])})
        history=rows[:index+1]
        candidate=strategy(symbol, history)
        pending=candidate if candidate and symbol not in account.positions else None
    return {'metrics':summary(account.equity_curve,account.trades,initial_equity=starting_equity,interval=interval),'trades':account.trades,'equity_curve':account.equity_curve}


def _rate_at(points, timestamp):
    """The most recently published value at or before this timestamp, or None."""
    if not points:
        return None
    if timestamp < points[0][0]:
        return None
    index = bisect_right(points, (timestamp, float('inf'))) - 1
    return points[index][1] if index >= 0 else None


def run_portfolio_backtest(series, starting_equity=10000, fee_rate=.0004, slippage_bps=2, strategy=None, interval=None,
                           risk=None, limits=None, specs=None, funding=None, funding_windows=None):
    strategy=strategy or rule_strategy(fee_rate, slippage_bps)
    if not series:
        raise ValueError('empty_portfolio')
    for symbol, rows in series.items():
        quality=validate_ohlcv(rows)
        if not quality['ready']:
            raise ValueError({'reason':'invalid_ohlcv','symbol':symbol,'quality':quality})
        if len(rows) < 51:
            raise ValueError({'reason':'insufficient_rows','symbol':symbol,'rows':len(rows)})
    account=PaperAccount(starting_equity, fee_rate, slippage_bps, initial_cash=starting_equity)
    account.equity_curve.append(float(starting_equity))
    broker=PaperBroker(fee_rate=fee_rate, slippage_bps=slippage_bps, specs=specs or {})
    risk=risk or live_risk(starting_equity)
    limits=limits or live_limits()
    pending={}
    funding_paid=0.0
    rows_by_symbol={symbol:{int(row['open_time']):row for row in rows} for symbol,rows in series.items()}
    # The open_time of every bar, per symbol, once. The entry loop needs "the history up to
    # this timestamp" for every (timestamp, symbol) pair, and it used to answer that with a
    # full scan of the symbol's whole series -- 105,120 timestamps x 15 symbols x 105,120
    # rows is 1.7e11 Python-level iterations, about four and a half hours per replay. That
    # quadratic scan is why the portfolio evidence never appeared: the promotion gate waits
    # on it, and it could not finish. The series are sorted by open_time, so the prefix
    # boundary is a binary search and the history is a slice.
    times_by_symbol={symbol:[int(row['open_time']) for row in rows] for symbol,rows in series.items()}
    ordered_events=DeterministicEventEngine().order(candle_events(series))
    timestamps=[]
    for event in ordered_events:
        if not timestamps or timestamps[-1] != event.event_time:
            timestamps.append(event.event_time)
    for timestamp in timestamps:
        bars={symbol:rows.get(timestamp) for symbol,rows in rows_by_symbol.items()}
        for symbol, pending_plan in list(pending.items()):
            bar=bars.get(symbol)
            if bar is None or symbol in account.positions:
                continue
            fill_entry(account, broker, risk, limits, symbol, pending_plan,
                       float(bar.get('open',bar['close'])), timestamp)
            pending.pop(symbol,None)
        for symbol,bar in bars.items():
            if bar is None or symbol not in account.positions:
                continue
            position=account.positions[symbol]
            exit_decision=evaluate_bar_exit(position,bar)
            if exit_decision:
                account.close(symbol,exit_decision['price'],exit_decision['reason'],int(bar.get('close_time',bar.get('open_time',0))))
        # Funding, before the mark so the equity curve at this timestamp carries the cost.
        # The live loop charges it every tick and the account advances each symbol's window
        # only when the charge is actually booked, so passing the whole rate map each bar is
        # the same call shape the service makes -- settlement timing belongs to the account,
        # not to the caller. A backtest without this reports net_return that excludes a real
        # perp cost while claiming costs are included.
        if funding:
            rates = {}
            for symbol in bars:
                rate = _rate_at(funding.get(symbol), timestamp)
                if rate is not None:
                    rates[symbol] = rate
            if rates:
                funding_paid += float(account.apply_funding(rates, timestamp,
                                                            interval_ms=funding_windows) or 0.0)
        account.mark({symbol:float(bar['close']) for symbol,bar in bars.items() if bar is not None})
        for symbol,bar in bars.items():
            if bar is None or symbol in account.positions:
                continue
            history=HistoryView(series[symbol], bisect_right(times_by_symbol[symbol],timestamp))
            if len(history) < 50:
                continue
            candidate=strategy(symbol,history)
            if candidate:
                pending[symbol]=candidate
    return {'metrics':summary(account.equity_curve,account.trades,initial_equity=starting_equity,interval=interval),
            'trades':account.trades,'equity_curve':account.equity_curve,'symbols':list(series),
            'funding':{'charged':funding_paid,
                       'missed':int(account.funding_missed),
                       'missing':list(account.funding_missing),
                       'modelled':bool(funding)}}


def load_rows(path):
    with open(path,encoding='utf-8') as handle: return [json.loads(line) for line in handle if line.strip()]

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('path'); parser.add_argument('--symbol',default='TEST')
    parser.add_argument('--models',action='store_true',help='decide with the real trained weights instead of the rule baseline')
    parser.add_argument('--data-dir',default='data')
    args=parser.parse_args(); strategy=None
    if args.models:
        from ..strategy.live_models import RealModelRuntime, ModelDecision
        decisions=ModelDecision(RealModelRuntime(args.data_dir,chronos_enabled=False),fee_rate=.0004,slippage_bps=2,rule_fallback=False)
        strategy=model_strategy(decisions)
    print(json.dumps(run_backtest(load_rows(args.path),args.symbol,strategy=strategy),ensure_ascii=False))
