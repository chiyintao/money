import math
import time
from dataclasses import asdict, dataclass, field
from .ledger import Ledger
from .execution_rules import evaluate_bar_exit
from .margin import MarginModel
from ..core.domain import Fill

@dataclass
class Lot:
    """One fill that opened (or added to) a position.

    A position used to be a single average price and quantity, which made every exit
    all-or-nothing: `close()` popped the whole thing. That is not a limitation the venue
    has -- you can close any fraction of a position at any time -- and it is the precise
    shape the labelling scheme needs, because a triple-barrier label is defined by
    scaling out at the upper barrier and letting the rest run.

    Keeping the lots also makes the accounting checkable: each one carries the fee that
    was actually charged for it, so a partial exit can charge exactly its share instead
    of allocating a blended average.
    """
    price:float; qty:float; fee:float=0.0; fill_id:str=''; at:int=0

@dataclass
class Position:
    symbol:str; side:str; qty:float; entry:float; stop:float; target:float; opened_at:int=0; funding_paid:float=0.0; order_id:str=''; entry_fee:float=0.0
    # The levels the trade was planned with. `stop` and `target` are the levels it
    # has now, and the exit policy moves them: breakeven_at_rr raises the stop to the
    # entry once the trade is 0.6R in profit, and the trailing rule follows the best
    # price after that. Recording only the live levels made every trade that reached
    # breakeven look like it was opened with a stop exactly at its entry -- the
    # dashboard showed a stop price identical to the entry price and read as broken,
    # while the number it was showing was the correct final stop.
    initial_stop:float=0.0; initial_target:float=0.0
    # Every fill that built this position, oldest first. `qty`/`entry`/`entry_fee` are
    # maintained as their sums so nothing that reads a position has to know about lots.
    lots:list=field(default_factory=list)
    def add_lot(self,price,qty,fee=0.0,fill_id='',at=0):
        """Fold one fill in, keeping the weighted average and the running fee total."""
        total=self.qty+qty
        if total>0: self.entry=(self.entry*self.qty+price*qty)/total
        self.qty=total; self.entry_fee+=fee
        self.lots.append(Lot(float(price),float(qty),float(fee),fill_id,int(at)))
    def take_lots(self,quantity):
        """Remove `quantity` from the front, returning the lots actually consumed.

        FIFO, which is what the venue does by default and what makes the realised pnl of a
        partial exit a fact rather than an allocation choice. Consuming across lots keeps
        the remainder's average entry correct, and the consumed lots carry their own fees
        so the exit can charge its true share.
        """
        if not self.lots and self.qty>0:
            # A snapshot written before lots existed. Synthesise one from the stored
            # average so a restored position can still be closed, partially or not.
            self.lots.append(Lot(self.entry,self.qty,self.entry_fee,self.order_id,self.opened_at))
        remaining=float(quantity); taken=[]
        while remaining>1e-12 and self.lots:
            lot=self.lots[0]
            if lot.qty<=remaining+1e-12:
                taken.append(lot); remaining-=lot.qty; self.lots.pop(0)
            else:
                share=remaining/lot.qty
                taken.append(Lot(lot.price,remaining,lot.fee*share,lot.fill_id,lot.at))
                lot.qty-=remaining; lot.fee-=lot.fee*share; remaining=0.0
        consumed=sum(lot.qty for lot in taken)
        if consumed>0:
            self.entry_fee=max(0.0,self.entry_fee-sum(lot.fee for lot in taken))
        self.qty=max(0.0,self.qty-consumed)
        return taken

#: The field names restore_position accepts, resolved once at import.
_POSITION_FIELDS = frozenset(Position.__dataclass_fields__)

@dataclass
class PaperAccount:
    cash: float = 10000
    fee_rate: float = .0004
    slippage_bps: float = 2.0
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)
    margin: MarginModel = field(default_factory=MarginModel)
    liquidations: list = field(default_factory=list)
    funding_bucket: int = -1
    initial_cash: float = 10000
    equity_curve_limit: int = 20000
    funding_interval_ms: int = 8 * 60 * 60 * 1000
    funding_missed: int = 0
    funding_missing: list = field(default_factory=list)
    funding_settlements: list = field(default_factory=list)
    funding_buckets: dict = field(default_factory=dict)
    bankruptcies: int = 0
    # The venue's mark price, kept apart from `marks`. They are different numbers with
    # different jobs: `marks` holds the last traded price and decides whether a stop or a
    # target was touched, while the mark price is what Binance settles funding on and what
    # maintenance margin is measured against. `marks` was written by four different
    # sources (mark events, trade events, book mid, bar close), so funding was charged on
    # whichever arrived last -- a trade price, usually. Where the venue has not given a
    # mark price, this falls back to the last price, which is the old behaviour.
    mark_prices:dict=field(default_factory=dict)
    # Whether a tick that is not a bar boundary may trigger a stop or a target. It is the
    # difference between two defensible systems and not a bug either way, so it is a
    # switch rather than a decision buried in the call graph:
    #
    #   False  a bar must close through the level (freqtrade's rule, and what the backtest
    #          does). The live path and the backtest then agree, which is the property that
    #          makes the backtest evidence about anything.
    #   True   the first tick through the level closes the position. Closer to a resting
    #          stop order, but the 2-second mark pump and the 5-minute bar loop then
    #          disagree about which one fired, and a wick ends a position a bar would not.
    #
    # Default False: the audit measured a median holding time of 0.93 bars with a 40bp
    # stop against 5.6-20bp of 5m movement, which is the signature of tick-triggered exits
    # firing on noise. Continuous-tick exits remain available for anyone who wants them.
    exit_on_tick:bool=False
    @property
    def equity(self): return self.cash+self.unrealized_pnl()
    def mark_price(self,symbol,fallback=None):
        """The venue mark price, or the last price where none has been seen."""
        value=self.mark_prices.get(symbol)
        if value is None: value=self.marks.get(symbol)
        return fallback if value is None else value
    def unrealized_pnl(self): return sum((self.mark_price(s,p.entry)-p.entry)*p.qty*(1 if p.side=='LONG' else -1) for s,p in self.positions.items())
    def available_margin(self): return self.equity-sum(self.margin.initial_margin(self.mark_price(s,p.entry)*p.qty) for s,p in self.positions.items())
    def open(self,approval,plan,timestamp=0):
        if not approval.get('approved') or plan['symbol'] in self.positions: return False
        execution_price=plan.get('execution_price')
        if execution_price is None:
            slip=plan['entry']*(self.slippage_bps/10000); execution_price=plan['entry']+slip if plan['side']=='LONG' else plan['entry']-slip
        entry=float(execution_price); qty=approval['quantity']
        # A broker fill already contains execution price and fee.
        fee=float(plan.get('fee', entry*qty*self.fee_rate))
        if self.available_margin()<self.margin.initial_margin(entry*qty)+fee: return False
        self.cash-=fee; self.ledger.add('fee',-fee,plan['symbol'])
        position=Position(plan['symbol'],plan['side'],qty,entry,plan['stop'],plan['take_profit'],timestamp,0.0,plan.get('order_id',''),fee)
        position.initial_stop=float(plan['stop'])
        position.initial_target=float(plan['take_profit'])
        position.lots.append(Lot(entry,qty,fee,plan.get('order_id',''),timestamp))
        self.positions[plan['symbol']]=position; self.marks[plan['symbol']]=entry
        self.ledger.add('open',0,plan['symbol']); return True
    def open_fill(self, fill: Fill, plan, timestamp=0):
        import math
        if (not all(math.isfinite(v) for v in (fill.quantity, fill.price, fill.fee))
                or fill.quantity <= 0 or fill.price <= 0 or fill.fee < 0):
            return False
        side = 'LONG' if fill.side in ('BUY', 'LONG') else 'SHORT'
        if plan.get('side') != side:
            return False
        existing = self.positions.get(fill.symbol)
        if existing is not None:
            if existing.order_id != fill.order_id or existing.side != side:
                return False
            if self.available_margin() < self.margin.initial_margin(fill.price*fill.quantity)+fill.fee:
                return False
            # The weighted average is now a consequence of the lot list rather than a
            # second implementation of it, so the two cannot disagree.
            existing.add_lot(fill.price, fill.quantity, fill.fee, fill.order_id,
                             timestamp or fill.event_time)
            self.cash -= fill.fee
            self.ledger.add('fee', -fill.fee, fill.symbol, fill.order_id)
            self.ledger.add('partial_fill', 0, fill.symbol, fill.order_id)
            return True
        fill_plan={**plan,'symbol':fill.symbol,'execution_price':fill.price,'fee':fill.fee,'order_id':fill.order_id}
        return self.open({'approved':True,'quantity':fill.quantity}, fill_plan, timestamp or fill.event_time)

    def audit(self):
        errors=[]
        if self.cash != self.cash or self.equity != self.equity:
            errors.append('non_finite_balance')
        for symbol,position in self.positions.items():
            if position.qty <= 0: errors.append(f'invalid_quantity:{symbol}')
            if position.entry <= 0: errors.append(f'invalid_entry:{symbol}')
            if symbol not in self.marks: errors.append(f'missing_mark:{symbol}')
        if self.available_margin() < -1e-8: errors.append('negative_available_margin')
        return {'ok':not errors,'errors':errors,'cash':self.cash,'equity':self.equity,'available_margin':self.available_margin()}

    def apply_funding(self,rates,event_time=None,interval_ms=None):
        """Charge funding on open positions, at most once per settlement window.

        Three things were wrong with the previous version and all of them made the paper
        account look better than the real one, because funding is a cost:

        * the window key was advanced *before* the loop, so a single failed market
          snapshot consumed the window and its cost was never charged at all;
        * a symbol with no published rate was charged zero via rates.get(symbol, 0),
          which silently reads "no data" as "no cost";
        * the window length was hardcoded to eight hours, so a venue that settles every
          four (or one) hour had most of its funding dropped.

        An empty rate map now leaves the window open so the next poll retries it. A single
        symbol missing from an otherwise good snapshot is charged nothing but is recorded
        in funding_missing and funding_missed -- blocking the whole account on one
        delisted symbol would freeze funding forever, which is the worse failure.
        """
        event_time=int(event_time if event_time is not None else time.time()*1000)
        if not rates:
            self.funding_missed+=1
            self.funding_missing=['<no_rates>']
            return 0.0
        windows = interval_ms if isinstance(interval_ms, dict) else {}
        default_window=int(self.funding_interval_ms or 8*60*60*1000)
        paid=0.0
        missing=[]
        for symbol,p in list(self.positions.items()):
            window=int(windows.get(symbol) or default_window)
            if window<=0: window=default_window
            bucket=event_time//window
            # Each symbol settles on its own clock. The venue runs 8h funding on most
            # contracts and 4h or 1h on some, so a single account-wide window either
            # charged a 4h contract half as often as it should or charged an 8h contract
            # twice. The key is only advanced when the charge is actually booked.
            if bucket == self.funding_buckets.get(symbol): continue
            if symbol not in rates:
                missing.append(symbol)
                continue
            rate=rates.get(symbol)
            # The venue settles funding on the mark price, not on the last trade.
            amount=self.mark_price(symbol,p.entry)*p.qty*float(rate or 0.0)*(1 if p.side=='LONG' else -1)
            if not math.isfinite(amount): continue
            self.funding_buckets[symbol]=bucket
            self.cash-=amount; p.funding_paid+=amount; self.ledger.add('funding',-amount,symbol); paid+=amount
            self.funding_settlements.append({'symbol':symbol,'rate':float(rate or 0.0),'amount':amount,
                                             'event_time':event_time})
        if missing:
            self.funding_missed+=1
            self.funding_missing=missing[:8]
        else:
            self.funding_missing=[]
        self.funding_bucket=max(self.funding_buckets.values()) if self.funding_buckets else self.funding_bucket
        del self.funding_settlements[:-200]
        return paid
    def mark(self,prices,bars=None,mark_prices=None,record=True):
        """Reprice, apply exits, then check the maintenance requirement.

        Two changes from the previous behaviour, both of which made the account look
        better than the contract it imitates:

        * exits were tested against the last or mark price alone. A bar whose low pierced
          the stop and closed back above it never triggered, so the paper account survived
          every wick that would have taken the real position out. When a closed bar is
          supplied the same conservative OHLC rule the backtest uses is applied, so the
          two paths finally agree about what a stop means -- and they are charged the stop
          price, not the closing price.
        * the exit reason was the single string 'stop_or_target', which makes a stop-out
          and a take-profit indistinguishable in the audit trail. Losing trades and winning
          trades are the two halves of every statistic this system reports.

        ``prices`` is the last traded price and drives stop/target; ``mark_prices`` is the
        venue mark and drives funding, liquidation and reported equity. ``record`` controls
        the equity sample: this method is called from a 2-second mark pump, from every feed
        event, and from the bar loop, so the series was sampled 1 millisecond apart in the
        median case and every annualised figure computed from it was noise. Callers that
        are not a bar boundary pass record=False.
        """
        self.marks.update({k:float(v) for k,v in prices.items()})
        self.mark_prices.update({k:float(v) for k,v in (mark_prices or {}).items()})
        for symbol,p in list(self.positions.items()):
            bar=(bars or {}).get(symbol)
            if bar:
                decision=evaluate_bar_exit(p,bar)
                if decision:
                    self.close(symbol,decision['price'],decision['reason'],
                               reason_detail={'ambiguous':decision['ambiguous'],'source':'bar'})
                    continue
            if not self.exit_on_tick:
                # Bar-confirmed exits only. A tick that is not a bar boundary can move the
                # last price through the stop and back within the same bar, and closing on
                # that is a wick, not a stop-out. This is the freqtrade convention and it
                # is the only way the live path can agree with the backtest, which sees
                # bars and nothing else.
                continue
            price=self.marks.get(symbol)
            if price is None: continue
            if p.side=='LONG' and price<=p.stop: self.close(symbol,price,'stop_loss',reason_detail={'source':'tick','confirmed_by':'none'})
            elif p.side=='LONG' and price>=p.target: self.close(symbol,price,'take_profit',reason_detail={'source':'tick','confirmed_by':'none'})
            elif p.side=='SHORT' and price>=p.stop: self.close(symbol,price,'stop_loss',reason_detail={'source':'tick','confirmed_by':'none'})
            elif p.side=='SHORT' and price<=p.target: self.close(symbol,price,'take_profit',reason_detail={'source':'tick','confirmed_by':'none'})
        self.settle_liquidation()
        if record:
            self.equity_curve.append(self.equity)
            # Unbounded growth here used to make every state snapshot megabytes wide.
            if len(self.equity_curve) > self.equity_curve_limit:
                del self.equity_curve[:len(self.equity_curve)-self.equity_curve_limit]
    def close(self,symbol,price,reason='manual',timestamp=0,reason_detail=None,quantity=None):
        """Close all or part of a position, and return the realised pnl.

        This used to be all-or-nothing: the position was popped whole, so a partial exit
        could not be expressed at all. That is not a venue limitation -- any fraction of a
        position can be closed at any time -- and it is exactly the shape a triple-barrier
        label needs, since that label is defined by scaling out at the upper barrier and
        letting the remainder run.

        ``quantity=None`` closes everything and removes the position. A smaller quantity
        consumes lots FIFO, realises that share of the pnl, and leaves the position open
        with the remainder. Each consumed lot carries the entry fee that was actually
        charged for it, so a partial exit charges its true share rather than a blended
        average -- and funding is apportioned by the same ratio, since funding is charged
        on the whole position but only part of it is leaving.
        """
        p=self.positions.get(symbol)
        if p is None: return 0.0
        closed_at=int(timestamp or time.time()*1000)
        slip=price*(self.slippage_bps/10000)
        exit_price=price-slip if p.side=='LONG' else price+slip
        fraction=1.0 if quantity is None else max(0.0,min(1.0,float(quantity)/p.qty if p.qty else 0.0))
        # A zero-quantity request is a caller bug, not a way to reset the position.
        if fraction<=0.0: return 0.0
        closing_qty=p.qty*fraction
        consumed=p.take_lots(closing_qty) if fraction<1.0 else p.take_lots(p.qty)
        entry_notional=sum(lot.price*lot.qty for lot in consumed)
        entry_fee=sum(lot.fee for lot in consumed)
        average_entry=entry_notional/closing_qty if closing_qty else p.entry
        gross_pnl=(exit_price-average_entry)*closing_qty*(1 if p.side=='LONG' else -1)
        exit_fee=exit_price*closing_qty*self.fee_rate
        # A liquidation is not an ordinary exit and the venue does not price it as one:
        # it charges its own fee on top of the taker fee. The rate was defined on the
        # margin model and never read, so every liquidation was undercharged.
        liquidation_fee=0.0
        if reason=='liquidation':
            liquidation_fee=abs(exit_price*closing_qty)*float(getattr(self.margin,'liquidation_fee_rate',0.0) or 0.0)
            exit_fee+=liquidation_fee
        # Funding was charged on the whole position, so only the departing share of it
        # belongs to this trade; the rest stays with the position that is still open.
        funding_share=p.funding_paid*fraction
        p.funding_paid-=funding_share
        total_fees=exit_fee+entry_fee
        pnl=gross_pnl-total_fees-funding_share
        margin_used=self.margin.initial_margin(entry_notional)
        self.cash+=gross_pnl-exit_fee
        self.ledger.add('close',gross_pnl-exit_fee,symbol,p.order_id)
        if liquidation_fee:
            self.ledger.add('liquidation_fee',-liquidation_fee,symbol,p.order_id)
        partial=fraction<1.0
        if not partial:
            self.positions.pop(symbol,None)
            self.marks.pop(symbol,None)
            self.mark_prices.pop(symbol,None)
        self.trades.append({
            'trade_id':symbol+'-'+str(p.opened_at)+'-'+str(len(self.trades)),
            'order_id':p.order_id,'symbol':symbol,'side':p.side,'qty':closing_qty,
            'entry':average_entry,'exit':exit_price,'entry_price':average_entry,'exit_price':exit_price,
            'entry_time':p.opened_at,'exit_time':closed_at,'holding_ms':max(0,closed_at-p.opened_at) if p.opened_at else 0,
            'entry_notional':entry_notional,'exit_notional':exit_price*closing_qty,'margin_used':margin_used,
            'gross_pnl':gross_pnl,'pnl':pnl,'pnl_pct':pnl/margin_used*100 if margin_used else 0.0,
            'entry_fee':entry_fee,'exit_fee':exit_fee,'fees':total_fees,'funding':funding_share,
            'stop_price':p.stop,'take_profit_price':p.target,
            'initial_stop':p.initial_stop,'initial_target':p.initial_target,
            'leverage':self.margin.leverage,'reason':reason,
            'reason_detail':dict(reason_detail or {}),'liquidation_fee':liquidation_fee,
            # Whether this closed the position or only part of it, and what is left.
            'partial':partial,'remaining_qty':0.0 if not partial else p.qty,'lots_consumed':len(consumed),
        })
        return pnl

    def settle_liquidation(self):
        """Flatten the account when equity falls through its maintenance requirement.

        The previous version closed each position at the mark and appended a record, and
        did nothing about the result: cash was allowed to go negative, so an account could
        keep running with a balance no venue would permit, and every later metric was
        measured from a number that could not exist.

        A real account is made whole by the insurance fund rather than by the trader, so
        the paper account is written down to zero and the shortfall is recorded explicitly.
        Losing more than the deposit is possible; pretending it did not happen is not.
        """
        liquidate,maintenance,detail=self.margin.should_liquidate(self.equity,self.positions,self.marks)
        if not liquidate:
            return None
        equity_before=self.equity
        for symbol in list(self.positions):
            # A liquidation is executed at the mark price, which is what the maintenance
            # requirement was measured against.
            self.close(symbol,self.mark_price(symbol,self.positions[symbol].entry),'liquidation',
                       reason_detail={'maintenance':maintenance})
        shortfall=0.0
        if self.cash < 0:
            shortfall=-self.cash
            self.cash=0.0
            self.ledger.add('bankruptcy_shortfall',-shortfall,'<account>')
        record={'event_time':int(time.time()*1000),'equity_before':equity_before,
                'maintenance':maintenance,'shortfall':shortfall,'detail':detail,
                'symbols':list(detail)}
        self.liquidations.append(record)
        if shortfall:
            self.bankruptcies+=1
        return record
    def snapshot(self):
        """Everything needed to rebuild this account, as plain JSON-able data.

        A mapping rather than one expression per field: the single-line form was eleven
        hundred characters of punctuation that no reviewer could diff, and it is the
        write half of the only format that has to stay in step with restore().
        """
        return {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'fee_rate': self.fee_rate,
            'slippage_bps': self.slippage_bps,
            'positions': [asdict(p) for p in self.positions.values()],
            'marks': self.marks,
            'mark_prices': self.mark_prices,
            'ledger': self.ledger.snapshot(),
            'liquidations': self.liquidations,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'funding_bucket': self.funding_bucket,
            'funding_interval_ms': self.funding_interval_ms,
            'funding_missed': self.funding_missed,
            'funding_missing': self.funding_missing,
            'funding_settlements': self.funding_settlements,
            'funding_buckets': self.funding_buckets,
            'bankruptcies': self.bankruptcies,
        }
    @classmethod
    def restore_position(cls, payload):
        """Rebuild one position from a snapshot, lots included.

        `asdict()` is recursive. It turned each Lot into a plain dict as well as the
        Position around them, and `restore` only rebuilt the outer object -- so after
        any restart `position.lots` was a list of dicts while every method on Position
        reads `lot.qty`. `take_lots` raised AttributeError on the first read, which
        means every exit a restored position could take was broken at once: the stop,
        the target, the exit policy and the manual close button all end in close(),
        and close() calls take_lots. The effect an operator saw was a close button
        that answered HTTP 500 and a position that survived every attempt to flatten
        it, with the real reason only in the server log.
        """
        data = dict(payload or {})
        raw_lots = data.pop("lots", None) or []
        position = Position(**{key: value for key, value in data.items()
                              if key in _POSITION_FIELDS})
        fields = set(Lot.__dataclass_fields__)
        position.lots = [
            lot if isinstance(lot, Lot) else Lot(**{k: v for k, v in lot.items()
                                                    if k in fields})
            for lot in raw_lots]
        return position

    @classmethod
    def restore(cls, data, default_cash=10000):
        """Rebuild an account from a snapshot.

        Every field is listed here rather than inferred, because a name that is in
        snapshot() and missing here is silently dropped on every restart -- which is
        exactly how the position lots came back as dicts and took every exit with them.
        """
        data = data or {}
        account = cls(float(data.get('cash', default_cash)),
                      float(data.get('fee_rate', .0004)),
                      float(data.get('slippage_bps', 2)))
        account.initial_cash = float(data.get('initial_cash', default_cash))
        account.positions = {p['symbol']: cls.restore_position(p)
                             for p in data.get('positions', [])}
        account.marks = {k: float(v) for k, v in data.get('marks', {}).items()}
        account.mark_prices = {k: float(v)
                               for k, v in (data.get('mark_prices') or {}).items()}
        account.ledger = Ledger.restore(data.get('ledger'))
        account.liquidations = list(data.get('liquidations', []))
        account.funding_bucket = int(data.get('funding_bucket', -1))
        account.trades = list(data.get('trades', []))
        account.equity_curve = list(data.get('equity_curve', []))
        account.funding_interval_ms = int(
            data.get('funding_interval_ms', 8 * 60 * 60 * 1000) or 8 * 60 * 60 * 1000)
        account.funding_missed = int(data.get('funding_missed', 0) or 0)
        account.funding_missing = list(data.get('funding_missing', []) or [])
        account.funding_settlements = list(data.get('funding_settlements', []) or [])
        account.funding_buckets = {str(k): int(v)
                                   for k, v in (data.get('funding_buckets') or {}).items()}
        account.bankruptcies = int(data.get('bankruptcies', 0) or 0)
        return account
