from dataclasses import dataclass, field
from ..core.domain import Fill, OrderIntent, PaperOrder, validate_order
from .fill_models import DEFAULT_MAKER_RATE, DEFAULT_TAKER_RATE, MAKER, TAKER, FeeModel, FillModel, participation
from ..core import order_state
import uuid

# Re-exported for the callers that imported them from here; each definition lives in one
# place. ContractSpec sits in core so the venue layer can build one without importing the
# order layer -- it was the edge that made `market` depend on `broker`.
from ..core.instruments import ContractSpec  # noqa: E402  (re-export, kept next to its users)
OPEN_STATUSES = order_state.OPEN_STATUSES

@dataclass
class PaperBroker:
    # fee_rate is the taker rate; it stays a field because callers read it to price a round
    # trip. The maker rate, the spread and the impact term live in the models below.
    fee_rate: float=DEFAULT_TAKER_RATE; slippage_bps: float=2.0; specs: dict=field(default_factory=dict); orders: dict=field(default_factory=dict); order_sink: object=None; order_ttl_ms: int=90000; open_by_symbol: dict=field(default_factory=dict); transition_refusals: dict=field(default_factory=dict)
    # The decomposed matching models. None means "build from fee_rate/slippage_bps", so
    # every existing construction keeps its old numbers while every new one can state its
    # assumptions separately.
    fee_model: object=None; fill_model: object=None; maker_fee_rate: float=DEFAULT_MAKER_RATE
    # Length of one bar, in milliseconds. Latency is charged as a share of a bar's
    # volatility, so the model has to know how long a bar is; 5 minutes is the default
    # interval and the only one the service runs.
    bar_ms: int=300_000

    def _bar_ms(self): return float(self.bar_ms or 300_000)

    def __post_init__(self):
        if self.fee_model is None:
            self.fee_model = FeeModel(maker_rate=float(self.maker_fee_rate),
                                      taker_rate=float(self.fee_rate))
        if self.fill_model is None:
            self.fill_model = FillModel(spread_bps=float(self.slippage_bps))
        # One source of truth from here on: the models own the numbers, and these two
        # fields mirror them for the callers that price a round trip from them.
        self.fee_rate = float(self.fee_model.taker_rate)
        self.slippage_bps = float(self.fill_model.spread_bps)

    def cost_model(self):
        """What this account assumes about costs, for the dashboard and the audit trail."""
        return {"fee": self.fee_model.describe(), "fill": self.fill_model.describe()}
    def spec(self,symbol): return self.specs.get(symbol,ContractSpec(symbol))

    def advance(self, order, status, timestamp=None, filled_quantity=None):
        """Move an order to a new status through the lifecycle table, or refuse.

        Every state change used to build a fresh PaperOrder with whatever status the
        caller passed, so the table in order_state was documentation: it described the
        lifecycle, nothing consulted it, and an illegal move was written to the book and
        to the audit trail exactly like a legal one. This is the one place that applies
        it. A refusal returns (None, reason) so the caller reports it instead of
        inventing a state the rest of the system has to cope with.
        """
        current = getattr(order, 'status', None)
        if current == status and (filled_quantity is None
                                  or float(filled_quantity) == float(order.filled_quantity)):
            # The no-op a late book event produces. A repeated partial fill is different:
            # the status is the same and the quantity is not, so it is written.
            return order, 'already_' + str(status).lower()
        now = int(timestamp if timestamp is not None else __import__('time').time_ns() // 1_000_000)
        updates = {'updated_at': now}
        if filled_quantity is not None:
            updates['filled_quantity'] = float(filled_quantity)
        # The legality check and the rebuild both live in order_state: one definition of
        # the lifecycle, applied here. Copying the table into the broker is what let the
        # two fill paths disagree about what an order is allowed to become.
        updated, reason = order_state.transition(order, status, updates)
        if updated is order or reason not in ('ok',):
            if reason.startswith('illegal_transition') or reason.startswith('unknown_status'):
                key = reason.split(':', 1)[-1]
                self.transition_refusals[key] = self.transition_refusals.get(key, 0) + 1
            return None, reason
        self.track_order(updated)
        if self.order_sink: self.order_sink(updated.__dict__)
        return updated, 'ok'

    def track_order(self, order):
        """The single write path for self.orders, keeping the per-symbol index honest.

        open_orders() used to scan every order ever created, and the websocket handler
        called it once per book event, so a few hundred dead orders made every tick
        quadratic. The index is maintained here instead of being recomputed.
        """
        self.orders[order.order_id]=order
        ids=self.open_by_symbol.setdefault(order.symbol,set())
        if order.status in OPEN_STATUSES: ids.add(order.order_id)
        else: ids.discard(order.order_id)
        return order

    def submit(self, intent: OrderIntent, timestamp=None, plan=None):
        valid,reason=validate_order(intent)
        if not valid: return None,reason
        spec=self.spec(intent.symbol)
        # The venue only accepts quantities on its step grid and prices on its tick grid.
        # Submitting the raw size created orders that could never fill: a quantity below one
        # step rounds to zero inside execute(), which returns below_min_notional forever, so
        # the order rested until its deadline while holding a position slot and a risk
        # reservation. Normalising here makes that a rejection at submit instead.
        quantity=spec.round_qty(intent.quantity)
        limit_price=spec.round_price(intent.limit_price) if intent.order_type=='LIMIT' else 0.0
        requested=float(intent.quantity)
        plan=dict(plan or {})
        if quantity<=0:
            return None,'below_step_size'
        reference=limit_price if intent.order_type=='LIMIT' and limit_price else float(plan.get('price') or 0)
        if reference and quantity*reference<spec.min_notional:
            return None,'below_min_notional'
        order_id=intent.order_id or uuid.uuid4().hex
        now=int(timestamp or __import__('time').time_ns()//1_000_000)
        # Every order carries a deadline. A plan describes one bar's price; letting it
        # rest indefinitely is what queued orders behind a single open position and
        # filled them hours later against a stop and target from the original bar.
        expires_at=int(plan.get('expires_at') or 0) or (now+self.order_ttl_ms if self.order_ttl_ms>0 else 0)
        if quantity < requested - 1e-12:
            # Recorded rather than silently rounded, because a strategy asking for a size the
            # venue cannot express is a sizing bug that would otherwise be invisible.
            plan['quantity_reduced_from']=requested
        order=PaperOrder(order_id,intent.symbol,intent.side,quantity,intent.order_type,limit_price,0.0,order_state.OPEN,now,now,intent.reduce_only,plan,expires_at)
        self.track_order(order)
        if self.order_sink: self.order_sink(order.__dict__)
        return order,'accepted'

    def expire_orders(self, now=None):
        now=int(now or __import__('time').time_ns()//1_000_000)
        expired=[]
        for order in list(self.open_orders()):
            if order.expires_at and now >= order.expires_at:
                updated,reason=self.cancel(order.order_id,now,status=order_state.EXPIRED)
                if updated is not None and reason=='expired': expired.append(updated)
        return expired

    def open_orders(self, symbol=None):
        if symbol is None:
            return [order for order in self.orders.values() if order.status in OPEN_STATUSES]
        return [self.orders[oid] for oid in self.open_by_symbol.get(symbol, ())
                if oid in self.orders and self.orders[oid].status in OPEN_STATUSES]

    def order_counts(self):
        """Working and terminal counts, including the fill rate nothing used to compute."""
        return order_state.summarise(self.orders.values())

    def has_open_order(self, symbol):
        """Whether this symbol already has a working order an entry would queue behind."""
        return any(self.orders[oid].status in OPEN_STATUSES
                   for oid in self.open_by_symbol.get(symbol, ()) if oid in self.orders)

    def cancel(self, order_id, timestamp=None, status=None):
        """Cancel an order, or mark it expired.

        One function with two endings because the alternative -- what this used to do -- was
        to write CANCELED for both, which made a deadline expiry indistinguishable from a
        decision to stop. The audit trail is where "why did this order disappear" is
        answered, and it could not answer it.
        """
        order=self.orders.get(order_id)
        if order is None: return None,'unknown_order'
        if not order_state.is_open(order.status): return order,'order_not_open'
        updated,reason=self.advance(order,status or order_state.CANCELED,timestamp)
        if updated is None: return None,reason
        return updated,order_state.outcome(updated.status)

    def execute_depth(self, intent: OrderIntent, bids=None, asks=None, volatility_bps=None):
        valid,reason=validate_order(intent)
        if not valid: return None,reason
        levels=asks if intent.side in ('BUY','LONG') else bids
        if not levels: return None,'no_liquidity'
        remaining=float(intent.quantity); value=0.0; filled=0.0
        for level in levels:
            price=float(level[0]); depth=float(level[1])
            take=min(remaining,depth)
            if take<=0: continue
            value+=take*price; filled+=take; remaining-=take
            if remaining<=1e-12: break
        if filled<=0: return None,'no_liquidity'
        if remaining>1e-12: return None,'insufficient_depth'
        average=value/filled
        spec=self.spec(intent.symbol); quantity=spec.round_qty(filled); price=spec.round_price(average)
        if quantity<=0 or quantity*price<spec.min_notional: return None,'below_min_notional'
        # Walking the book is the definition of taking liquidity, so the class is not in
        # question here -- but the participation is, and it is measurable: the share of the
        # ladder this order consumed. It used to be hardcoded to 'taker' and the size was
        # not passed to any model at all.
        depth=sum(float(level[1]) for level in levels)
        # Walking the ladder is a taker fill, so it pays the spread and the impact too --
        # this path used to return the raw depth average with no cost model applied at
        # all, which made a book-walk fill strictly cheaper than the identical quote fill
        # decided one branch over.
        price = spec.round_price(self.fill_model.price(
            average, intent.side, TAKER, participation=participation(quantity,depth),
            notional=average*quantity, volatility_bps=volatility_bps,
            bar_ms=self._bar_ms(), include_spread=False, include_impact=False))
        if price<=0 or quantity*price<spec.min_notional: return None,'below_min_notional'
        fee=self.fee_model.fee(price*quantity, TAKER)
        return Fill(intent.order_id or uuid.uuid4().hex,intent.symbol,intent.side,quantity,price,fee,liquidity=TAKER,event_time=__import__('time').time_ns()//1_000_000,participation=participation(quantity,depth)),'filled'

    def process(self, order_id, market_price: float, bid: float|None=None, ask: float|None=None, available_qty=None, accept_fill=None, timestamp=None, volatility_bps=None):
        order=self.orders.get(order_id)
        if order is None: return [],'unknown_order'
        if not order_state.is_open(order.status): return [],'order_not_open'
        remaining=order.quantity-order.filled_quantity
        if remaining <= 0: return [],'filled'
        reference=ask if order.side in ('BUY','LONG') and ask is not None else bid if order.side in ('SELL','SHORT') and bid is not None else market_price
        if order.order_type=='LIMIT':
            executable=(order.side in ('BUY','LONG') and reference<=order.limit_price) or (order.side in ('SELL','SHORT') and reference>=order.limit_price)
            if not executable: return [],'not_marketable'
        qty=remaining if available_qty is None else min(remaining,float(available_qty))
        if qty<=0: return [],'no_liquidity'
        intent=OrderIntent(order.symbol,order.side,qty,'MARKET',order.reduce_only,order.limit_price,0,'baseline-v1',order.order_id)
        # Who provided liquidity. A limit order reaching this point has been resting on the
        # book -- that is what process() is for -- so when it fills it is hit at its own
        # price and pays the maker rate. A market order crosses. The one case that makes a
        # limit order a taker is being marketable the instant it arrives, which is only
        # knowable when the caller supplies a timestamp equal to the order's creation.
        if order.order_type=='LIMIT':
            same_instant = timestamp is not None and int(timestamp) <= int(order.created_at or 0)
            liquidity = TAKER if same_instant else MAKER
        else:
            liquidity = TAKER
        fill,reason=self.execute(intent,market_price,bid,ask,
                                 apply_slippage=(liquidity==TAKER),timestamp=timestamp,
                                 liquidity=liquidity,
                                 participation_share=participation(qty,available_qty),
                                 volatility_bps=volatility_bps)
        if fill is None: return [],reason
        if accept_fill is not None and not accept_fill(fill):
            return [],'account_rejected_fill'
        filled=order.filled_quantity+fill.quantity
        status=order_state.status_for_fill(order.quantity,filled)
        updated,reason=self.advance(order,status,fill.event_time,filled_quantity=filled)
        if updated is None:
            # The fill happened at the venue and the book cannot record it. Returning the
            # fill would book a trade the order does not show; dropping it silently would
            # leave the account holding a position nothing opened. Refused and reported.
            return [],reason
        return [fill],status

    def execute(self, intent: OrderIntent, market_price: float, bid: float|None=None, ask: float|None=None, apply_slippage=True, timestamp=None, liquidity=None, participation_share=None, volatility_bps=None):
        valid,reason=validate_order(intent)
        if not valid: return None,reason
        spec=self.spec(intent.symbol); qty=spec.round_qty(intent.quantity); price=intent.limit_price if intent.order_type=='LIMIT' else (ask if intent.side in ('BUY','LONG') and ask else bid if intent.side in ('SELL','SHORT') and bid else market_price)
        if intent.order_type=='LIMIT' and ((intent.side in ('BUY','LONG') and price>market_price) or (intent.side in ('SELL','SHORT') and price<market_price)): return None,'limit_not_marketable'
        # Which side of the spread this fill was on decides both its price and its fee,
        # and nothing used to say. A caller that does not know defaults to taker, which is
        # the conservative reading -- but a resting order that gets hit is a maker fill and
        # charging it the taker rate overstates its cost by the whole fee differential.
        book = TAKER if apply_slippage else MAKER
        if liquidity is None: liquidity = book
        # Latency is charged through the same call as spread and impact, so there is one
        # place where a fill's price is decided and no path can quietly skip a cost.
        fill_price = self.fill_model.price(price, intent.side, liquidity,
                                           participation=participation_share,
                                           notional=float(price)*qty,
                                           volatility_bps=volatility_bps,
                                           bar_ms=self._bar_ms())
        fill_price=spec.round_price(fill_price)
        fee=self.fee_model.fee(fill_price*qty, liquidity)
        if qty<=0 or fill_price*qty<spec.min_notional: return None,'below_min_notional'
        order_id=intent.order_id or uuid.uuid4().hex
        event_time=int(timestamp if timestamp is not None else __import__('time').time_ns()//1_000_000)
        # The label travels with the fill. It was computed, used for the fee, and then
        # dropped here, so every fill in the audit trail read 'taker' while half of them
        # had been charged the maker rate -- a record that disagreed with its own fee.
        return Fill(order_id,intent.symbol,intent.side,qty,fill_price,fee,liquidity=liquidity,
                    event_time=event_time,
                    participation=float(participation_share or 0.0)), 'filled'

    def liquidation_price(self, side, entry, leverage, maintenance_rate):
        buffer=1/leverage-maintenance_rate
        return entry*(1-buffer) if side in ('BUY','LONG') else entry*(1+buffer)
