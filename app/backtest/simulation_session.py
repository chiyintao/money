from dataclasses import dataclass, field
from time import time
from uuid import uuid4
from ..core.domain import Event
from ..trading.margin import MarginModel
from ..trading.simulation import PaperAccount


@dataclass
class SimulationSession:
    """A restartable, fully-audited paper futures session."""
    store: object
    account: PaperAccount = field(default_factory=PaperAccount)
    session_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "idle"
    source: str = "gainers"
    symbol_count: int = 3
    initial_cash: float = 10000.0
    leverage: float = 3.0
    selected_symbols: list = field(default_factory=list)
    step: int = 0
    started_at: int = 0
    ended_at: int = 0
    end_reason: str = ""
    cooldown_until: dict = field(default_factory=dict)
    last_selection_at: int = 0
    persist_min_interval_ms: int = 1000
    last_persist_at: int = 0
    persist_skipped: int = 0
    snapshot_equity_points: int = 500
    # Last time an order book was seen per symbol, so a symbol whose book never arrives
    # can be retired instead of consuming a position slot indefinitely.
    book_seen_at: dict = field(default_factory=dict)
    last_book_at: int = 0

    def _event(self, kind, payload):
        self.store.record_event(Event(kind, {"session_id": self.session_id, "step": self.step, **payload}).json())

    @classmethod
    def restore(cls, store, data, default_cash=10000):
        if not data: return cls(store, account=PaperAccount(cash=default_cash, initial_cash=default_cash), initial_cash=default_cash)
        account=PaperAccount.restore(data.get('account', {}), default_cash)
        account.margin=MarginModel(leverage=float(data.get('leverage', 3)))
        session = cls(
            store, account=account,
            session_id=data.get('session_id', uuid4().hex),
            status=data.get('status', 'idle'),
            source=data.get('source', 'gainers'),
            symbol_count=int(data.get('symbol_count', 3)),
            initial_cash=float(data.get('initial_cash', default_cash)),
            leverage=float(data.get('leverage', 3)),
            selected_symbols=list(data.get('selected_symbols', [])),
            step=int(data.get('step', 0)),
            started_at=int(data.get('started_at', 0)),
            ended_at=int(data.get('ended_at', 0)),
            end_reason=data.get('end_reason', ''),
            cooldown_until=dict(data.get('cooldown_until', {})),
            last_selection_at=int(data.get('last_selection_at', 0)))
        # A restored session predates this field, so it may be absent or None. Starting
        # the grace window now lets bookless symbols be retired on the next pass instead
        # of surviving forever because there is no timestamp to measure from.
        session.book_seen_at = dict(data.get('book_seen_at') or {})
        session.last_book_at = int(data.get('last_book_at', 0) or 0)
        if session.selected_symbols and not session.last_selection_at:
            session.last_selection_at = int(time() * 1000)
        return session

    def start(self, initial_cash, leverage, source, symbol_count):
        if self.status == "running":
            raise ValueError("session_already_running")
        initial_cash = float(initial_cash)
        leverage = float(leverage)
        symbol_count = int(symbol_count)
        # The only bound used to be "> 0", so the starting capital was effectively free
        # text: a session ran against 100 while the form defaulted to 10,000, and the two
        # are not comparable in dollars. Every stored statistic that adds PnL across
        # sessions then reported a number with no meaning. The range is deliberately wide
        # -- it is here to catch a slipped decimal point or an empty box, not to make a
        # sizing decision for the operator -- and PnL is now compared as a return as well,
        # so accounts of different sizes stay interpretable rather than merely forbidden.
        if not (self.MIN_INITIAL_CASH <= initial_cash <= self.MAX_INITIAL_CASH):
            raise ValueError("invalid_initial_cash")
        if leverage < 1 or leverage > 20 or symbol_count < 1 or symbol_count > 10:
            raise ValueError("invalid_session_parameters")
        self.session_id = uuid4().hex
        self.initial_cash, self.leverage = initial_cash, leverage
        self.source, self.symbol_count = source if source in ("trained", "gainers", "losers") else "gainers", symbol_count
        self.account = PaperAccount(cash=initial_cash, initial_cash=initial_cash)
        self.account.equity_curve.append(float(initial_cash))
        self.account.margin = MarginModel(leverage=leverage)
        self.status, self.started_at, self.ended_at, self.end_reason = "running", int(time()*1000), 0, ""
        self.step, self.selected_symbols, self.cooldown_until, self.last_selection_at = 0, [], {}, 0
        self._event("simulation_started", {"initial_cash": initial_cash, "leverage": leverage, "source": self.source, "symbol_count": symbol_count})
        self._persist(force=True)

    def pause(self):
        if self.status != "running": raise ValueError("session_not_running")
        self.status = "paused"; self._event("simulation_paused", {}); self._persist(force=True)

    def resume(self):
        if self.status != "paused": raise ValueError("session_not_paused")
        self.status = "running"; self._event("simulation_resumed", {}); self._persist(force=True)

    def end(self, reason="manual"):
        if self.status not in ("running", "paused"): raise ValueError("session_not_active")
        for symbol in list(self.account.positions):
            price = self.account.marks.get(symbol, self.account.positions[symbol].entry)
            # Marked out as 'session_end', not as the session's end reason. The two were
            # the same string, so every position still open when a session stopped was
            # recorded as a 'manual' close -- and a mark-out at whatever price happened to
            # be current is not a discretionary exit. In the stored history that made
            # 'manual' look like the only profitable exit rule (16 rounds, +404.47, 75%
            # win rate) when 9 of the 16 were sessions being stopped mid-position. The
            # exit rule being evaluated and the reason the session stopped are different
            # facts and the audit trail has to keep them apart.
            self.account.close(symbol, price, 'session_end',
                               reason_detail={'source': 'session_end',
                                              'session_reason': reason})
        self.status, self.end_reason, self.ended_at = "ended", reason, int(time()*1000)
        trades=list(self.account.trades)
        for trade in trades:
            trade.setdefault('session_id', self.session_id)
        wins=[float(x.get('pnl',0)) for x in trades if float(x.get('pnl',0))>0]
        losses=[float(x.get('pnl',0)) for x in trades if float(x.get('pnl',0))<=0]
        gross_profit=sum(wins); gross_loss=abs(sum(losses))
        summary={
            'reason':reason,'equity':self.account.equity,'final_equity':self.account.equity,
            'net_pnl':sum(float(x.get('pnl',0)) for x in trades),
            'return_pct':(self.account.equity-self.initial_cash)/self.initial_cash*100 if self.initial_cash else 0.0,
            'trades':len(trades),'wins':len(wins),'losses':len(losses),
            'win_rate_pct':len(wins)/len(trades)*100 if trades else 0.0,
            'gross_profit':gross_profit,'gross_loss':gross_loss,
            'profit_factor':gross_profit/gross_loss if gross_loss else None,
            'total_fees':sum(float(x.get('fees',0)) for x in trades),
            'total_funding':sum(float(x.get('funding',0)) for x in trades),
            'average_holding_ms':sum(int(x.get('holding_ms',0)) for x in trades)/len(trades) if trades else 0.0,
            'best_trade':max((float(x.get('pnl',0)) for x in trades),default=0.0),
            'worst_trade':min((float(x.get('pnl',0)) for x in trades),default=0.0),
            'symbols':sorted({x.get('symbol') for x in trades if x.get('symbol')}),
            'selected_symbols':list(self.selected_symbols),'decision_count':self.step,
            'duration_ms':max(0,self.ended_at-self.started_at) if self.started_at else 0,
            'trades_detail':trades,
        }
        self._event("simulation_ended", summary)
        self._persist()

    # A symbol whose order book never arrives cannot be filled: the entry gate requires a
    # fresh bid/ask, so every decision for it is rejected and the whole tick was wasted.
    # Live probing showed Binance publishes no bookTicker for some actively traded
    # contracts, so this is a data-coverage property, not a liquidity one.
    BOOKLESS_GRACE_S = 60

    # Bounds on the operator-chosen starting capital. Wide on purpose: the point is to
    # reject a slipped decimal or an empty field, not to second-guess position sizing.
    # 1,000x range keeps every account within an order of magnitude or so of the others,
    # which is what makes a cross-session return comparable at all.
    MIN_INITIAL_CASH = 100.0
    MAX_INITIAL_CASH = 100_000.0

    def _limit(self, max_positions):
        limit = self.symbol_count
        if max_positions:
            # One slot per symbol, so asking for more symbols than slots is a
            # contradiction rather than extra opportunity.
            limit = min(limit, int(max_positions))
        return max(0, int(limit))

    def _rank(self, rows, limit, tier=None, explore_slots=0, exclude=()):
        """Tradable symbols first, then unproven ones, then the ranking's order.

        The ranking picks by price move; the meta-labeling gate rejects by measured
        cost-adjusted edge. Left independent they disagree, and the failure is silent: a
        session selected five symbols that all failed the gate and simply never traded.

        Three tiers rather than a yes/no answer, because the two kinds of "no" are not
        equivalent. A symbol with too little evidence might become tradable and is worth a
        slot; a symbol already measured as a loser has been answered, and a slot spent on
        it buys nothing. Ranking them together filled the session with symbols carrying
        fifty samples of negative edge while unproven candidates went unobserved.

        The reservation matters as much as the ordering. Evidence is gathered from signals,
        and signals are only computed for symbols the session holds, so a selector that
        took only proven symbols would never observe a new one and the gate would freeze
        on whatever it happened to learn first.
        """
        exclude = {symbol for symbol in exclude}
        usable = [row for row in rows
                  if row.get("symbol") and row.get("price", 0) > 0
                  and row["symbol"] not in exclude]
        if tier is None:
            return [row["symbol"] for row in usable][:limit]
        eligible, unknown, rejected = [], [], []
        for row in usable:
            symbol = row["symbol"]
            rank = tier(symbol)
            (eligible if rank >= 2 else unknown if rank == 1 else rejected).append(symbol)
        # Never surrender every slot to exploration, and never spend them all on what is
        # already proven; one slot is enough to keep discovering.
        quota = max(0, min(int(explore_slots), limit - 1)) if limit else 0
        picked = eligible[:limit - quota]
        # Unknown before rejected, and rejected only once the unproven pool is empty. An
        # empty slot produces no signals at all, and evidence is gathered from signals, so
        # a symbol already measured as a loser is still worth a slot when there is nothing
        # else: its evidence keeps refreshing and the gate can flip back. Handing it a slot
        # *ahead* of an unproven symbol is what the tiering exists to prevent.
        for pool in (unknown, rejected):
            if len(picked) >= limit:
                break
            picked.extend(pool[:limit - len(picked)])
        return picked

    def choose(self, market, max_positions=None, tier=None, explore_slots=0):
        """Pick this session's symbols, never more than the portfolio can hold.

        Two constraints the previous selection ignored, each of which made symbols pure
        overhead. The operator selected seven symbols while MAX_POSITIONS was four, so
        three could never hold: max_positions was 185 of 257 entry rejections. And two of
        the seven never received an order book at all, so all 40 of their decisions were
        rejected as missing_book_time. Together five of seven symbols produced nothing.
        """
        rows = list(market.get(self.source, []))
        # Filter before slicing: taking the head first lets an unusable row consume one
        # of the slots and silently returns fewer symbols than the session asked for.
        self.selected_symbols = self._rank(rows, self._limit(max_positions),
                                           tier, explore_slots)
        self.last_selection_at = int(time() * 1000)
        self.book_seen_at = {}
        return self.selected_symbols

    def reselect(self, market, max_positions=None, drop=(), tier=None, explore_slots=0):
        """Replace dead symbols with the next best candidates from the same ranking.

        Keeps the session at its configured symbol count while retiring symbols that
        cannot be filled, instead of leaving a position slot permanently idle.
        """
        drop = {symbol for symbol in drop}
        rows = list(market.get(self.source, []))
        keep = [symbol for symbol in self.selected_symbols if symbol not in drop]
        # Refill up to the symbol count the session was configured for, and never above
        # the position cap. Refilling to the cap is what matters: a dead symbol occupying
        # a slot is strictly worse than an empty slot, because the empty one can still be
        # taken by something that trades.
        configured = self._limit(max_positions) or self.symbol_count
        # Ask only for the slots still open, and exclude what is held or retired. Ranking
        # the full count and then skipping over the held symbols returned nothing to
        # replace the dropped one, because the dropped symbol was itself in the top slice.
        need = configured - len(keep)
        if need > 0:
            keep.extend(self._rank(rows, need, tier, explore_slots,
                                   exclude=set(keep) | drop))
        if not keep:
            # Everything was dropped and nothing replaced it; the session would be left
            # with no symbols at all, so keep what it has and retry on a later pass.
            return []
        # Forget the retired symbols so a later reselection can pick them again if their
        # book comes back, rather than treating them as permanently broken.
        for symbol in drop:
            self.book_seen_at.pop(symbol, None)
        self.selected_symbols = keep
        self.last_selection_at = int(time() * 1000)
        self._event("session_symbols_reselected", {"dropped": sorted(drop), "selected": keep})
        self._persist(force=True)
        return list(drop)

    def note_book(self, symbol, now_ms=None):
        """Record that an order book arrived for a symbol, as bookTicker only tells us
        what it chooses to publish."""
        now_ms = int(now_ms or time() * 1000)
        if self.book_seen_at is None:
            self.book_seen_at = {}
        self.book_seen_at[symbol] = now_ms
        if now_ms > self.last_book_at:
            self.last_book_at = now_ms

    def without_market_data(self, now_ms=None, feed_live=True):
        """Selected symbols that have produced no order book within the grace period.

        They can be dropped for replacement: keeping them costs a model evaluation and a
        subscription per tick and can never produce a fill.

        ``feed_live`` is the caller's knowledge of whether the book transport is up at
        all, and it is a parameter rather than something this class infers because the
        session cannot tell "this symbol is dead" from "every symbol is dead": both look
        like an empty book_seen_at. Retiring symbols is only valid for the first. During a
        five-hour websocket outage the second held, and the session dropped its symbols
        every ~70s -- 54 reselections in 24 hours. Because the bar loop resolves forecasts
        only for symbols it still holds, each reselection discarded the pending forecasts
        along with the symbol that made them, so the per-symbol meta-labeling gate never
        accumulated a single sample and refused every symbol forever. A transport outage
        was being reported as a property of the symbols.
        """
        if not feed_live:
            return []
        now_ms = int(now_ms or time() * 1000)
        seen_at = self.book_seen_at or {}
        reference = self.last_selection_at
        if not reference or now_ms - reference < self.BOOKLESS_GRACE_S * 1000:
            # Give a freshly selected symbol the full grace period before judging it, so
            # a symbol is not retired merely for being slow to start.
            return []
        # Silence is measured per symbol from the last book IT produced, counted from
        # selection for one that has never produced any. Measuring from the newest book
        # anywhere would make a live symbol its own reference and expire it alongside the
        # dead ones.
        return [symbol for symbol in self.selected_symbols
                if now_ms - seen_at.get(symbol, reference) >= self.BOOKLESS_GRACE_S * 1000]

    def can_enter(self, symbol, now_ms=None):
        return int(now_ms or time()*1000) >= int(self.cooldown_until.get(symbol, 0))

    def set_cooldown(self, symbol, minutes=5):
        self.cooldown_until[symbol]=int(time()*1000)+int(minutes*60000)
        self._persist(force=True)

    def record_decision(self, symbol, decision, market, audit=True):
        """Count one decision, and append it to the audit trail only when it is news.

        The tick path re-evaluates each symbol about ten times a second and repeats the
        previous side almost every time. Persisting every evaluation is how the event
        table reached a million rows that were duplicates of a few hundred real
        decisions. Every evaluation still advances the step counter; only a decision
        that changed, or a periodic heartbeat, becomes a row.
        """
        self.step += 1
        if not audit:
            return
        self._event("strategy_decision", {"symbol": symbol, "decision": decision, "market": market})
        self._persist(force=False)

    def mark(self, prices, bars=None, mark_prices=None, record=True):
        if self.status != "running": return
        before = len(self.account.liquidations)
        self.account.mark(prices, bars, mark_prices=mark_prices, record=record)
        if len(self.account.liquidations) > before:
            self.status, self.end_reason, self.ended_at = "liquidated", "maintenance_margin", int(time()*1000)
            self._event("simulation_liquidated", {"liquidations": self.account.liquidations[before:]})
            self._persist(force=True)
            return
        self._persist(force=False)

    def _persist(self, force=True):
        # The runtime row is a crash-recovery cache; intermediate tick decisions do not
        # need their own megabyte-sized rewrite of the whole session.
        now = int(time()*1000)
        if not force and now - self.last_persist_at < self.persist_min_interval_ms:
            self.persist_skipped += 1
            return False
        self.last_persist_at = now
        self.store.set_runtime("simulation_session", self.snapshot())
        return True

    def snapshot(self):
        account = self.account.snapshot()
        # The full curve lives in the equity table; the cached copy only needs a recent tail.
        account["equity_curve"] = list(self.account.equity_curve[-self.snapshot_equity_points:])
        # A mapping rather than one expression over fifteen fields: the single-line form
        # could not be read, and a field added to the dataclass could be left out of it
        # without anything failing until a restart dropped it.
        return {
            "session_id": self.session_id,
            "status": self.status,
            "source": self.source,
            "symbol_count": self.symbol_count,
            "initial_cash": self.initial_cash,
            "leverage": self.leverage,
            "selected_symbols": self.selected_symbols,
            "step": self.step,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "cooldown_until": self.cooldown_until,
            "last_selection_at": self.last_selection_at,
            "persist_skipped": self.persist_skipped,
            "book_seen_at": dict(self.book_seen_at or {}),
            "last_book_at": self.last_book_at,
            "account": account,
        }
