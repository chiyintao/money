"""Position sizing and the daily loss circuit breaker.

Sizing used to reduce to a single line:

    qty = min(equity * max_risk / unit_risk,
              max(0, (equity * max_notional_multiple - open_notional) / entry))

Only the first term decides anything on a normal account: it risks max_risk of equity
over the stop distance, so a symbol whose stop is wide gets a tiny position no matter
how much capital is idle. In the stored sessions a 100 USDT account with a 1.8% stop
and max_risk of 0.002 deployed about 11 USDT of notional and left 95% of the balance
unused, while the second term silently overrode the leverage the operator had chosen
for the session.

Sizing now balances the constraints that actually matter and reports which one bound,
so a small order can be explained instead of guessed at:

* risk   -- never lose more than max_risk of equity when the stop is hit, further
            limited by what is left of the whole portfolio's risk budget;
* target -- deploy target_exposure of equity when risk alone would under-deploy;
* room   -- stay inside the per-symbol and gross notional caps.

The binding constraint is returned with every decision, which is what turns "the
sizing looks conservative" into an answerable question.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RiskEngine:
    max_risk: float=.005
    max_daily_loss: float=.20
    max_gross_leverage: float=5.0
    day_start_equity: float=10000
    halted: bool=False
    day_key: str=''
    halt_reason: str=''
    halt_equity: float=0.0
    halt_threshold: float=0.0
    max_portfolio_risk: float=.02
    target_exposure: float=1.0
    max_symbol_leverage: float=3.0
    # Appended after the original fields on purpose. This dataclass is constructed
    # positionally in several places, so inserting a field in the middle silently shifts
    # every value after it: adding these before max_portfolio_risk made restore() read
    # target_exposure as 1.0 while the snapshot said 0.7, and nothing raised.
    #
    # High-water mark and the drawdown from it that stops the account until a human resets
    # it. The daily breaker cannot express this: it is deliberately forgiving at the date
    # boundary, and a cumulative loss is not.
    peak_equity: float=0.0
    drawdown_limit: float=0.0
    halt_scope: str=''
    halt_drawdown: float=0.0
    # Consecutive losing closes and the size multiplier applied while they persist. This is
    # the standard response to a model that has stopped working: trade smaller, do not stop
    # measuring.
    loss_streak: int=0
    loss_streak_limit: int=0
    loss_streak_scale: float=0.5
    # The edge at which a position gets the full per-trade risk budget. Below it the
    # budget scales linearly down to zero at no edge. Set to 0 to disable edge scaling
    # entirely, which restores the pre-audit behaviour of sizing every signal the same.
    sizing_reference_edge_bps: float=25.0
    # The fraction of the per-trade risk budget a signal receives once it has cleared the
    # entry gate, however small its measured edge.
    #
    # This exists because the reference above turned out to be a *scale* rather than a
    # ceiling. Measured on 150 scored signals, the median net edge is -12 bps and the
    # signals that clear the 0.5 bps floor cluster below 4 bps, while the reference sat at
    # 25 -- so three quarters of everything that traded was sized at under a sixth of its
    # budget, and the eight trades of the last session used a mean 13.9% of the $100 budget
    # they were granted. The ramp is exact (scale == edge/25 up to the cap, independent of
    # stop and target geometry), so that is not a rounding effect.
    #
    # A floor says something different and narrower than lowering the reference: it does not
    # claim the model can rank a 2 bps signal against an 8 bps one. It claims only that
    # clearing the gate is itself evidence worth acting on, which is the one judgement the
    # gate was calibrated to make. 0 disables it and restores the pure ramp.
    sizing_floor_scale: float=0.0
    # Volatility targeting. target_volatility is per bar and expressed in the same units as
    # the realized measure; 0 disables the scaling entirely.
    target_volatility: float=0.0
    volatility_floor: float=0.25
    # Entries taken since the UTC date rolled. A backstop, not a strategy constraint: the
    # entry gate once let a single symbol produce 37 round trips in under ten seconds at a
    # net loss, and nothing counted. Every other cap here is about size -- how much a trade
    # may lose -- and none of them can see a strategy that takes many small losing trades.
    # 0 disables it.
    daily_trade_limit: int=0
    trades_today: int=0

    def _roll_day(self, equity, key=None):
        """Start a new UTC trading day. Both entry points funnel through here.

        approve() used to carry a second, inline copy of this for the branch that is given
        an event time -- which is the branch production always takes. The two copies had
        already drifted: the inline one reset the day key and the equity but left
        halt_reason, halt_equity and halt_threshold describing yesterday, so a stale halt
        reason outlived the halt. Adding the trade count to one copy and not the other
        would have made the daily budget latch shut permanently after the first day.
        """
        key=key or datetime.now(timezone.utc).date().isoformat()
        if key == self.day_key:
            return False
        self.day_key=key
        self.day_start_equity=equity
        # The trade count is a per-day quantity like the daily loss budget, so it rolls
        # with the same date. A drawdown halt is not, and a new date does not undo it:
        # only a human reset clears it, otherwise the breaker is a daily speed bump.
        self.trades_today=0
        if self.halt_scope != 'high_water_mark':
            self.halted=False
            self.halt_reason=''; self.halt_equity=0.0; self.halt_threshold=0.0
        return True

    def apply_profile(self, profile):
        """Install a risk profile, keeping the day's running state intact.

        Only the sizing parameters change. day_start_equity, the halt flag and the day
        key are state, not policy: clearing them mid-session would silently disarm the
        daily loss circuit breaker that the profile is meant to work with.
        """
        self.max_risk = float(profile.max_risk_per_trade)
        self.max_portfolio_risk = float(profile.max_portfolio_risk)
        self.target_exposure = float(profile.target_exposure)
        self.max_symbol_leverage = float(profile.max_symbol_leverage)
        self.max_gross_leverage = float(profile.max_gross_leverage)
        self.max_daily_loss = float(profile.max_daily_loss)
        return self.snapshot()

    def observe_equity(self, equity):
        """Track the high-water mark and the current losing streak."""
        equity=float(equity or 0.0)
        if not math.isfinite(equity):
            return
        if equity > self.peak_equity:
            self.peak_equity=equity
        return self.peak_equity

    def record_close(self, pnl):
        """Feed a closed trade's realized result back into the risk state.

        Losing streaks are the cheapest available evidence that the edge has changed, and
        they are the one signal this account already produces for free.
        """
        try:
            value=float(pnl)
        except (TypeError, ValueError):
            return self.loss_streak
        if not math.isfinite(value):
            return self.loss_streak
        self.loss_streak = self.loss_streak + 1 if value < 0 else 0
        return self.loss_streak

    def record_entry(self):
        """Count an entry that was actually submitted.

        Separate from approve() on purpose. approve() also runs for plans that are then
        refused by the portfolio caps, the execution-readiness check or the lot step, and
        counting those would spend the day's budget on orders that never existed -- a cap
        that tightens itself the more the rest of the system says no.
        """
        self.trades_today = int(self.trades_today or 0) + 1
        return self.trades_today

    def trades_remaining(self):
        if self.daily_trade_limit <= 0:
            return None
        return max(0, int(self.daily_trade_limit) - int(self.trades_today or 0))

    def streak_scale(self):
        """Size multiplier for the current losing streak: 1.0 until the streak is long."""
        if self.loss_streak_limit <= 0 or self.loss_streak < self.loss_streak_limit:
            return 1.0
        scale=float(self.loss_streak_scale or 0.0)
        return scale if 0.0 < scale < 1.0 else 1.0

    def reset_for_session(self, initial_equity, day_key=None):
        self.day_start_equity=float(initial_equity)
        self.halted=False
        self.day_key=day_key or datetime.now(timezone.utc).date().isoformat()
        self.halt_reason=''; self.halt_equity=0.0; self.halt_threshold=0.0
        # A new session is a fresh account, so its high-water mark starts from its own
        # capital and no losing streak carries across the boundary.
        self.peak_equity=float(initial_equity or 0.0)
        self.halt_scope=''; self.halt_drawdown=0.0; self.loss_streak=0
        self.trades_today=0

    def snapshot(self):
        return {'max_risk':self.max_risk,'max_daily_loss':self.max_daily_loss,
                'peak_equity':self.peak_equity,'drawdown_limit':self.drawdown_limit,
                'halt_scope':self.halt_scope,'halt_drawdown':self.halt_drawdown,
                'loss_streak':self.loss_streak,'loss_streak_limit':self.loss_streak_limit,
                'loss_streak_scale':self.loss_streak_scale,'streak_scale':self.streak_scale(),
                'target_volatility':self.target_volatility,'volatility_floor':self.volatility_floor,
                'max_gross_leverage':self.max_gross_leverage,'max_symbol_leverage':self.max_symbol_leverage,
                'max_portfolio_risk':self.max_portfolio_risk,'target_exposure':self.target_exposure,
                'day_start_equity':self.day_start_equity,'halted':self.halted,'day_key':self.day_key,
                'halt_reason':self.halt_reason,'halt_equity':self.halt_equity,'halt_threshold':self.halt_threshold,
                'daily_trade_limit':self.daily_trade_limit,'trades_today':self.trades_today,
                'trades_remaining':self.trades_remaining()}

    @classmethod
    def restore(cls, data, default_equity=10000):
        data=data or {}
        day_start=float(data.get('day_start_equity',default_equity) or 0.0)
        peak=data.get('peak_equity')
        # Every field by name. The previous version passed twelve values positionally into
        # a constructor whose field order had changed, so two of them landed in the wrong
        # slots and a restored engine sized with settings nobody had chosen.
        return cls(max_risk=float(data.get('max_risk',.005)),
                   max_daily_loss=float(data.get('max_daily_loss',.20)),
                   max_gross_leverage=float(data.get('max_gross_leverage',5.0)),
                   day_start_equity=day_start,
                   halted=bool(data.get('halted',False)),
                   day_key=data.get('day_key',''),
                   halt_reason=data.get('halt_reason',''),
                   halt_equity=float(data.get('halt_equity',0)),
                   halt_threshold=float(data.get('halt_threshold',0)),
                   max_portfolio_risk=float(data.get('max_portfolio_risk',.02)),
                   target_exposure=float(data.get('target_exposure',1.0)),
                   max_symbol_leverage=float(data.get('max_symbol_leverage',3.0)),
                   peak_equity=day_start if peak is None else float(peak),
                   drawdown_limit=float(data.get('drawdown_limit',0.0) or 0.0),
                   halt_scope=str(data.get('halt_scope') or ''),
                   halt_drawdown=float(data.get('halt_drawdown',0.0) or 0.0),
                   loss_streak=int(data.get('loss_streak',0) or 0),
                   loss_streak_limit=int(data.get('loss_streak_limit',0) or 0),
                   loss_streak_scale=float(data.get('loss_streak_scale',0.5) or 0.5),
                   target_volatility=float(data.get('target_volatility',0.0) or 0.0),
                   volatility_floor=float(data.get('volatility_floor',0.25) or 0.25),
                   daily_trade_limit=int(data.get('daily_trade_limit',0) or 0),
                   trades_today=int(data.get('trades_today',0) or 0))

    def edge_scale(self, plan, edge_bps=None, probability_up=None):
        """How much of the risk budget this particular edge deserves, in [0, 1].

        Returns (scale, inputs). A scale of 1.0 is the existing behaviour, so a plan with
        no edge information is sized exactly as before and every caller that does not pass
        one is unaffected.

        The scale is the edge relative to a reference edge, clipped to [0, 1]:

            edge >= reference   -> 1.0   (full risk budget, i.e. the status quo)
            edge == 0           -> 0.0   (no edge, no position)
            in between          -> linear

        A reference is needed because raw Kelly does not work here. Kelly for a bet with
        win probability p and payoff ratio b is f* = p - (1-p)/b, and against a per-trade
        cap of 0.5% it is enormous for any real edge -- a 40% hit rate at 1.8 reward:risk
        gives f* = 0.067, thirteen times the cap. Normalising Kelly by the cap therefore
        saturates at 1.0 for everything, which is the same as not scaling at all.

        Kelly is still computed and reported, because it answers a different question:
        whether the plan is worth taking at all. At or below the break-even probability
        the growth-optimal bet is nothing, so the scale is zero regardless of the edge.

        Two properties this deliberately keeps:

        * It can only shrink a position. Every existing cap, limit and test still means
          what it meant, and the worst case is the previous behaviour.
        * It is monotone in the edge, which is the specific defect it fixes: a 0.5bp edge
          and a 50bp edge used to produce identical quantities.
        """
        inputs = {'edge_bps': None if edge_bps is None else round(float(edge_bps), 4),
                  'kelly_fraction': None, 'implied_probability': None}
        if edge_bps is None and probability_up is None:
            return 1.0, inputs
        entry = float(plan.get('entry') or 0.0)
        stop = float(plan.get('stop') or 0.0)
        target = float(plan.get('target') or plan.get('take_profit') or 0.0)
        risk_per_unit = abs(entry - stop)
        reward_per_unit = abs(target - entry) if target else 0.0
        if entry <= 0 or risk_per_unit <= 0:
            return 1.0, inputs
        reward_ratio = (reward_per_unit / risk_per_unit) if reward_per_unit > 0 else 1.0
        # Loss and reward as fractions of notional, which is the unit `edge_bps` is in.
        loss_frac = risk_per_unit / entry
        reward_frac = reward_per_unit / entry
        span = loss_frac + reward_frac
        if span <= 0:
            return 1.0, inputs
        # Break-even: the win probability at which the bet has zero expected value.
        break_even = loss_frac / span

        def kelly_at(probability):
            return probability - (1.0 - probability) / reward_ratio

        reference = max(0.0, float(self.sizing_reference_edge_bps))
        if reference <= 0:
            # Disabled: every signal gets the full budget, as before the audit.
            return 1.0, inputs

        if probability_up is not None:
            # The model's calibrated probability of the favourable barrier. Used directly:
            # it already answers the question the geometry would otherwise infer.
            probability = max(0.0, min(1.0, float(probability_up)))
        else:
            # Convert the claimed edge into the same space. Expected value per unit of
            # notional is p*reward_frac - (1-p)*loss_frac, so for a given edge
            # p = break_even + edge / span. At a zero edge this returns break-even exactly,
            # which is what makes the two input forms agree instead of merely resembling
            # each other -- feeding edge_bps=0 and probability_up=break_even must produce
            # the same scale, and previously did not.
            #
            # The edge keeps its sign. This read abs(edge_bps), so a prediction of -10bp
            # was sized exactly like +10bp: the worse the model expected the trade to be,
            # the more of the risk budget it received. The serving layer computes a NET
            # figure that is negative for any prediction below the round trip, so this was
            # reachable on ordinary signals rather than only on malformed input, and the
            # direction was already chosen by that layer -- a negative net edge means there
            # was no trade there to size.
            probability = break_even + (float(edge_bps) / 10000.0) / span
            probability = max(0.0, min(1.0, probability))
        inputs['implied_probability'] = round(probability, 6)
        kelly = kelly_at(probability)
        inputs['kelly_fraction'] = round(kelly, 6)
        reference_probability = max(0.0, min(1.0, break_even + (reference / 10000.0) / span))
        reference_kelly = kelly_at(reference_probability)
        inputs['reference_kelly'] = round(reference_kelly, 6)
        if kelly <= 0:
            # At or below break-even the growth-optimal bet is nothing at all. A zero
            # scale still returns a plan, and approve() reports the refusal with its
            # reason, so this is not a silent drop.
            return 0.0, inputs
        if reference_kelly <= 0:
            return 1.0, inputs
        # Normalised by the Kelly of the reference edge, not by the raw fraction: a raw
        # Kelly against a 0.5% per-trade cap saturates at 1.0 for any real edge, which is
        # the same as not scaling at all.
        scale = kelly / reference_kelly
        # At exactly break-even the arithmetic above yields a Kelly of ~5e-17 rather than
        # 0, because break_even is loss/span and that division does not land on the same
        # float twice. Without this the no-edge case would submit a position of 2e-14
        # units instead of refusing, which is a real order at a size the venue rounds to
        # zero -- and it also made the binding cap read 'risk' rather than the exhaustion
        # the caller needs to see.
        if scale <= 1e-9:
            return 0.0, inputs
        # The floor is applied only to a signal that already has positive Kelly, so it can
        # never turn a refusal into a position: the zero-scale return above has already
        # happened by this point. It raises a small position, never creates one, and the
        # report carries both numbers so the difference is visible in the artifact.
        floor = max(0.0, min(1.0, float(self.sizing_floor_scale)))
        inputs['scale_before_floor'] = round(scale, 6)
        inputs['sizing_floor'] = floor
        return min(1.0, max(scale, floor)), inputs

    def risk_budget(self, equity, open_risk=0.0):
        """Cash this entry may still lose: the per-trade cap inside what is left of the
        portfolio budget. Without the second term every concurrent position could risk
        the full per-trade amount and the portfolio cap would never mean anything."""
        per_trade=max(0.0, float(equity)*self.max_risk)
        remaining=max(0.0, float(equity)*self.max_portfolio_risk-max(0.0, float(open_risk)))
        return min(per_trade, remaining)

    def size(self, plan, equity, open_notional=0.0, open_risk=0.0, symbol_notional=0.0, vol_scale=1.0,
             edge_bps=None, probability_up=None):
        """Quantity for one plan plus the constraint that decided it.

        Returns None for an unusable plan, otherwise a dict whose quantity may be zero
        when a cap leaves no room at all.

        ``edge_bps`` and ``probability_up`` are optional and scale the risk budget DOWN,
        never up. The audit measured the previous behaviour directly: a 0.5bp edge and a
        50bp edge produced exactly the same quantity, because neither was an input. Two
        consequences, both bad -- the position carried no information about how good the
        opportunity was, and a marginal signal that barely cleared the cost floor was
        sized as if it were the best one the model would ever produce.

        The form is a Kelly fraction bounded by the existing per-trade cap:

            f* = kelly_fraction(p, reward:risk)      # the theoretical optimum
            f  = min(max_risk, f* * confidence)      # capped, and shrunk by confidence

        ``max_risk`` remains the ceiling, so this can only ever make a position smaller.
        That is deliberate: every existing cap, limit and test still means what it meant,
        and the change is strictly a reduction in exposure to weak signals.
        """
        entry=float(plan['entry']); unit_risk=abs(entry-float(plan['stop']))
        if entry<=0 or unit_risk<=0:
            return None
        equity=float(equity)
        stop_pct=unit_risk/entry
        # The candidate that reaches this function is the planner's own dict, and the model
        # already writes its edge into it. Reading the plan as a fallback means a caller
        # gets edge-weighted sizing by passing the candidate it already has, rather than
        # having to know to unpack two fields into keywords -- which is exactly the kind of
        # step that gets forgotten and silently reverts to flat sizing.
        if edge_bps is None:
            edge_bps = plan.get('edge_bps')
        if probability_up is None:
            probability_up = plan.get('probability_up')
        edge_scale, edge_inputs = self.edge_scale(plan, edge_bps, probability_up)
        # A losing streak shrinks the risk budget rather than halting trading. The gate
        # that judges the model is a separate mechanism and needs new signals to work
        # from, so the response to a bad run is smaller, not silent.
        # Both multipliers shrink the risk budget. Neither ever enlarges it: the streak
        # scale is at most 1 and volatility_scale is capped at 1 by default.
        risk_cash=self.risk_budget(equity, open_risk)*self.streak_scale()*float(vol_scale or 1.0)*edge_scale
        if risk_cash<=0:
            return {'quantity':0.0,'notional':0.0,'risk_cash':0.0,'binding':'risk_budget_exhausted',
                    'risk_notional':0.0,'target_notional':0.0,'room':0.0,
                    'edge_scale':round(edge_scale,6),**edge_inputs}
        risk_notional=risk_cash/stop_pct
        target_notional=max(0.0, equity*self.target_exposure)
        room=min(max(0.0, equity*self.max_symbol_leverage-max(0.0, float(symbol_notional))),
                 max(0.0, equity*self.max_gross_leverage-max(0.0, float(open_notional))))
        if room<=0:
            return {'quantity':0.0,'notional':0.0,'risk_cash':0.0,'binding':'notional_room',
                    'risk_notional':risk_notional,'target_notional':target_notional,'room':0.0,
                    'edge_scale':round(edge_scale,6),**edge_inputs}
        caps=[('risk',risk_notional)]
        # A target of zero means "unset", not "trade nothing", so it is simply absent.
        if target_notional>0:
            caps.append(('target',target_notional))
        caps.append(('room',room))
        binding,notional=min(caps,key=lambda item: item[1])
        quantity=notional/entry
        return {'quantity':quantity,'notional':notional,'risk_cash':quantity*unit_risk,
                'binding':binding,'risk_notional':risk_notional,'target_notional':target_notional,'room':room,
                'edge_scale':round(edge_scale,6),**edge_inputs}

    def approve(self, plan, equity, open_notional=0, event_time_ms=None, open_risk=0.0,
                symbol_notional=0.0, realized_volatility=0.0, edge_bps=None,
                probability_up=None):
        if event_time_ms is not None:
            # The live path: the day is the exchange event time, not the wall clock, so a
            # replay and a live session roll the same way.
            self._roll_day(equity, datetime.fromtimestamp(
                int(event_time_ms)/1000, timezone.utc).date().isoformat())
        else:
            self._roll_day(equity)
        # The daily breaker resets itself at the UTC date boundary against the equity at
        # that moment, so "lose the daily limit every day, forever" was an allowed path:
        # each new day began from the reduced balance and the account bled in steps with
        # nothing ever refusing. The high-water mark below is what makes a losing run
        # cumulative rather than daily.
        self.observe_equity(equity)
        if self.halted:
            return {'approved':False,'reason':self.halt_reason or 'halted','equity':equity,
                    'threshold':self.halt_threshold,'scope':self.halt_scope}
        if self.drawdown_limit > 0 and self.peak_equity > 0:
            drawdown = 1.0 - (equity / self.peak_equity)
            if drawdown >= self.drawdown_limit:
                self.halted=True
                self.halt_reason='max_drawdown_circuit_breaker'
                self.halt_scope='high_water_mark'
                self.halt_equity=float(equity)
                self.halt_threshold=float(self.peak_equity*(1-self.drawdown_limit))
                self.halt_drawdown=drawdown
                return {'approved':False,'reason':self.halt_reason,'equity':equity,
                        'threshold':self.halt_threshold,'drawdown':drawdown,
                        'scope':'high_water_mark','requires_manual_reset':True}
        threshold=self.day_start_equity*(1-self.max_daily_loss)
        if equity <= threshold:
            self.halted=True
            self.halt_reason='daily_loss_circuit_breaker'
            self.halt_scope='daily'
            self.halt_equity=float(equity); self.halt_threshold=float(threshold)
            return {'approved':False,'reason':self.halt_reason,'equity':equity,'threshold':threshold,
                    'scope':'daily'}
        if self.daily_trade_limit > 0 and self.trades_today >= self.daily_trade_limit:
            return {'approved':False,'reason':'daily_trade_limit','equity':equity,
                    'trades_today':self.trades_today,'daily_trade_limit':self.daily_trade_limit}
        side=plan.get('side','LONG')
        if side not in ('LONG','SHORT') or plan.get('entry',0)<=0 or plan.get('stop',0)<=0 or plan['entry']==plan['stop']:
            return {'approved':False,'reason':'invalid_plan'}
        vol_scale=1.0
        if self.target_volatility>0 and realized_volatility:
            from .concentration import volatility_scale
            vol_scale=volatility_scale(realized_volatility, self.target_volatility,
                                       floor=self.volatility_floor)
        sized=self.size(plan, equity, open_notional, open_risk, symbol_notional,
                        vol_scale=vol_scale, edge_bps=edge_bps, probability_up=probability_up)
        if sized is None:
            return {'approved':False,'reason':'invalid_plan'}
        sized['volatility_scale']=vol_scale
        sized['realized_volatility']=float(realized_volatility or 0.0)
        if sized['quantity']<=0:
            reason='notional_limit' if sized['binding']=='notional_room' else 'risk_budget_exhausted'
            return {'approved':False,'reason':reason,**sized}
        return {'approved':True,'order_type':'PAPER_MARKET',**sized}
