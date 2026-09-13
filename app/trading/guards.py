from dataclasses import dataclass, field
import time

@dataclass
class MarketGuard:
    """Refuses entries for a symbol whose live feed has gone quiet.

    The timestamps observed here are exchange event times from the live socket. The
    closed-bar loop used to be the only writer and it wrote a bar's close_time, which
    trails the wall clock by up to a full bar. With a two-minute window against a
    five-minute bar, the guard marked every symbol stale for most of each bar and
    rejected the bulk of all entries. Observed from the socket, the same window means
    what it says: this symbol has stopped publishing.
    """
    max_age_ms: int=120000
    last_event_ms: dict=field(default_factory=dict)
    def observe(self,symbol,event_time=None):
        # Monotonic. Two feeds stamp the same symbol from different clocks, and a
        # lower timestamp must not walk the freshness window backwards.
        stamp=int(event_time or time.time()*1000)
        if stamp>self.last_event_ms.get(symbol,0): self.last_event_ms[symbol]=stamp
    def fresh(self,symbol,now_ms=None): return symbol in self.last_event_ms and int(now_ms or time.time()*1000)-self.last_event_ms[symbol] <= self.max_age_ms
    def approve_new_entry(self,symbol,now_ms=None): return (True,'fresh_data') if self.fresh(symbol,now_ms) else (False,'stale_market_data')

def portfolio_limits(settings):
    """Build PortfolioLimits from Settings, keeping the caps coherent.

    Both caps are multiples of equity rather than absolute amounts. As dollars they
    could not mean the same thing twice: a 20000 cap was unreachable on a 100 account
    and binding on a 10000 one, and it also contradicted the leverage chosen for the
    session. A per-symbol cap above the total cap can never bind, so it is still
    clamped down to the total.
    """
    symbol_cap=float(settings.max_symbol_leverage)
    total_cap=float(settings.max_gross_leverage)
    if symbol_cap>total_cap:
        symbol_cap=total_cap
    return PortfolioLimits(max_positions=int(settings.max_positions),
                           max_symbol_leverage=symbol_cap,
                           max_gross_leverage=total_cap)


@dataclass
class PortfolioLimits:
    max_positions: int=5; max_symbol_leverage: float=3.0; max_gross_leverage: float=5.0
    # Notional allowed in symbols correlated with the candidate, as a multiple of equity.
    # 0 disables the check. The gross cap cannot substitute for it: nine symbols that move
    # together are one position wearing nine tickers, and gross notional treats them as
    # nine independent bets.
    max_correlated_leverage: float=0.0
    correlation_threshold: float=0.6

    @classmethod
    def from_profile(cls, profile):
        """Portfolio caps for a named risk profile.

        Startup used to build these from Settings while the risk engine used the
        persisted profile, so the two only agreed by coincidence.
        """
        return cls().apply_profile(profile)

    def apply_profile(self, profile):
        """Install a risk profile's portfolio caps, keeping the symbol cap coherent."""
        symbol_cap = float(profile.max_symbol_leverage)
        total_cap = float(profile.max_gross_leverage)
        if symbol_cap > total_cap:
            symbol_cap = total_cap
        self.max_positions = int(profile.max_positions)
        self.max_symbol_leverage = symbol_cap
        self.max_gross_leverage = total_cap
        # A profile may carry the concentration cap; older profiles do not, and then the
        # check stays off rather than guessing a number.
        # Only overwritten when the profile actually carries the setting. A profile that
        # does not must not reset a cap the operator configured in the environment.
        if hasattr(profile, 'max_correlated_leverage'):
            self.max_correlated_leverage = float(profile.max_correlated_leverage or 0.0)
        if hasattr(profile, 'correlation_threshold'):
            self.correlation_threshold = float(profile.correlation_threshold or 0.6)
        return self
    def approve(self,symbol,notional,open_notionals,equity=None,correlations=None):
        if symbol not in open_notionals and len(open_notionals)>=self.max_positions: return False,'max_positions'
        equity=float(equity or 0.0)
        if equity<=0:
            # Without equity the money caps are unquantifiable; the count cap still holds.
            return True,'portfolio_ok'
        if notional>equity*self.max_symbol_leverage: return False,'max_symbol_notional'
        if sum(open_notionals.values())+notional>equity*self.max_gross_leverage: return False,'max_total_notional'
        if self.max_correlated_leverage>0 and correlations is not None:
            from .concentration import correlated_notional
            crowded=correlated_notional(symbol, open_notionals, correlations, self.correlation_threshold)
            if crowded+notional>equity*self.max_correlated_leverage:
                return False,'max_correlated_notional'
        return True,'portfolio_ok'
