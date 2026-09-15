"""Replay the forward tracker over history; candles from live db, events to a scratch store."""
import shutil, tempfile
from app.storage.storage import Store
from app.strategy import forward_track as ft, cross_sectional as cs

tmp = tempfile.mkdtemp()
live = Store("data")


class Split:
    """Candles and the catalog read from the live store; events written to a scratch store.

    The two reads the tracker performs are separated on purpose: the symbol catalog and the
    candle series come from whatever holds market data, while the signal record goes to the
    store that owns the run. Pointing both at one database would be simpler and would also
    mean a forward record could not exist without a market-data database, which is the wrong
    coupling for a tracking process that is meant to be independent of the data collector.
    """
    def __init__(self, market, events):
        self.market = market
        self.events = Store(events)
        self.db = market.db

    def candles(self, *a, **k):
        return self.market.candles(*a, **k)

    def record_event(self, event):
        return self.events.record_event(event)

    def record_events(self, events):
        return self.events.record_events(events)


sink = Split(live, tmp)
tracker = ft.ForwardTracker(sink, lookback=7, hold=1, quantile=0.25, cost_bps=6.0)

universe = ft.read_universe(sink, days=ft.REPLAY_READ_DAYS)
days, prices, meta = cs.align(universe)
print("universe %d symbols, %d aligned days" % (len(universe), len(days)))
print("excluded for short history:", meta.get("excluded_short_history"))
print()

recorded = 0
for index in range(cs.DEFAULT_LOOKBACK_DAYS + 1, len(days) - 1):
    day = days[index]
    cut = {}
    for symbol, bars in universe.items():
        keep = [b for b in bars if cs.day_index(b["open_time"]) <= day]
        if keep:
            cut[symbol] = keep
    if tracker.record(cut, force_day=day).get("recorded"):
        recorded += 1
print("signals recorded this run:", recorded)
print("stored signal events:", len(tracker.records(ft.SIGNAL_EVENT)))

settled = tracker.settle(universe)
print("settled:", settled["settled"])
print()
report = tracker.report()
print("=== FORWARD RECORD ===")
for key in ("signals", "settled", "pending", "missed"):
    print("  %-10s %s" % (key, report[key]))
if report.get("summary"):
    s = report["summary"]
    print("  gross_bps_mean %.1f  net_bps_mean %.1f" % (report["gross_bps_mean"], s["net_bps_mean"]))
    print("  t_stat %.2f  hit_rate %.1f%%" % (s.get("t_stat", 0), s["hit_rate"] * 100))
print()
print("=== VERDICT ===")
for line in tracker.verdict():
    print("  " + line)
shutil.rmtree(tmp, ignore_errors=True)