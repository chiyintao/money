"""Check the live forward-tracking record written by the track command."""
from app.storage.storage import Store
from app.strategy import forward_track as ft

store = Store("data")
tracker = ft.ForwardTracker(store)
report = tracker.report()
print("=== LIVE FORWARD RECORD (data/research.sqlite3) ===")
for key in ("signals", "settled", "pending", "missed", "lookback", "hold", "quantile"):
    print("  %-10s %s" % (key, report[key]))
print()
signals = tracker.records(ft.SIGNAL_EVENT)
for payload in signals:
    print("day %s: %d longs, %d shorts, universe %d"
          % (payload["day"], len(payload["longs"]), len(payload["shorts"]),
             payload["universe"]))
    print("  longs : %s" % ", ".join(sorted(payload["longs"])))
    print("  shorts: %s" % ", ".join(sorted(payload["shorts"])))
    print("  recorded_at %s" % payload["recorded_at"])
print()
print("=== VERDICT ===")
for line in tracker.verdict():
    print("  " + line)