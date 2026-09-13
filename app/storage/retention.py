"""Audit retention: move aged event rows out of the hot table.

The events table reached 1.07M rows and 542 MB, dominated by market_event and repeated
strategy_decision rows that no decision reads. Ordering by event_time without an index
scanned and sorted the whole table: recent_events(100) measured 6.3 seconds, which
stalled the dashboard and, through the shared connection, the decision path behind it.

Retention is by ROW COUNT, not by age. The table held 1.07M rows covering only 0.92
days, of which 758k were strategy_decision rows that collapsed to 32 distinct decisions
per 5000. Volume scales with tick rate, not with time, so a row window is what actually
bounds the cost of every query that reads the table.

  python -m app.retention --stats
  python -m app.retention --prune                 # keep the newest EVENT_RETENTION_ROWS
  python -m app.retention --prune --rows 20000
  python -m app.retention --archive-session <session_id>
  python -m app.retention --vacuum
"""
import argparse
import json
import sys
import time

from ..core.config import Settings
from .storage import Store


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024:
            return '%.1f %s' % (n, unit)
        n /= 1024
    return '%.1f TB' % n


def cmd_stats(store):
    counts = store.event_counts()
    print(json.dumps(counts, indent=2))
    print()
    print('%-24s %10s' % ('type', 'rows'))
    for row in store.db.execute("SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY 2 DESC LIMIT 20"):
        print('%-24s %10d' % (row[0], row[1]))


def cmd_prune(store, keep_rows, batch, vacuum, durable=True):
    started = time.time()
    moved = store.prune_events(keep_rows, batch=batch, keep_durable=durable)
    total = sum(moved.values())
    print('archived %d rows, keeping the newest %d in the hot table (%.1fs)'
          % (total, keep_rows, time.time() - started))
    for kind, count in sorted(moved.items(), key=lambda item: -item[1]):
        print('  %-24s %8d' % (kind, count))
    if not total:
        print('  (nothing to prune)')
    print()
    print(json.dumps(store.event_counts(), indent=2))
    if vacuum:
        print('vacuuming...')
        started = time.time()
        store.vacuum()
        print('vacuum finished in %.1fs' % (time.time() - started))


def cmd_archive_session(store, session_id):
    """Move every non-durable row belonging to one session out of the hot table."""
    types = Store.DURABLE_EVENT_TYPES
    placeholders = ','.join('?' for _ in types)
    rows = store.db.execute(
        f"SELECT event_id FROM events WHERE json_extract(payload,'$.session_id')=? "
        f"AND type NOT IN ({placeholders})", (session_id, *types)).fetchall()
    ids = [row['event_id'] for row in rows]
    if not ids:
        print('no rows to archive for session', session_id)
        return
    marks = ','.join('?' for _ in ids)
    store.db.execute(f"INSERT OR IGNORE INTO events_archive(event_id,type,event_time,payload) "
                     f"SELECT event_id,type,event_time,payload FROM events WHERE event_id IN ({marks})", ids)
    store.db.execute(f"DELETE FROM events WHERE event_id IN ({marks})", ids)
    store.db.commit()
    print('archived %d rows for session %s' % (len(ids), session_id))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--stats', action='store_true', help='show table sizes and event counts')
    parser.add_argument('--prune', action='store_true', help='archive everything but the newest --rows')
    parser.add_argument('--rows', type=int, default=None, help='hot-table size (default from Settings)')
    parser.add_argument('--no-durable', action='store_true',
                        help='also prune fills and session lifecycle rows')
    parser.add_argument('--batch', type=int, default=5000)
    parser.add_argument('--vacuum', action='store_true', help='reclaim disk after pruning')
    parser.add_argument('--archive-session', default=None, metavar='SESSION_ID')
    args = parser.parse_args(argv)

    settings = Settings()
    store = Store(args.data_dir or settings.data_dir)
    try:
        if args.archive_session:
            cmd_archive_session(store, args.archive_session)
        if args.prune:
            rows = args.rows if args.rows is not None else settings.event_retention_rows
            cmd_prune(store, rows, args.batch, args.vacuum,
                      durable=not args.no_durable and settings.event_retention_durable)
        elif args.vacuum:
            print('vacuuming...')
            started = time.time()
            store.vacuum()
            print('vacuum finished in %.1fs' % (time.time() - started))
        if args.stats or not (args.prune or args.vacuum or args.archive_session):
            cmd_stats(store)
    finally:
        store.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
