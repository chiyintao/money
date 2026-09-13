import json
import sqlite3
import time
from pathlib import Path

def _optional_float(value):
    """A missing or unusable numeric field is stored as NULL rather than dropping the row.

    NULL, not 0.0. These are the optional order-flow columns: field 7/8/9 of a kline. A
    stored 0.0 cannot be told apart from a bar whose trade count was never recorded -- and
    every gate that reads them treats 0 as a measurement. Writing NULL is what makes
    "unmeasured" representable all the way to the gate that has to decide about it.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float('inf') else None


def _finite(value):
    """A float that can be stored, or None.

    The null check alone (value != value) lets infinity through, and an infinity in a
    research column propagates into every mean, every z-score and every correlation that
    reads it -- one bad row poisons a whole feature. Non-finite means missing here, which
    is what a nullable column is for.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float('inf'), float('-inf')):
        return None
    return number


def _optional_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class Store:
    def __init__(self, data_dir="data"):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "research.sqlite3"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init()

    def _init(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
            close_time INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
            is_closed INTEGER NOT NULL, ingest_time INTEGER NOT NULL,
            PRIMARY KEY(symbol, interval, open_time)
        );
        CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, interval, open_time DESC);
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL,
            order_type TEXT NOT NULL, limit_price REAL NOT NULL, filled_quantity REAL NOT NULL, status TEXT NOT NULL,
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, reduce_only INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT UNIQUE, timestamp INTEGER NOT NULL, symbol TEXT NOT NULL,
            side TEXT NOT NULL, entry REAL NOT NULL, exit REAL NOT NULL, qty REAL NOT NULL,
            pnl REAL NOT NULL, fees REAL NOT NULL, reason TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS equity (
            timestamp INTEGER PRIMARY KEY, cash REAL NOT NULL, total REAL NOT NULL,
            unrealized REAL NOT NULL, positions INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runtime (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS derivatives (symbol TEXT NOT NULL, event_time INTEGER NOT NULL, open_interest REAL NOT NULL, funding_rate REAL NOT NULL DEFAULT 0, mark_price REAL NOT NULL DEFAULT 0, PRIMARY KEY(symbol,event_time));
        CREATE INDEX IF NOT EXISTS idx_derivatives_time ON derivatives(symbol, event_time DESC);
        -- Derivative statistics whose detail does not fit the three original columns.
        -- Separate from derivatives because these are nullable by design: a missing
        -- endpoint must be distinguishable from a reading of zero, and the original table
        -- cannot express that.
        CREATE TABLE IF NOT EXISTS derivatives_detail (
            symbol TEXT NOT NULL, event_time INTEGER NOT NULL,
            open_interest REAL, open_interest_value REAL,
            long_short_ratio REAL, top_account_ratio REAL, global_account_ratio REAL,
            taker_buy_sell_ratio REAL, taker_buy_volume REAL, taker_sell_volume REAL,
            PRIMARY KEY(symbol, event_time)
        );
        CREATE INDEX IF NOT EXISTS idx_derivatives_detail_time ON derivatives_detail(symbol, event_time DESC);
        -- Per-minute trade flow and liquidations. The socket carries every trade print
        -- and the liquidation stream, and both were discarded after being used to trigger
        -- a decision tick; this is the bounded summary that is actually informative.
        CREATE TABLE IF NOT EXISTS flow (
            symbol TEXT NOT NULL, event_time INTEGER NOT NULL,
            buy_volume REAL NOT NULL DEFAULT 0, sell_volume REAL NOT NULL DEFAULT 0,
            trades INTEGER NOT NULL DEFAULT 0, buy_trades INTEGER NOT NULL DEFAULT 0,
            notional REAL NOT NULL DEFAULT 0, largest_trade REAL NOT NULL DEFAULT 0,
            delta_volume REAL NOT NULL DEFAULT 0, delta_ratio REAL,
            liquidations INTEGER NOT NULL DEFAULT 0,
            liquidation_long_volume REAL NOT NULL DEFAULT 0,
            liquidation_short_volume REAL NOT NULL DEFAULT 0,
            liquidation_long_notional REAL NOT NULL DEFAULT 0,
            liquidation_short_notional REAL NOT NULL DEFAULT 0,
            liquidation_delta REAL NOT NULL DEFAULT 0,
            liquidation_volume REAL NOT NULL DEFAULT 0,
            largest_liquidation REAL NOT NULL DEFAULT 0,
            liquidation_share REAL,
            PRIMARY KEY(symbol, event_time)
        );
        CREATE INDEX IF NOT EXISTS idx_flow_time ON flow(symbol, event_time DESC);
        CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, type TEXT NOT NULL, event_time INTEGER NOT NULL, payload TEXT NOT NULL);
        -- Archive for everything retention prunes out of events. Same shape, so rows can
        -- be moved between the two without reshaping, and nothing is ever lost.
        CREATE TABLE IF NOT EXISTS events_archive (event_id TEXT PRIMARY KEY, type TEXT NOT NULL, event_time INTEGER NOT NULL, payload TEXT NOT NULL);
        """)
        # Without this index "ORDER BY event_time DESC LIMIT 100" scans the whole table
        # and sorts it in a temporary B-tree. At 1.07M rows and 542 MB of payload that
        # measured 6.3 seconds per call, which is what stalled the dashboard and, through
        # the shared connection, the decision path's writes behind it.
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time DESC)')
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(type, event_time DESC)')
        try: self.db.execute('ALTER TABLE trades ADD COLUMN trade_id TEXT')
        except sqlite3.OperationalError: pass
        # Order-flow columns. The kline endpoint has always returned taker buy volume and
        # trade count and the ingest path threw both away, so the cheapest available
        # order-flow family -- which needs no extra request, no extra rate-limit budget and
        # no new data source -- was simply not recorded. Existing databases get the columns
        # added in place.
        for column, kind in (('taker_buy_volume', 'REAL'), ('quote_volume', 'REAL'),
                             ('trades', 'INTEGER')):
            try:
                self.db.execute('ALTER TABLE candles ADD COLUMN %s %s' % (column, kind))
            except sqlite3.OperationalError:
                pass
        # The flow table is new rather than extended, so an existing database needs it
        # created here; executescript above only runs for a fresh file.
        self.db.execute('''CREATE TABLE IF NOT EXISTS flow (
            symbol TEXT NOT NULL, event_time INTEGER NOT NULL,
            buy_volume REAL NOT NULL DEFAULT 0, sell_volume REAL NOT NULL DEFAULT 0,
            trades INTEGER NOT NULL DEFAULT 0, buy_trades INTEGER NOT NULL DEFAULT 0,
            notional REAL NOT NULL DEFAULT 0, largest_trade REAL NOT NULL DEFAULT 0,
            delta_volume REAL NOT NULL DEFAULT 0, delta_ratio REAL,
            liquidations INTEGER NOT NULL DEFAULT 0,
            liquidation_long_volume REAL NOT NULL DEFAULT 0,
            liquidation_short_volume REAL NOT NULL DEFAULT 0,
            liquidation_long_notional REAL NOT NULL DEFAULT 0,
            liquidation_short_notional REAL NOT NULL DEFAULT 0,
            liquidation_delta REAL NOT NULL DEFAULT 0,
            liquidation_volume REAL NOT NULL DEFAULT 0,
            largest_liquidation REAL NOT NULL DEFAULT 0,
            liquidation_share REAL,
            PRIMARY KEY(symbol, event_time))''')
        self.db.execute('CREATE INDEX IF NOT EXISTS idx_flow_time ON flow(symbol, event_time DESC)')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id)')
        self.db.commit()

    def upsert_candles(self, symbol, interval, rows, now_ms=None):
        """Single-batch write. Delegates so there is only one INSERT statement to keep right.

        The two copies had already drifted: the batch path the live loop uses and the
        single path the ingest CLI uses were separate statements, so any column added to
        one would silently be missing from the other.
        """
        written = self._upsert_candles(symbol, interval, rows, now_ms)
        self.db.commit()
        return written

    def upsert_candles_batch(self, batches):
        total = 0
        for symbol, interval, rows, now_ms in batches:
            total += self._upsert_candles(symbol, interval, rows, now_ms)
        self.db.commit()
        return total

    def _upsert_candles(self, symbol, interval, rows, now_ms=None):
        now_ms = now_ms or int(time.time() * 1000)
        clean = []
        for row in rows:
            required = ('open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume')
            if any(key not in row for key in required):
                continue
            values = [float(row[key]) for key in ('open', 'high', 'low', 'close', 'volume')]
            if min(values[:4]) <= 0 or values[1] < max(values[0], values[3]) or values[2] > min(values[0], values[3]):
                continue
            # Optional so an older caller still stores a valid bar. Absent order-flow fields
            # are written as NULL: a zero trade count does not mark anything as unknown, it
            # marks the bar as having had no trades, which is a different claim.
            extra = [_optional_float(row.get('taker_buy_volume')),
                     _optional_float(row.get('quote_volume')),
                     _optional_int(row.get('trades'))]
            clean.append((symbol, interval, int(row['open_time']), int(row['close_time']), *values,
                          int(row.get('is_closed', int(row['close_time']) <= now_ms)), now_ms, *extra))
        self.db.executemany('''
            INSERT INTO candles(symbol,interval,open_time,close_time,open,high,low,close,volume,is_closed,ingest_time,taker_buy_volume,quote_volume,trades)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol,interval,open_time) DO UPDATE SET
              close_time=excluded.close_time,open=excluded.open,high=excluded.high,low=excluded.low,
              close=excluded.close,volume=excluded.volume,is_closed=excluded.is_closed,ingest_time=excluded.ingest_time,
              taker_buy_volume=excluded.taker_buy_volume,quote_volume=excluded.quote_volume,trades=excluded.trades
        ''', clean)
        return len(clean)

    def candles(self, symbol, interval, limit=250, closed_only=True, check_gaps=False):
        """The newest ``limit`` bars, oldest first.

        With ``check_gaps`` the returned series is verified to be contiguous at ``interval``
        and a `candle_gap` event is recorded for every hole found. A series with a hole is
        not visibly broken -- a 20-bar average spanning 22 bars of elapsed time is not a
        20-bar average -- and nothing downstream can tell, because every indicator only
        ever sees a list of rows. The check is opt-in because it is O(n) on the read path
        and the live loop calls this on every tick; callers that build indicators from a
        history ask for it explicitly.
        """
        where = " AND is_closed=1" if closed_only else ""
        rows = self.db.execute(f"SELECT * FROM candles WHERE symbol=? AND interval=?{where} ORDER BY open_time DESC LIMIT ?", (symbol, interval, limit)).fetchall()
        series = [dict(row) for row in reversed(rows)]
        if check_gaps:
            self.check_candle_gaps(symbol, interval, series)
        return series

    def check_candle_gaps(self, symbol, interval, series):
        """Record a `candle_gap` event for every discontinuity in ``series``.

        Returns the list of open_times that precede a hole, so a caller can also act on it.
        """
        from ..core.config import interval_to_ms
        from ..core.domain import Event
        from .reconcile import find_gaps
        gaps = find_gaps(series, interval_to_ms(interval))
        if gaps:
            self.record_event(Event('candle_gap', {
                'symbol': symbol, 'interval': interval, 'gap_count': len(gaps),
                'first_gap_open_time': gaps[0], 'bars_returned': len(series),
            }).json())
        return gaps

    def candles_range(self, symbols, start_ms, end_ms, interval=None, limit=None,
                      page=200_000):
        """Closed candles for several symbols over a time window, oldest first.

        The existing reader takes one symbol and the newest N bars, which cannot answer
        "what was trading in this window" -- that is a range question across symbols.

        ``limit`` used to default to 500,000 and the statement ended with
        ``ORDER BY open_time ASC LIMIT ?``, so a window holding more rows returned the
        *oldest* 500,000 and dropped the remainder without saying so. A year of 5-minute
        bars across a 14-symbol tier is 1,391,212 rows, so the point-in-time
        reconstruction was reading 36% of the window: it measured the trailing 24 hours
        at a moment 234 days before the end of the data and then reported the absent 234
        days as the whole universe having stopped trading.

        The range is now paged to completion. A caller that cannot afford the rows passes
        ``limit`` and gets an exception rather than a truncated answer, because a silent
        truncation is indistinguishable from a window that was always that small.
        """
        wanted = [str(symbol) for symbol in (symbols or ()) if symbol]
        if not wanted:
            return []
        ceiling = None if limit is None else int(limit)
        size = max(1, int(page))
        marks = ','.join('?' * len(wanted))
        clauses = ['symbol IN (%s)' % marks, 'is_closed=1',
                   'close_time>=?', 'close_time<=?']
        params = [*wanted, int(start_ms), int(end_ms)]
        if interval:
            clauses.append('interval=?')
            params.append(str(interval))
        where = ' AND '.join(clauses)
        rows = []
        cursor = None
        while True:
            if cursor is None:
                page_rows = self.db.execute(
                    'SELECT * FROM candles WHERE %s ORDER BY open_time ASC, symbol ASC '
                    'LIMIT ?' % where, [*params, size]).fetchall()
            else:
                # Keyset paging rather than OFFSET: the tuple comparison is what keeps a
                # row from being skipped when several symbols share an open_time.
                page_rows = self.db.execute(
                    'SELECT * FROM candles WHERE %s AND (open_time > ? OR '
                    '(open_time = ? AND symbol > ?)) '
                    'ORDER BY open_time ASC, symbol ASC LIMIT ?' % where,
                    [*params, cursor[0], cursor[0], cursor[1], size]).fetchall()
            if not page_rows:
                break
            rows.extend(dict(row) for row in page_rows)
            cursor = (int(page_rows[-1]['open_time']), str(page_rows[-1]['symbol']))
            if ceiling is not None and len(rows) >= ceiling:
                break
            if len(page_rows) < size:
                break
        if ceiling is not None and len(rows) > ceiling:
            raise ValueError('candle_range_over_limit:%d>%d' % (len(rows), ceiling))
        return rows

    def candle_windows(self, symbols, moments, interval=None, window_ms=86_400_000):
        """Each symbol's first and last bar, and the bars in a trailing window per moment.

        This is what point-in-time reconstruction actually asks: it needs the trailing 24
        hours ending at the window start, the same ending at the end, and the listing time
        of every symbol -- not the 1.39M rows in between, which measure at 1.6 GB of
        allocation for a year of 5-minute bars across a tier.
        """
        wanted = [str(symbol) for symbol in (symbols or ()) if symbol]
        if not wanted or not moments:
            return {}
        marks = ','.join('?' * len(wanted))
        suffix = ' AND interval=?' if interval else ''
        extra = [str(interval)] if interval else []
        spans = {}
        for row in self.db.execute(
                'SELECT symbol, MIN(close_time) AS first, MAX(close_time) AS last '
                'FROM candles WHERE symbol IN (%s) AND is_closed=1%s GROUP BY symbol'
                % (marks, suffix), [*wanted, *extra]).fetchall():
            spans[str(row['symbol'])] = (int(row['first']), int(row['last']))
        out = {}
        for symbol in wanted:
            if symbol not in spans:
                continue
            first, last = spans[symbol]
            windows = {}
            for moment in moments:
                moment = int(moment)
                rows = self.db.execute(
                    'SELECT * FROM candles WHERE symbol=? AND is_closed=1 '
                    'AND close_time>? AND close_time<=?%s ORDER BY open_time ASC' % suffix,
                    [symbol, moment - int(window_ms), moment, *extra]).fetchall()
                windows[moment] = [dict(row) for row in rows]
            out[symbol] = {'listed_at': first, 'last_at': last, 'windows': windows}
        return out

    def save_order(self, order):
        payload = json.dumps(order, ensure_ascii=False)
        self.db.execute(
            'INSERT OR REPLACE INTO orders('
            'order_id,symbol,side,quantity,order_type,limit_price,filled_quantity,'
            'status,created_at,updated_at,reduce_only,payload'
            ') VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (order['order_id'], order['symbol'], order['side'], order['quantity'],
             order['order_type'], order.get('limit_price', 0),
             order.get('filled_quantity', 0), order['status'],
             order.get('created_at', 0), order.get('updated_at', 0),
             int(order.get('reduce_only', False)), payload))
        self.db.commit()

    def orders(self, active_only=False):
        from ..core.order_state import OPEN_STATUSES

        # Read from the one definition rather than a literal list, so a status added there
        # cannot be silently missing here -- which is how a restored order can be left
        # without its pending plan.
        where=" WHERE status IN (%s)" % ','.join("'%s'" % status for status in OPEN_STATUSES) if active_only else ''
        rows=self.db.execute('SELECT payload FROM orders'+where+' ORDER BY created_at ASC').fetchall()
        return [json.loads(row['payload']) for row in rows]

    def record_trade(self, trade):
        trade_id = trade.get('trade_id') or json.dumps(trade, sort_keys=True, ensure_ascii=False)
        self.db.execute("INSERT OR IGNORE INTO trades(trade_id,timestamp,symbol,side,entry,exit,qty,pnl,fees,reason,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (trade_id, int(time.time()*1000), trade["symbol"], trade["side"], trade["entry"], trade["exit"], trade["qty"], trade["pnl"], trade.get("fees", 0), trade.get("reason", "unknown"), json.dumps(trade, ensure_ascii=False)))
        self.db.commit()

    @staticmethod
    def _trade_row(row):
        item=dict(row)
        try:
            payload=json.loads(item.get('payload') or '{}')
        except (TypeError,json.JSONDecodeError):
            payload={}
        merged={**item,**payload}
        merged['timestamp']=int(merged.get('exit_time') or item.get('timestamp') or 0)
        merged.setdefault('entry_price',merged.get('entry',0))
        merged.setdefault('exit_price',merged.get('exit',0))
        merged.setdefault('gross_pnl',float(merged.get('pnl',0))+float(merged.get('fees',0))+float(merged.get('funding',0)))
        entry_notional=float(merged.get('entry_notional') or float(merged.get('entry',0))*float(merged.get('qty',0)))
        merged.setdefault('entry_notional',entry_notional)
        merged.setdefault('exit_notional',float(merged.get('exit',0))*float(merged.get('qty',0)))
        merged.setdefault('pnl_pct',float(merged.get('pnl',0))/entry_notional*100 if entry_notional else 0.0)
        merged.pop('payload',None)
        return merged

    def recent_trades(self, limit=50):
        rows = self.db.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [self._trade_row(row) for row in rows]

    def trades_for_session(self, session_id, limit=500):
        if not session_id:
            return []
        rows = self.db.execute("SELECT * FROM trades WHERE json_extract(payload,'$.session_id')=? ORDER BY timestamp DESC LIMIT ?", (session_id, limit)).fetchall()
        return [self._trade_row(row) for row in rows]

    def all_trades(self):
        rows = self.db.execute("SELECT * FROM trades ORDER BY timestamp ASC, id ASC").fetchall()
        return [self._trade_row(row) for row in rows]

    def session_capitals(self):
        """Starting capital per session id, from the started events.

        Needed because trade rows carry a dollar PnL but no denominator, and the
        denominator is not constant: the session form lets the operator choose it, so the
        table holds trades made against 100 and against 10,000. Adding those dollars
        together produces a number that is not a quantity of anything -- a 7-dollar loss
        on a 100-dollar account is a 7% drawdown, and summing it with a 7-dollar loss on a
        10,000-dollar account reports 14 dollars of damage where the real damage is 7.07%.
        """
        capitals = {}
        rows = self.db.execute(
            "SELECT payload FROM events WHERE type='simulation_started'").fetchall()
        for row in rows:
            payload = json.loads(row['payload'])
            session_id = payload.get('session_id')
            cash = payload.get('initial_cash')
            if session_id and cash:
                capitals[session_id] = float(cash)
        return capitals

    def record_equity(self, cash, total, unrealized, positions):
        last = self.db.execute('SELECT COALESCE(MAX(timestamp), 0) FROM equity').fetchone()[0]
        timestamp = max(int(time.time()*1000), int(last) + 1)
        self.db.execute("INSERT INTO equity(timestamp,cash,total,unrealized,positions) VALUES(?,?,?,?,?)", (timestamp, cash, total, unrealized, positions))
        self.db.commit()

    def equity_curve(self, limit=180):
        rows = self.db.execute("SELECT timestamp,total,unrealized,positions FROM equity ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def record_flow(self, rows):
        """Per-minute trade flow and liquidation summary."""
        columns = ('buy_volume', 'sell_volume', 'trades', 'buy_trades', 'notional',
                   'largest_trade', 'delta_volume', 'delta_ratio', 'liquidations',
                   'liquidation_long_volume', 'liquidation_short_volume',
                   'liquidation_long_notional', 'liquidation_short_notional',
                   'liquidation_delta', 'liquidation_volume', 'largest_liquidation',
                   'liquidation_share')
        clean = []
        for row in rows or ():
            try:
                symbol = str(row['symbol'])
                event_time = int(row['event_time'])
            except (KeyError, TypeError, ValueError):
                continue
            clean.append((symbol, event_time,
                          *[_finite(row.get(column)) for column in columns]))
        if not clean:
            return 0
        self.db.executemany(
            'INSERT OR REPLACE INTO flow(symbol,event_time,%s) VALUES(%s)'
            % (','.join(columns), ','.join('?' * (len(columns) + 2))), clean)
        self.db.commit()
        return len(clean)

    def flow(self, symbol=None, limit=200, since=None, until=None):
        """Recent flow rows, oldest first.

        The since argument is what makes this usable beside candles: a backtest needs the
        flow that existed at each bar, which means a bounded window rather than the last N.

        Both ends, though. A lower bound alone is not a window: the statement ends in
        ORDER BY event_time DESC LIMIT n, so on a table longer than n rows "everything at or
        after since" returns the newest n rows overall -- which for an early bar are all in
        its future, and the caller that then filters to "not after this bar" gets an empty
        window and reports a measured zero. The upper bound is what makes the read a range.
        """
        clauses = []
        params = []
        if symbol:
            clauses.append('symbol=?')
            params.append(symbol)
        if since is not None:
            clauses.append('event_time>=?')
            params.append(int(since))
        if until is not None:
            clauses.append('event_time<=?')
            params.append(int(until))
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        params.append(int(limit))
        rows = self.db.execute('SELECT * FROM flow%s ORDER BY event_time DESC LIMIT ?'
                               % where, params).fetchall()
        return [dict(row) for row in reversed(rows)]

    def prune_flow(self, keep_rows, batch=20000):
        """Bound the flow table the same way the audit archive is bounded."""
        total = self.db.execute('SELECT COUNT(*) FROM flow').fetchone()[0]
        removed = 0
        while total - removed > keep_rows:
            chunk = min(batch, total - removed - keep_rows)
            self.db.execute('DELETE FROM flow WHERE rowid IN ('
                            'SELECT rowid FROM flow ORDER BY event_time ASC LIMIT ?)', (chunk,))
            self.db.commit()
            removed += chunk
        return removed

    def record_derivatives_detail(self, rows):
        """Partial-column write for the collected derivative statistics.

        Only the keys present in a row are written, so a run that got open interest but not
        the ratio endpoints does not overwrite the ratio history with nulls.
        """
        columns = ('open_interest', 'open_interest_value', 'long_short_ratio',
                   'top_account_ratio', 'global_account_ratio', 'taker_buy_sell_ratio',
                   'taker_buy_volume', 'taker_sell_volume')
        clean = []
        for row in rows:
            try:
                symbol = str(row['symbol'])
                event_time = int(row['event_time'])
            except (KeyError, TypeError, ValueError):
                continue
            # Same finite check as the flow writer: a nullable column exists precisely so
            # an unusable reading can be stored as absent rather than as a number.
            clean.append((symbol, event_time,
                          *[_finite(row.get(column)) for column in columns]))
        if not clean:
            return 0
        self.db.executemany(
            'INSERT OR REPLACE INTO derivatives_detail(symbol,event_time,%s) VALUES(%s)'
            % (','.join(columns), ','.join('?' * (len(columns) + 2))), clean)
        self.db.commit()
        return len(clean)

    def derivatives_detail(self, symbol=None, limit=200, since=None, until=None):
        """Open interest and ratio rows, oldest first.

        The since bound is what makes this usable beside candles: a feature row needs the
        positioning that existed at its bar, which is a bounded window rather than the
        last N rows overall. The default is unchanged, so an existing caller reading the
        newest snapshot still gets it.
        """
        clauses, params = [], []
        if symbol:
            clauses.append('symbol=?')
            params.append(symbol)
        if since is not None:
            clauses.append('event_time>=?')
            params.append(int(since))
        if until is not None:
            clauses.append('event_time<=?')
            params.append(int(until))
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        params.append(int(limit))
        rows = self.db.execute('SELECT * FROM derivatives_detail' + where +
                               ' ORDER BY event_time DESC LIMIT ?', tuple(params)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def record_derivatives(self, rows):
        self.db.executemany('INSERT OR REPLACE INTO derivatives(symbol,event_time,open_interest,funding_rate,mark_price) VALUES(?,?,?,?,?)', [(x['symbol'],int(x.get('event_time',0)),float(x.get('open_interest',0)),float(x.get('funding_rate',0)),float(x.get('mark_price',0))) for x in rows])
        self.db.commit()

    def derivatives(self, symbol=None, limit=200):
        if symbol: rows=self.db.execute('SELECT * FROM derivatives WHERE symbol=? ORDER BY event_time DESC LIMIT ?', (symbol,limit)).fetchall()
        else: rows=self.db.execute('SELECT * FROM derivatives ORDER BY event_time DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    def record_event(self,event):
        self.record_events([event])

    def record_events(self, events):
        self.db.executemany('INSERT OR IGNORE INTO events(event_id,type,event_time,payload) VALUES(?,?,?,?)', [(event['event_id'], event['type'], event['event_time'], json.dumps(event['payload'], ensure_ascii=False)) for event in events])
        self.db.commit()

    def recent_events(self,limit=100):
        rows=self.db.execute('SELECT * FROM events ORDER BY event_time DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    # Types whose whole purpose is the audit trail a decision is reconstructed from.
    DURABLE_EVENT_TYPES = ('simulation_started', 'simulation_ended', 'simulation_liquidated',
                           'fill', 'liquidation', 'order_filled', 'order_canceled')

    def prune_events(self, keep_rows, batch=5000, keep_durable=True):
        """Move all but the newest keep_rows audit rows into events_archive.

        Retention is by row count, not by age. The table held 1.07M rows covering only
        0.92 days, of which 758k were strategy_decision rows that collapsed to 32
        distinct decisions per 5000. Volume here scales with tick rate, not with time, so
        a time window cannot bound it while a row window bounds the cost of every query
        that reads the table.

        Rows are copied to the archive before being deleted, so nothing is lost. Durable
        types are kept regardless of age. Returns a count per moved type.
        """
        keep_rows = max(0, int(keep_rows))
        if keep_rows == 0:
            # Nothing is off limits, so every candidate is prunable.
            cutoff = None
        else:
            # The oldest timestamp still inside the window. Everything strictly older
            # than it goes; rows sharing that timestamp stay, so the table can exceed
            # keep_rows by however many events landed in the same millisecond.
            row = self.db.execute(
                'SELECT event_time FROM events ORDER BY event_time DESC LIMIT 1 OFFSET ?',
                (keep_rows - 1,)).fetchone()
            if row is None:
                return {}
            cutoff = int(row[0])
        moved = {}
        while True:
            where = '1=1' if cutoff is None else 'event_time < ?'
            params = [] if cutoff is None else [cutoff]
            if keep_durable:
                placeholders = ','.join('?' for _ in self.DURABLE_EVENT_TYPES)
                where += f" AND type NOT IN ({placeholders})"
                params.extend(self.DURABLE_EVENT_TYPES)
            rows = self.db.execute(
                f"SELECT event_id,type FROM events WHERE {where} LIMIT ?",
                (*params, batch)).fetchall()
            if not rows:
                break
            ids = [row['event_id'] for row in rows]
            marks = ','.join('?' for _ in ids)
            self.db.execute(
                f"INSERT OR IGNORE INTO events_archive(event_id,type,event_time,payload) "
                f"SELECT event_id,type,event_time,payload FROM events WHERE event_id IN ({marks})", ids)
            self.db.execute(f"DELETE FROM events WHERE event_id IN ({marks})", ids)
            self.db.commit()
            for row in rows:
                moved[row['type']] = moved.get(row['type'], 0) + 1
            if len(ids) < batch:
                break
        return moved

    def prune_archive(self, keep_rows, batch=20000):
        """Delete the oldest archive rows beyond a row count.

        The archive had no retention at all and no reader. Every row retention moved out
        of the hot table accumulated there permanently -- 1.02M rows and growing -- so the
        cost of the audit trail grew without bound while the trail itself was unreadable.
        Bounding it, and giving it a reader, are the two halves of making the claim that
        nothing is lost actually true.
        """
        keep_rows = int(keep_rows)
        if keep_rows <= 0:
            return 0
        row = self.db.execute(
            'SELECT event_time FROM events_archive ORDER BY event_time DESC LIMIT 1 OFFSET ?',
            (keep_rows - 1,)).fetchone()
        if row is None:
            return 0
        cutoff = int(row[0])
        removed = 0
        while True:
            ids = [item['event_id'] for item in self.db.execute(
                'SELECT event_id FROM events_archive WHERE event_time < ? LIMIT ?',
                (cutoff, batch)).fetchall()]
            if not ids:
                break
            marks = ','.join('?' for _ in ids)
            self.db.execute(f'DELETE FROM events_archive WHERE event_id IN ({marks})', ids)
            self.db.commit()
            removed += len(ids)
            if len(ids) < batch:
                break
        return removed

    def archived_events(self, event_type=None, limit=200, symbol=None, session_id=None):
        """Read back what retention moved. Newest last, matching recent_events."""
        where, params = ['1=1'], []
        if event_type:
            where.append('type=?')
            params.append(event_type)
        if symbol:
            where.append("json_extract(payload,'$.symbol')=?")
            params.append(symbol)
        if session_id:
            where.append("json_extract(payload,'$.session_id')=?")
            params.append(session_id)
        rows = self.db.execute(
            'SELECT event_id,type,event_time,payload FROM events_archive WHERE %s '
            'ORDER BY event_time DESC LIMIT ?' % ' AND '.join(where),
            (*params, int(limit))).fetchall()
        return [{'event_id': row['event_id'], 'type': row['type'],
                 'event_time': row['event_time'], 'payload': json.loads(row['payload'])}
                for row in reversed(rows)]

    def event_counts(self):
        row = self.db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0) FROM events").fetchone()
        archived = self.db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0) FROM events_archive").fetchone()
        return {'events': row[0], 'payload_bytes': row[1],
                'archived': archived[0], 'archived_bytes': archived[1],
                'archive_by_type': {item['type']: item['n'] for item in self.db.execute(
                    'SELECT type, COUNT(*) AS n FROM events_archive GROUP BY type '
                    'ORDER BY n DESC LIMIT 12').fetchall()}}

    def vacuum(self):
        """Reclaim the disk the pruned rows occupied. Not run automatically: it
        rewrites the file and can take minutes on a large database."""
        self.db.execute('VACUUM')
        self.db.commit()

    def set_runtime(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO runtime(key,value) VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    def simulation_history(self, limit=50):
        sessions = {}
        rows = self.db.execute("SELECT type,event_time,payload FROM events WHERE type IN ('simulation_started','simulation_ended','simulation_liquidated') ORDER BY event_time ASC").fetchall()
        for row in rows:
            payload=json.loads(row['payload']); sid=payload.get('session_id')
            if not sid: continue
            item=sessions.setdefault(sid, {'session_id':sid,'started_at':None,'ended_at':None,'status':'unknown','initial_cash':None,'leverage':None,'source':None,'symbol_count':None,'end_reason':'','trades':0,'net_pnl':0.0,'symbols':[],'trade_details':[]})
            if row['type']=='simulation_started': item.update({'started_at':row['event_time'],'status':'running','initial_cash':payload.get('initial_cash'),'leverage':payload.get('leverage'),'source':payload.get('source'),'symbol_count':payload.get('symbol_count')})
            elif row['type']=='simulation_ended':
                item.update({'ended_at':row['event_time'],'status':'ended','end_reason':payload.get('reason','manual'),'final_equity':payload.get('final_equity',payload.get('equity'))})
                for key in ('return_pct','net_pnl','trades','wins','losses','win_rate_pct','gross_profit','gross_loss','profit_factor','total_fees','total_funding','average_holding_ms','best_trade','worst_trade','decision_count','selected_symbols','symbols','duration_ms'):
                    if key in payload: item[key]=payload[key]
                if payload.get('trades_detail'): item['trade_details']=list(payload['trades_detail'])
            else: item.update({'ended_at':row['event_time'],'status':'liquidated','end_reason':'maintenance_margin'})
        for row in self.db.execute("SELECT payload FROM trades ORDER BY timestamp ASC").fetchall():
            trade=json.loads(row['payload']); sid=trade.get('session_id')
            if sid in sessions:
                item=sessions[sid]
                if not item.get('trade_details'): item['trades']+=1; item['net_pnl']+=float(trade.get('pnl',0)); item['trade_details'].append(trade)
                if trade.get('symbol') and trade['symbol'] not in item['symbols']: item['symbols'].append(trade['symbol'])
        for item in sessions.values():
            details=item.get('trade_details',[])
            wins=[float(x.get('pnl',0)) for x in details if float(x.get('pnl',0))>0]
            losses=[float(x.get('pnl',0)) for x in details if float(x.get('pnl',0))<=0]
            pnls = [float(x.get('pnl', 0)) for x in details]
            wins = [v for v in pnls if v > 0]
            losses = [v for v in pnls if v <= 0]
            gross_loss = abs(sum(losses))
            # One dict per session rather than one statement per statistic: the single-
            # expression form ran to nearly eight hundred characters and mixed eleven
            # different measurements into a line that could not be read or diffed.
            item.update({
                'wins': len(wins),
                'losses': len(losses),
                'win_rate_pct': len(wins) / len(details) * 100 if details else 0.0,
                'gross_profit': sum(wins),
                'gross_loss': gross_loss,
                'profit_factor': sum(wins) / gross_loss if gross_loss else None,
                'total_fees': sum(float(x.get('fees', 0)) for x in details),
                'total_funding': sum(float(x.get('funding', 0)) for x in details),
                'average_holding_ms': (
                    sum(float(x.get('holding_ms', 0)) for x in details) / len(details)
                    if details else 0.0),
                'best_trade': max(pnls, default=0.0),
                'worst_trade': min(pnls, default=0.0),
                'avg_trade_pnl': item['net_pnl'] / len(details) if details else 0.0,
                'training_label': ('profitable' if item['net_pnl'] > 0
                                   else 'loss' if item['net_pnl'] < 0 else 'flat'),
            })
            curve=[float(item.get('initial_cash') or 0)]; peak=curve[0]; max_dd=0.0
            for trade in details:
                curve.append(curve[-1]+float(trade.get('pnl',0))); peak=max(peak,curve[-1]); max_dd=max(max_dd,(peak-curve[-1])/peak*100 if peak else 0.0)
            item['max_drawdown_pct']=max_dd
            item['duration_ms']=max(0,(item.get('ended_at') or item.get('started_at') or 0)-(item.get('started_at') or 0))
            symbol_stats={}
            for trade in details:
                symbol=trade.get('symbol','UNKNOWN'); stat=symbol_stats.setdefault(symbol,{'symbol':symbol,'trades':0,'wins':0,'net_pnl':0.0,'fees':0.0,'avg_pnl':0.0})
                pnl=float(trade.get('pnl',0)); stat['trades']+=1; stat['wins']+=1 if pnl>0 else 0; stat['net_pnl']+=pnl; stat['fees']+=float(trade.get('fees',0))
            for stat in symbol_stats.values():
                stat['win_rate_pct']=stat['wins']/stat['trades']*100 if stat['trades'] else 0.0; stat['avg_pnl']=stat['net_pnl']/stat['trades'] if stat['trades'] else 0.0
            item['symbol_stats']=sorted(symbol_stats.values(),key=lambda x:x['net_pnl'],reverse=True)
            item.pop('trade_details',None)
        return sorted(sessions.values(), key=lambda x:x.get('started_at') or 0, reverse=True)[:limit]

    def get_runtime(self, key, default=None):
        row=self.db.execute("SELECT value FROM runtime WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row['value'])

    def close(self):
        self.db.close()
