"""Read Binance's public data archives.

The REST API only serves the recent past -- `/futures/data/*` keeps 30 days and the
kline endpoint a few thousand bars -- but `data.binance.vision` keeps the monthly and
daily archives permanently. Anything older than the API window has exactly one source,
and it is this one.

Three properties of the archives shape the code:

* **They are CSV inside a zip**, one file per symbol per (interval, month) or per day.
  Column layout is positional and has changed over time (an `ignore` column, a `count`
  header where the live API calls it `trades`), so columns are read by header name with a
  positional fallback rather than by index.
* **A missing archive answers 404, not an error.** The archive for the current, incomplete
  month does not exist yet, and a symbol that listed mid-month has no earlier file. That
  is a normal answer and is reported as `missing`, not raised.
* **Downloads are large and repeat**, so they land in a local cache directory. Refetching
  a 400KB zip per symbol per month across a year is minutes of pointless traffic, and the
  archives are immutable once published.
"""
import csv
import http.client
import io
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_ROOT = "https://data.binance.vision/data"
DEFAULT_TIMEOUT = 60
USER_AGENT = "money-backfill/1.0"


class ArchiveError(Exception):
    """A download that failed for a reason other than "no such archive"."""


def archive_url(market, frequency, kind, symbol, filename, interval=None):
    """data.binance.vision URL for one archive file.

    `market` is `futures/um`, `futures/cm` or `spot`; `frequency` is `monthly` or `daily`;
    `kind` is `klines`, `metrics`, `trades` and so on.

    `klines` is the one kind filed under an interval directory, so it is the one kind that
    needs `interval`:

        <market>/<frequency>/klines/<symbol>/<interval>/<symbol>-<interval>-<period>.zip
        <market>/<frequency>/metrics/<symbol>/<symbol>-metrics-<day>.zip

    Omitting that segment does not fail loudly. The CDN answers a well-formed 404 whose body
    names the key it looked for, so a caller that treats 404 as "this archive does not
    exist" records a permanent gap in the data and reports nothing wrong. That is exactly
    what happened: every one of 154 monthly klines archives was reported missing, the
    three order-flow columns stayed at 0.8% coverage, and the features derived from them
    could never be trained. Requiring the argument turns a silent 404 back into an error.
    """
    parts = [ARCHIVE_ROOT, market, frequency, kind, symbol]
    if kind == "klines":
        if not interval:
            raise ValueError("interval_required_for_klines")
        parts.append(str(interval))
    parts.append(filename)
    return "/".join(parts)


def klines_filename(symbol, interval, period):
    """`BTCUSDT-5m-2024-01.zip` for a month, `...-2024-01-05.zip` for a day."""
    return "%s-%s-%s.zip" % (symbol, interval, period)


def metrics_filename(symbol, day):
    """`BTCUSDT-metrics-2024-01-05.zip`. Metrics are published daily only."""
    return "%s-metrics-%s.zip" % (symbol, day)


# What the CDN actually does, measured rather than assumed.
#
# The host resolves to four edge addresses. In one window, three of them (13.35.190.101,
# .114, .23) served an immutable 2024 archive in 0.3-0.4s while the fourth (.120) refused
# connections outright. Python resolves one address per attempt and does not try the next,
# so roughly a quarter of all fetches hang on a node that will never answer, and the
# timeout surfaces as a failure with no relation to the request. curl escaped it only
# because it walks the address list itself.
#
# Separately, the same URL was observed returning 404 for minutes at a time and 200 again
# afterwards, on both clients. That is why a 404 is retried rather than believed.
#
# A short retry crosses neither. The failure mode if either is taken at face value is the
# worst available: a backfill reports success having silently skipped whole months.
# The budget is a compromise: long enough to cross a short bad window, short enough that a
# backfill over hundreds of files is not dominated by waiting. Each window is retried
# separately, so a run that continues can still finish everything a stalled one could not.
MISSING_RETRIES = 3
MISSING_RETRY_DELAY = 5.0
MISSING_RETRY_BACKOFF = 3.0


def _resolve(host, port=443):
    """Every IPv4 address for `host`, in resolution order, or empty if it fails."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    ordered = []
    for info in infos:
        address = info[4][0]
        if address not in ordered:
            ordered.append(address)
    return ordered


def _download_any_address(request, url, timeout):
    """Fetch, trying each resolved address until one answers.

    This is not belt-and-braces. The archive host answers from four addresses and one of
    them refuses connections while the others serve the same immutable file in 0.3s, and
    `urllib` opens only the first address it is handed. Roughly a quarter of all fetches
    would therefore hang on a node that will never answer, with a timeout that says nothing
    about the request. curl hides the problem by walking the address list itself -- which is
    exactly why the same URL can look healthy from a shell and hang from Python.

    Host and TLS SNI stay the real hostname; only the socket target changes, so a shared CDN
    still routes and certifies correctly. The per-address timeout is the caller's budget
    split across the addresses, so a dead node costs seconds rather than the whole budget.
    """
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = _resolve(parts.hostname, port)
    if not addresses:
        # Nothing to walk; let urllib report the resolution failure itself.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    # Enough to open a connection and read a small file, but bounded so one dead node
    # cannot consume the caller's whole budget before the next address is tried.
    per_address = max(4.0, min(15.0, float(timeout) / max(1, len(addresses))))
    last = None
    not_found = None
    for address in _ordered_addresses(addresses):
        # The socket goes to the chosen address, but TLS SNI and the Host header stay the
        # real hostname. Sending an IP in either place fails: the CDN answers an IP SNI
        # with SSLV3_ALERT_HANDSHAKE_FAILURE, because no certificate matches a bare IP.
        # That is why this cannot be done by rewriting the URL and calling urllib.
        try:
            body = _fetch_from(address, parts, port, per_address)
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise ArchiveError("%s: HTTP %s" % (url, exc.code)) from exc
            # A 404 that arrives as an HTTP response is an answer about the archive, and it
            # comes from a node that is demonstrably working. Deliberately NOT requiring
            # every address to agree: one address here is routinely dead, so requiring
            # unanimity would make it impossible to ever conclude "absent", and every
            # genuinely missing archive would surface as a network error instead.
            not_found = exc
            continue
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            # A connection that never completed says nothing about the archive.
            last = exc
            continue
        _remember_address(address)
        return body
    if not_found is not None:
        raise not_found
    raise ArchiveError("%s: every resolved address failed (%s)" % (url, last))


# The last address that served a file. The host resolves to four addresses and one of them
# is routinely dead, so without this every request pays that node's timeout before reaching
# a working one -- on a backfill of thousands of files that is the difference between
# minutes and hours. Purely an ordering hint: if it stops working, the walk moves on.
_LAST_GOOD_ADDRESS = None


def _ordered_addresses(addresses):
    """`addresses`, with the most recently successful one first when it is still offered."""
    if _LAST_GOOD_ADDRESS in addresses:
        return [_LAST_GOOD_ADDRESS] + [a for a in addresses if a != _LAST_GOOD_ADDRESS]
    return list(addresses)


def _remember_address(address):
    global _LAST_GOOD_ADDRESS
    _LAST_GOOD_ADDRESS = address


def _fetch_from(address, parts, port, timeout):
    """One request against one address, with SNI and Host set to the real hostname.

    A non-2xx answer is raised as HTTPError so the retry logic above can treat a 404 as a
    possible absence, exactly as it does for the plain urllib path.
    """
    if parts.scheme == "https":
        context = ssl.create_default_context()
        raw = socket.create_connection((address, port), timeout=timeout)
        try:
            sock = context.wrap_socket(raw, server_hostname=parts.hostname)
        except Exception:
            raw.close()
            raise
        connection = http.client.HTTPSConnection(parts.hostname, port, timeout=timeout,
                                                 context=context)
        connection.sock = sock
    else:
        connection = http.client.HTTPConnection(address, port, timeout=timeout)
    try:
        target = urllib.parse.urlunsplit(("", "", parts.path, parts.query, ""))
        connection.request("GET", target, headers={"Host": parts.netloc,
                                                   "User-Agent": USER_AGENT,
                                                   "Accept": "*/*",
                                                   "Connection": "close"})
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise urllib.error.HTTPError(parts.geturl(), response.status, response.reason,
                                         response.headers, None)
        return body
    finally:
        connection.close()


def fetch(url, cache_dir=None, timeout=DEFAULT_TIMEOUT, retries=None, delay=None):
    """The bytes at `url`, from the cache when present.

    Returns None when the archive does not exist, and raises ArchiveError for anything
    that is not an answer about whether it exists. The distinction matters in one direction
    only: a network problem must never be mistaken for a missing month, because the
    backfill would then report success having written nothing.

    Each attempt walks the host's resolved addresses until one connects; see
    `_download_any_address` for why that is necessary rather than defensive.

    A 404 or 403 is retried on an exponential backoff before being accepted as absence; see
    MISSING_RETRIES. Because that budget is minutes long, a caller that already knows the
    archive is absent -- or has a cached copy -- should pass retries=0 rather than pay it.
    """
    if cache_dir is not None:
        path = Path(cache_dir) / url.rsplit("/", 1)[-1]
        if path.exists() and path.stat().st_size > 0:
            return path.read_bytes()
    else:
        path = None
    attempts = (MISSING_RETRIES if retries is None else int(retries)) + 1
    wait = MISSING_RETRY_DELAY if delay is None else float(delay)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(attempts):
        try:
            payload = _download_any_address(request, url, timeout)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise ArchiveError("%s: HTTP %s" % (url, exc.code)) from exc
            if attempt == attempts - 1:
                return None
            last_error = "HTTP %s" % exc.code
            time.sleep(wait * (MISSING_RETRY_BACKOFF ** attempt))
        except (urllib.error.URLError, TimeoutError, OSError, ArchiveError) as exc:
            # A connection failure is NOT an absent archive, so it must not fall through to
            # the `return None` below -- that is the confusion this module exists to prevent,
            # and it is why ArchiveError is listed here rather than swallowed.
            last_error = "%s: %s" % (type(exc).__name__, getattr(exc, "reason", exc))
            if attempt == attempts - 1:
                raise ArchiveError("%s: %s" % (url, last_error)) from exc
            time.sleep(min(2.0, wait) * (attempt + 1))
    else:
        raise ArchiveError("%s: %s" % (url, last_error or "no attempt succeeded"))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and moved, so an interrupted download cannot leave a
        # truncated file that the cache would then serve as complete forever.
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return payload


def exists(url, timeout=30, cache_dir=None):
    """Whether the archive is published, without keeping the bytes.

    Uses the same retry rules as fetch, for the same reason. A backfill calls this to plan
    a run, so the plan lists the archives that are actually there and a month that is
    genuinely absent is reported up front rather than discovered mid-download.
    """
    try:
        return fetch(url, cache_dir=cache_dir, timeout=timeout) is not None
    except ArchiveError:
        return False


def read_csv(payload):
    """Rows from a zipped single-CSV archive, as dicts keyed by header.

    Returns (header, rows). Blank lines are skipped: several archives end with one, and an
    empty row would otherwise parse as a bar of zeros.
    """
    if payload is None:
        return [], []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            return [], []
        text = archive.read(names[0]).decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = []
    header = None
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if header is None:
            header = [cell.strip() for cell in row]
            continue
        rows.append(row)
    return header or [], rows


def _column(header, names, position):
    """Index of the first matching header name, or `position` when none matches.

    Archives have renamed columns between vintages (`count` became `trades` in the live
    API) and appended an `ignore` column, so a fixed index silently reads the wrong field
    on the months where the layout differs.
    """
    lowered = [name.strip().lower() for name in (header or [])]
    for name in names:
        if name.lower() in lowered:
            return lowered.index(name.lower())
    return position


def _value(row, index, default=None):
    if index is None or index >= len(row):
        return default
    text = row[index].strip()
    return default if text == "" else text


def parse_kline_rows(header, rows):
    """Archive kline rows in the shape `Store.upsert_candles` accepts.

    The archive carries the three order-flow columns the live REST snapshot also carries,
    which is the whole point: 99.9% of the stored bars were written before those columns
    were collected, so `taker_buy_volume`, `quote_volume` and `trades` are NULL for
    almost the entire history and every order-flow feature is unbuildable from it.
    """
    open_i = _column(header, ["open_time", "open time"], 0)
    close_i = _column(header, ["close_time", "close time"], 6)
    count_i = _column(header, ["count", "trades", "number_of_trades"], 8)
    taker_i = _column(header, ["taker_buy_volume"], 9)
    quote_i = _column(header, ["quote_volume"], 7)
    parsed = []
    for row in rows:
        try:
            open_time = int(float(_value(row, open_i, 0)))
            close_time = int(float(_value(row, close_i, 0)))
            bar = {"open_time": open_time, "close_time": close_time,
                   "open": float(_value(row, _column(header, ["open"], 1), 0)),
                   "high": float(_value(row, _column(header, ["high"], 2), 0)),
                   "low": float(_value(row, _column(header, ["low"], 3), 0)),
                   "close": float(_value(row, _column(header, ["close"], 4), 0)),
                   "volume": float(_value(row, _column(header, ["volume"], 5), 0)),
                   "is_closed": 1}
        except (TypeError, ValueError):
            continue
        quote = _value(row, quote_i)
        taker = _value(row, taker_i)
        count = _value(row, count_i)
        if quote is not None:
            try:
                bar["quote_volume"] = float(quote)
            except ValueError:
                pass
        if taker is not None:
            try:
                bar["taker_buy_volume"] = float(taker)
            except ValueError:
                pass
        if count is not None:
            try:
                bar["trades"] = int(float(count))
            except ValueError:
                pass
        parsed.append(bar)
    return parsed


def parse_metrics_rows(header, rows, interval_ms=300_000):
    """Archive metrics rows in the shape `Store.record_derivatives` accepts.

    The archive samples every 5 minutes (289 rows a day, not 288 -- the first lands at
    00:10 on some days, so rows are deduplicated by timestamp rather than counted).

    Note the two columns that are absent: `taker_buy_volume` and `taker_sell_volume` are
    in the stored schema but NOT in the archive. They come from `/futures/data/takerlongshortRatio`
    over REST, which is a 30-day endpoint, so they are unrecoverable for any earlier date
    and are left NULL here rather than filled with a zero -- zero would read as "no
    aggressive selling", which is a claim the archive does not support.
    """
    time_i = _column(header, ["create_time", "create time"], 0)
    oi_i = _column(header, ["sum_open_interest"], 2)
    oi_value_i = _column(header, ["sum_open_interest_value"], 3)
    acct_i = _column(header, ["count_toptrader_long_short_ratio"], 4)
    top_i = _column(header, ["sum_toptrader_long_short_ratio"], 5)
    global_i = _column(header, ["count_long_short_ratio"], 6)
    taker_ratio_i = _column(header, ["sum_taker_long_short_vol_ratio"], 7)
    parsed = []
    for row in rows:
        stamp = _value(row, time_i)
        if not stamp:
            continue
        event_time = _parse_time(stamp)
        if event_time is None:
            continue
        record = {"event_time": event_time}
        for key, index in (("open_interest", oi_i),
                           ("open_interest_value", oi_value_i),
                           ("top_account_ratio", acct_i),
                           ("long_short_ratio", top_i),
                           ("global_account_ratio", global_i),
                           ("taker_buy_sell_ratio", taker_ratio_i)):
            text = _value(row, index)
            if text is None:
                continue
            try:
                record[key] = float(text)
            except ValueError:
                continue
        parsed.append(record)
    return parsed


def _parse_time(text):
    """`2024-01-05 00:10:00` or an epoch in ms or us, to milliseconds UTC."""
    text = text.strip()
    if text.isdigit():
        value = int(text)
        # Archives have used ms and us for this column in different vintages.
        while value > 10 ** 14:
            value //= 1000
        return value
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            import datetime as _dt
            return int(_dt.datetime.strptime(text, pattern)
                       .replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return None
