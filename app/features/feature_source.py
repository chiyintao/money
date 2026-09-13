"""The single feature builder shared by training and by live serving.

Before this module there were two feature paths and they did not agree. The training job
computed the funding family -- funding_rate, funding_z, funding_carry_24h, mark_basis --
from the derivatives table and passed it into the snapshot; the live path called
snapshot(closed) with no funding argument at all, so those four keys fell back to
FUNDING_DEFAULTS and were constant zero on every bar the service ever evaluated.

Four of the ten modelled features were therefore dead at serving time while carrying real
variance in training. Every aggregate the service reported was produced by a model that
was being fed something other than what it was fit on, and nothing in the system could
notice: the out-of-range gate compares against the training bounds, and zero sits inside
them.

The fix is structural rather than a patch on the funding call. Train and serve now build
their features through the same object, so a feature that is present in one is present in
the other, and the object reports which modelled features it had to fill from a default.
That report is what the decision layer uses to refuse to trade on a degenerate input.
"""
from bisect import bisect_right
import time

from .feature_spec import FAMILY_FEATURES, FEATURES, FLOW_FEATURES, ORDER_FLOW_FEATURES, POSITIONING_FEATURES
from .features import FUNDING_DEFAULTS, snapshot

# Families that come from somewhere other than the candle and funding history. Each is
# declared with the store read that fills it, so a family that was never wired is a missing
# entry in this table rather than a function nobody calls.
EXTRA_DEFAULTS = {name: 0.0 for name in ORDER_FLOW_FEATURES + POSITIONING_FEATURES
                  + FLOW_FEATURES}

# How far back the derivative families look, expressed in bars rather than milliseconds so
# the same span means the same thing on a 5m dataset and on a 4h one.
POSITIONING_BARS = 48
FLOW_MINUTES = 60
# A store read is reused until the caller's bar walks out of it. Without this the dataset
# builder issues three queries per training row -- 150,000 rows times three -- where a
# block that covers four spans needs one.
BLOCK_FACTOR = 4
DEFAULT_STEP_MS = 300_000

# The features that come from the derivatives feed rather than from the candles. Kept as a
# named set so the guard can say precisely which input went missing instead of reporting a
# generic failure.
FUNDING_FEATURES = ("funding_rate", "funding_z", "funding_carry_24h", "mark_basis")

# How long a per-symbol funding series is reused before it is re-read. The series only
# changes when the venue settles or a new funding record is stored, so a minute is far
# more often than necessary and still keeps the live bar current.
CACHE_TTL_MS = 60_000


class FeatureSource:
    """Feature snapshots with an explicit account of what could not be filled.

    The store may be None, which is the honest configuration for a caller that has no
    derivatives history: the row is still returned (the model needs ten values, not nine)
    but it is reported as degraded so nothing downstream mistakes a placeholder for data.
    """

    def __init__(self, store=None, ttl_ms=CACHE_TTL_MS, max_basis_age_ms=None,
                 extras=True, features=None, reuse=False):
        self.store = store
        self.ttl_ms = int(ttl_ms)
        # Reuse the derivative series for the whole life of this object, ignoring the ttl.
        # Correct exactly when the store is not being written while the source is used,
        # which is the case for a dataset build replaying a closed history. The ttl is
        # there for the live loop, where a new settlement can land between two bars; a
        # builder that walks a year of bars re-read both series on every row to guard
        # against a write that cannot happen.
        self.reuse = bool(reuse)
        # Which features the caller is going to read. A model is served on the list its
        # artifact declares; the builder on whatever the contract currently is. Anything
        # absent from the row is reported as degraded rather than quietly filled.
        self.feature_names = tuple(features if features is not None else FEATURES)
        # Off for a caller that only wants price and funding. The extra families need
        # store reads that are pure waste when nothing consumes the answer.
        self.extras = bool(extras)
        # How stale a mark price may be before the basis is reported as unfillable rather
        # than computed from it. Settable so a caller with a different mark feed can say so.
        from ..market.funding import MAX_BASIS_AGE_MS

        self.max_basis_age_ms = int(MAX_BASIS_AGE_MS if max_basis_age_ms is None
                                    else max_basis_age_ms)
        self._series = {}
        self._blocks = {}
        self.counts = {'rows': 0, 'degraded': 0, 'funding_rows': 0,
                       'extra_rows': 0, 'extra_reads': 0}
        self.last_degraded = []
        self.last_missing = []

    # ------------------------------------------------------- outside the candles
    def _extra_reader(self, family):
        """The store read that fills one family, or None when it cannot be filled.

        The store may be absent or predate the table: both are the same answer here.
        """
        if self.store is None:
            return None
        if family == 'positioning':
            reader = getattr(self.store, 'derivatives_detail', None)
        elif family == 'flow':
            reader = getattr(self.store, 'flow', None)
        else:
            return None
        return reader if callable(reader) else None

    def _block(self, family, symbol, bar_time, since, until, limit):
        """Rows for one family, reused while bar_time stays inside the fetched block.

        How far forward the block may reach depends on whether the history is closed.
        Replaying a stored dataset, the rows after the anchor exist and were fetched with
        it, so extending forward turns a query per row into a query per few hundred. In the
        live loop they do not exist yet: a block anchored at 12:00 and cached for three
        hours contains nothing for 13:00, and the window for that bar comes back empty --
        a feature that degrades every time the cache would have helped. So a live source
        fetches per bar, which at one bar per five minutes per symbol is nothing, and an
        offline one extends.

        Every caller still filters to event_time <= bar_time, which is the same visibility
        rule the funding features apply and what keeps a forward walk through the history
        from reading its own future.
        """
        if not self.reuse:
            until = bar_time
        key = (family, symbol)
        cached = self._blocks.get(key)
        if cached and cached[0] <= bar_time <= cached[1]:
            return cached[2]
        reader = self._extra_reader(family)
        rows = []
        if reader is not None:
            try:
                # Both bounds. See Store.flow: with only a lower bound the statement
                # returns the newest rows overall, which for an early bar is the whole
                # future, and the window that filters to "not after this bar" is empty.
                rows = reader(symbol=symbol, limit=limit, since=since, until=until) or []
            except Exception:
                rows = []
            self.counts['extra_reads'] += 1
        self._blocks[key] = (since, until, rows)
        return rows

    def _extra_values(self, bars, symbol, bar_time):
        """The three collected families, or None per feature when its source is absent.

        None is the whole point. A family with no rows returns the same faces as a family
        whose value really is zero -- exactly the defect that made four funding features
        constant at serving time while carrying variance in training. The caller turns None
        into a default and records the name, so "balanced flow" and "no flow data" are
        distinguishable everywhere downstream.
        """
        from .order_flow import flow_features, order_flow_features, positioning_features

        # Only what the caller is going to read. A v3 artifact names ten columns, and a
        # source that reads two tables to fill twenty features nothing consumes is doing
        # the work and then reporting the result as a missing input.
        def wanted(names):
            return [name for name in names if name in self.feature_names]

        asked = {'order_flow': wanted(ORDER_FLOW_FEATURES),
                 'positioning': wanted(POSITIONING_FEATURES),
                 'flow': wanted(FLOW_FEATURES)}
        values = {}
        if not any(asked.values()):
            return values

        bars = bars or []
        if asked['order_flow']:
            # Computed from the bars already in hand: the ingest path has stored taker
            # volume, trade count and quote volume on every candle since the endpoint
            # existed, and the reader dropped them.
            measured = order_flow_features(bars) if bars else {}
            measured = measured if measured.get('order_flow_bars') else {}
            # The columns are absent on a database written before they were stored, so the
            # family is unmeasured rather than balanced.
            for name in asked['order_flow']:
                values[name] = measured.get(name)

        if self.store is None or not symbol or bar_time is None or not bars:
            for family in ('positioning', 'flow'):
                for name in asked[family]:
                    values[name] = None
            return values

        if asked['positioning']:
            span = POSITIONING_BARS * self._step_ms(bars)
            rows = self._block('positioning', symbol, int(bar_time), int(bar_time) - span,
                               int(bar_time) + span * (BLOCK_FACTOR - 1), POSITIONING_BARS * 4)
            visible = self._visible(rows, int(bar_time) - span, int(bar_time))
            measured = positioning_features(visible, price_change=self._price_change(bars))
            measured = measured if measured.get('positioning_rows') else {}
            for name in asked['positioning']:
                values[name] = measured.get(name)

        if asked['flow']:
            span = FLOW_MINUTES * 60_000
            rows = self._block('flow', symbol, int(bar_time), int(bar_time) - span,
                               int(bar_time) + span * (BLOCK_FACTOR - 1), FLOW_MINUTES * 4)
            visible = self._visible(rows, int(bar_time) - span, int(bar_time))
            measured = flow_features(visible)
            measured = measured if measured.get('flow_buckets') else {}
            for name in asked['flow']:
                values[name] = measured.get(name)
        return values

    @staticmethod
    def _visible(rows, since, until):
        """The rows inside one bounded window, oldest first.

        Bounded on both sides, and that is the point. Filtering only on "not after bar_time"
        makes the value depend on how much further back the cached block happened to reach:
        the same bar read at the start of a block saw 47 flow buckets and read one bar later
        saw 13, which is a feature whose value depends on cache alignment rather than on the
        market. Training and serving would then disagree for the same bar for a reason
        neither could see.
        """
        return [row for row in rows
                if since <= int(row.get('event_time') or 0) <= until]

    @staticmethod
    def _price_change(bars):
        """Last closed bar's return, the price leg of the open-interest quadrant."""
        if len(bars) < 2:
            return None
        previous = float(bars[-2].get('close') or 0)
        current = float(bars[-1].get('close') or 0)
        if previous <= 0 or current <= 0:
            return None
        return current / previous - 1.0

    @staticmethod
    def _step_ms(bars):
        """Bar spacing, from the bars themselves rather than from a configuration default."""
        for index in range(len(bars) - 1, 0, -1):
            step = (int(bars[index].get('close_time') or 0)
                    - int(bars[index - 1].get('close_time') or 0))
            if step > 0:
                return step
        return DEFAULT_STEP_MS

    # ------------------------------------------------------------------ funding
    def _funding_series(self, symbol, now_ms=None):
        """Both derivatives series for one symbol: funding, and mark prices.

        They are cached together because a caller always needs both, and because returning
        only the funding one is how the basis came to be computed off the funding cadence
        in the first place.
        """
        if self.store is None or not symbol:
            return None, ([], [])
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        cached = self._series.get(symbol)
        if cached and (self.reuse or now - cached[0] < self.ttl_ms):
            return cached[1], cached[2]
        try:
            from ..market.funding import funding_series, mark_series

            series = funding_series(self.store, symbol)
            # The mark cadence is not the funding cadence, so it is a second series on its
            # own clock rather than a third element of the same one.
            marks = mark_series(self.store, symbol)
        except Exception:
            series = None
            marks = ([], [])
        self._series[symbol] = (now, series, marks)
        return series, marks

    def invalidate(self, symbol=None):
        if symbol is None:
            self._series.clear()
            self._blocks.clear()
        else:
            self._series.pop(symbol, None)
            for key in [key for key in self._blocks if key[1] == symbol]:
                self._blocks.pop(key, None)

    # ----------------------------------------------------------------- snapshot
    def snapshot(self, rows, symbol=None, bar_time=None, price=None):
        """Feature row for one symbol, plus the modelled features that fell back.

        Returns (row, degraded) where degraded is a tuple of modelled feature names whose
        value is a default rather than data.
        """
        degraded = []
        funding = None
        series, mark_pair = self._funding_series(symbol)
        if series:
            times, rates, marks = series
            if times:
                if bar_time is None:
                    bar_time = int(rows[-1].get('close_time') or rows[-1].get('open_time') or 0)
                if price is None:
                    price = float(rows[-1].get('close') or 0) or None
                # Only publications at or before the bar are visible, which is the same
                # rule the training job applies -- the two must not disagree about the
                # information set, or backtest and service see different features.
                from ..market.funding import funding_features

                funding = funding_features(times, rates, marks, int(bar_time), price=price,
                                           mark_series=mark_pair if mark_pair[0] else None,
                                           max_basis_age_ms=self.max_basis_age_ms)
        if not funding:
            funding = dict(FUNDING_DEFAULTS)
            degraded = [name for name in FUNDING_FEATURES if name in FEATURES]
        else:
            # A feature the source could not fill comes back as None rather than as its
            # default, so the fallback is reported instead of being indistinguishable from
            # a real reading of zero.
            missing = [name for name, value in funding.items()
                       if value is None and name in FEATURES]
            for name in missing:
                funding[name] = FUNDING_DEFAULTS.get(name, 0.0)
            degraded = missing
        row = snapshot(rows, funding)
        if self.extras:
            values = self._extra_values(rows, symbol, bar_time)
            for name, value in values.items():
                if value is None:
                    row[name] = EXTRA_DEFAULTS.get(name, 0.0)
                    degraded.append(name)
                else:
                    row[name] = value
        # Anything the caller is going to read and did not get, including a feature from a
        # family that was switched off. Filling it silently is what produced a model whose
        # training rows and serving rows disagreed.
        missing = [name for name in self.feature_names
                   if name not in row or row[name] is None]
        for name in missing:
            row[name] = EXTRA_DEFAULTS.get(name, FUNDING_DEFAULTS.get(name, 0.0))
        degraded.extend(name for name in missing if name not in degraded)
        self.counts['rows'] += 1
        if self.extras:
            self.counts['extra_rows'] += 0 if degraded else 1
        if degraded:
            self.counts['degraded'] += 1
            self.last_degraded = list(degraded)
        else:
            self.counts['funding_rows'] += 1
        self.last_missing = list(missing)
        return row, tuple(degraded)

    def health(self):
        """What the feature layer actually managed to fill.

        reported_features is the list the caller reads; extra_rows counts rows where every
        one of the collected families was present. A service whose flow tables are empty
        shows extra_rows at zero here, which is the fact that the model panel had no way to
        state before.
        """
        return {'rows': self.counts['rows'], 'degraded_rows': self.counts['degraded'],
                'funding_rows': self.counts['funding_rows'],
                'extra_rows': self.counts['extra_rows'],
                'extra_reads': self.counts['extra_reads'],
                'degraded_features': list(self.last_degraded),
                'missing_features': list(self.last_missing),
                'reported_features': len(self.feature_names),
                'extras_enabled': self.extras,
                'families': {name: len(names) for name, names in FAMILY_FEATURES.items()},
                'store': self.store is not None, 'cached_symbols': len(self._series),
                'cached_blocks': len(self._blocks)}


def funding_age_ms(store, symbol, bar_time):
    """Age of the newest funding publication visible at bar_time, in milliseconds.

    Funding settles every eight hours on most contracts, so a series that is merely old is
    not necessarily stale. This reports the fact and lets the caller decide the bound;
    None means no publication was visible at all.
    """
    if store is None or not symbol:
        return None
    try:
        from ..market.funding import funding_series

        times, _rates, _marks = funding_series(store, symbol)
    except Exception:
        return None
    if not times:
        return None
    index = bisect_right(times, int(bar_time)) - 1
    if index < 0:
        return None
    return max(0, int(bar_time) - int(times[index]))
