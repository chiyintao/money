"""The horizon analysis that decides how long a trade should be held.

The audit found the ensemble's edge peaking in the 5-15 minute band and turning negative
by 30 minutes, while the exit geometry aimed at targets needing hours to reach. A
reward:risk ratio is meaningless if the edge it waits for has decayed by the time the
target is touched, so the holding period has to come from measurement.
"""
import json

import pytest

from app.strategy.edge_report import render, _parse_symbols
from app.strategy.symbol_edge import horizon_analysis

EPOCH = 1_700_000_000_000
BAR = 300_000


class FakeStore:
    """Minimal store: decision events plus one candle series per symbol."""

    def __init__(self, events, candles):
        self._events = events
        self._candles = candles
        self.db = self

    def execute(self, sql, params=()):
        if 'strategy_decision' in sql:
            return list(self._events)[: params[-1] if params else None]
        return []

    def candles(self, symbol, interval, limit=250, closed_only=True):
        return self._candles.get(symbol, [])


def make_events(symbol, side, bar_times, price=100.0):
    return [(int(bar_time),
             json.dumps({'symbol': symbol, 'decision': {'side': side},
                         'market': {'price': price}}))
            for bar_time in bar_times]


def make_candles(count, start=100.0, step=1.0):
    return [{'open_time': EPOCH + index * BAR, 'close': start + index * step}
            for index in range(count)]


def test_a_rising_symbol_shows_positive_edge_at_every_horizon():
    bars = [EPOCH + index * BAR for index in range(20)]
    store = FakeStore(make_events('UPUSDT', 'LONG', bars), {'UPUSDT': make_candles(40, step=1.0)})
    analysis = horizon_analysis(store, '5m', horizons=(1, 3, 6), cost=0.0, min_samples=5)
    for horizon in (1, 3, 6):
        assert analysis['symbols']['UPUSDT'][horizon]['mean_net_bps'] > 0


def test_costs_are_deducted_from_the_reported_edge():
    bars = [EPOCH + index * BAR for index in range(20)]
    candles = {'UPUSDT': make_candles(40, step=0.0)}      # perfectly flat
    free = horizon_analysis(FakeStore(make_events('UPUSDT', 'LONG', bars), candles),
                            '5m', horizons=(1,), cost=0.0, min_samples=5)
    charged = horizon_analysis(FakeStore(make_events('UPUSDT', 'LONG', bars), candles),
                               '5m', horizons=(1,), cost=0.0012, min_samples=5)
    assert free['symbols']['UPUSDT'][1]['mean_net_bps'] == pytest.approx(0.0)
    assert charged['symbols']['UPUSDT'][1]['mean_net_bps'] == pytest.approx(-12.0)


def test_a_short_is_scored_with_the_opposite_sign():
    bars = [EPOCH + index * BAR for index in range(20)]
    store = FakeStore(make_events('DOWNUSDT', 'SHORT', bars), {'DOWNUSDT': make_candles(40, step=-1.0)})
    analysis = horizon_analysis(store, '5m', horizons=(3,), cost=0.0, min_samples=5)
    assert analysis['symbols']['DOWNUSDT'][3]['mean_net_bps'] > 0


def test_repeated_decisions_in_one_bar_count_once():
    # The tick path repeats one view many times per bar. Without deduplication a single
    # opinion would fill the sample and read as overwhelming evidence.
    bars = [EPOCH + index * BAR for index in range(20)]
    many = []
    for bar_time in bars:
        for offset in range(10):
            many.append((int(bar_time) + offset,
                         json.dumps({'symbol': 'AUSDT', 'decision': {'side': 'LONG'},
                                     'market': {'price': 100.0}})))
    store = FakeStore(many, {'AUSDT': make_candles(40, step=1.0)})
    analysis = horizon_analysis(store, '5m', horizons=(1,), cost=0.0, min_samples=1)
    assert analysis['symbols']['AUSDT'][1]['n'] == len(bars)


def test_symbols_below_the_sample_floor_are_omitted():
    bars = [EPOCH + index * BAR for index in range(5)]
    store = FakeStore(make_events('THINUSDT', 'LONG', bars), {'THINUSDT': make_candles(40, step=1.0)})
    analysis = horizon_analysis(store, '5m', horizons=(1,), cost=0.0, min_samples=50)
    assert 'THINUSDT' not in analysis['symbols']


def test_the_best_horizon_is_the_one_with_the_highest_net_edge():
    bars = [EPOCH + index * BAR for index in range(20)]
    # Flat for one bar then rising, so the longer horizon is the profitable one.
    candles = [{'open_time': EPOCH + index * BAR, 'close': 100.0 if index < 20 else 120.0}
               for index in range(40)]
    store = FakeStore(make_events('AUSDT', 'LONG', bars), {'AUSDT': candles})
    analysis = horizon_analysis(store, '5m', horizons=(1, 12), cost=0.0, min_samples=2)
    assert analysis['best_horizon'] == 12


def test_no_positive_horizon_reports_none():
    # This is the signal to leave the time stop disabled rather than to invent one.
    bars = [EPOCH + index * BAR for index in range(20)]
    store = FakeStore(make_events('AUSDT', 'LONG', bars), {'AUSDT': make_candles(40, step=-1.0)})
    analysis = horizon_analysis(store, '5m', horizons=(1, 3), cost=0.0, min_samples=5)
    assert analysis['best_horizon'] is None


def test_a_symbol_without_candles_is_skipped_rather_than_crashing():
    bars = [EPOCH + index * BAR for index in range(20)]
    store = FakeStore(make_events('NOCANDLES', 'LONG', bars), {})
    analysis = horizon_analysis(store, '5m', horizons=(1,), cost=0.0, min_samples=1)
    assert analysis['symbols'] == {}


def test_the_report_renders_both_passes():
    bars = [EPOCH + index * BAR for index in range(20)]
    candles = {'UPUSDT': make_candles(40, step=1.0), 'DOWNUSDT': make_candles(40, step=-1.0)}
    events = make_events('UPUSDT', 'LONG', bars) + make_events('DOWNUSDT', 'LONG', bars)
    analysis = horizon_analysis(FakeStore(events, candles), '5m', horizons=(1, 3), cost=0.0,
                                min_samples=5)
    text = render(analysis)
    assert 'PASS 1' in text and 'PASS 2' in text
    assert 'UPUSDT' in text and 'DOWNUSDT' in text
    assert 'best horizon' in text
    # The losing symbol must be labelled as such; the table exists to show who pays.
    assert 'reject' in text


def test_the_report_handles_having_no_symbol_data():
    analysis = {'interval': '5m', 'cost': 0.0012, 'horizons': [1], 'pooled': {},
                'symbols': {}, 'best_horizon': None}
    text = render(analysis)
    assert 'no horizon is positive' in text
    assert 'no symbol has enough resolved observations' in text


def test_symbol_filter_parsing_ignores_empty_tokens():
    assert _parse_symbols('a, b ,,c') == {'A', 'B', 'C'}
    assert _parse_symbols('') is None
    assert _parse_symbols(None) is None
