"""The meta-labeling gate: per-symbol evidence decides what is worth trading.

The audit behind this found the ensemble predicting at roughly coin-flip accuracy, but
not uniformly: one symbol scored 64% at 15 minutes while another scored 33%. All of them
were traded with identical rules, so the losers were funded by the winners.
"""
import pytest

from app.strategy.symbol_edge import SymbolEdge, signed_return


# A realistic epoch, because a zero timestamp is treated as "unset" and would be
# rejected rather than scored.
EPOCH = 1_700_000_000_000


def series(count, start=100.0, step=0.0, start_time=EPOCH):
    times = [start_time + index * 300000 for index in range(count)]
    closes = [start + index * step for index in range(count)]
    return times, closes


def test_signed_return_flips_for_a_short():
    assert signed_return('LONG', 100.0, 110.0) == pytest.approx(0.1)
    assert signed_return('SHORT', 100.0, 110.0) == pytest.approx(-0.1)
    assert signed_return('SHORT', 100.0, 90.0) == pytest.approx(0.1)


def test_signed_return_rejects_unusable_prices():
    assert signed_return('LONG', 0.0, 110.0) is None
    assert signed_return('LONG', 100.0, 0.0) is None


def test_a_forecast_is_not_scored_before_its_horizon_elapses():
    # Scoring early would grade a forecast against a move that has not happened yet.
    tracker = SymbolEdge(horizon_bars=3, min_samples=1)
    times, closes = series(10, step=1.0)
    tracker.observe('AUSDT', 'LONG', times[0], closes[0])
    assert tracker.resolve('AUSDT', times[:3], closes[:3]) == 0
    assert tracker.stats('AUSDT')['n'] == 0
    # Once the horizon exists the same forecast scores.
    assert tracker.resolve('AUSDT', times, closes) == 1
    assert tracker.stats('AUSDT')['n'] == 1


def test_a_correct_long_scores_positive_and_a_wrong_one_negative():
    tracker = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0)
    times, closes = series(5, start=100.0, step=1.0)          # rising
    tracker.observe('UPUSDT', 'LONG', times[0], closes[0])
    tracker.resolve('UPUSDT', times, closes)
    assert tracker.stats('UPUSDT')['mean_net_bps'] > 0

    falling = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0)
    times, closes = series(5, start=100.0, step=-1.0)         # falling
    falling.observe('DOWNUSDT', 'LONG', times[0], closes[0])
    falling.resolve('DOWNUSDT', times, closes)
    assert falling.stats('DOWNUSDT')['mean_net_bps'] < 0


def test_the_round_trip_cost_is_subtracted_from_every_outcome():
    # A symbol whose gross edge is smaller than the cost of trading it is a losing symbol
    # no matter how good the hit rate looks.
    times, closes = series(5, start=100.0, step=0.01)          # +1bp per bar
    free = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0)
    free.observe('AUSDT', 'LONG', times[0], closes[0])
    free.resolve('AUSDT', times, closes)
    charged = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0012)
    charged.observe('AUSDT', 'LONG', times[0], closes[0])
    charged.resolve('AUSDT', times, closes)
    assert free.stats('AUSDT')['mean_net_bps'] == pytest.approx(1.0, abs=0.01)
    assert charged.stats('AUSDT')['mean_net_bps'] == pytest.approx(-11.0, abs=0.01)


def test_a_symbol_below_the_minimum_sample_count_is_not_traded():
    # Deny until proven. Safe because observations are recorded whether or not the symbol
    # is tradable, so a blocked symbol still accumulates the evidence to come back.
    tracker = SymbolEdge(horizon_bars=1, min_samples=5, round_trip_cost=0.0)
    times, closes = series(10, start=100.0, step=1.0)
    for index in range(3):
        tracker.observe('AUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('AUSDT', times, closes)
    allowed, reason = tracker.allows('AUSDT')
    assert allowed is False
    assert reason == 'insufficient_edge_samples'


def test_a_symbol_with_a_negative_record_is_rejected():
    tracker = SymbolEdge(horizon_bars=1, min_samples=3, round_trip_cost=0.0)
    times, closes = series(10, start=100.0, step=-1.0)         # every long loses
    for index in range(5):
        tracker.observe('BADUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('BADUSDT', times, closes)
    allowed, reason = tracker.allows('BADUSDT')
    assert allowed is False
    assert reason == 'symbol_edge_below_floor'


def test_a_symbol_with_a_positive_record_is_allowed():
    tracker = SymbolEdge(horizon_bars=1, min_samples=3, round_trip_cost=0.0)
    times, closes = series(10, start=100.0, step=1.0)
    for index in range(5):
        tracker.observe('GOODUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('GOODUSDT', times, closes)
    allowed, reason = tracker.allows('GOODUSDT')
    assert allowed is True
    assert reason == 'symbol_edge_ok'


def test_observations_are_recorded_even_while_a_symbol_is_blocked():
    # This is what makes deny-until-proven safe: rejection must not blind the tracker, or
    # a blocked symbol could never earn its way back.
    tracker = SymbolEdge(horizon_bars=1, min_samples=2, round_trip_cost=0.0)
    times, closes = series(10, start=100.0, step=1.0)
    assert tracker.allows('NEWUSDT')[0] is False
    for index in range(4):
        tracker.observe('NEWUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('NEWUSDT', times, closes)
    assert tracker.stats('NEWUSDT')['n'] == 4
    assert tracker.allows('NEWUSDT')[0] is True


def test_flat_proposals_are_never_recorded():
    tracker = SymbolEdge(horizon_bars=1, min_samples=1)
    times, closes = series(5)
    tracker.observe('AUSDT', 'FLAT', times[0], closes[0])
    assert tracker.pending == {}


def test_the_window_is_bounded_so_stale_evidence_ages_out():
    tracker = SymbolEdge(horizon_bars=1, window=10, min_samples=1, round_trip_cost=0.0)
    times, closes = series(60, start=100.0, step=1.0)
    for index in range(50):
        tracker.observe('AUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('AUSDT', times, closes)
    assert tracker.stats('AUSDT')['n'] == 10


def test_a_forecast_older_than_the_series_is_dropped_not_mis_scored():
    # Anchoring an old forecast to the wrong bar would manufacture a result.
    tracker = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0)
    times, closes = series(5, start=100.0, step=1.0)
    tracker.observe('AUSDT', 'LONG', times[0] - 10_000_000, closes[0])
    assert tracker.resolve('AUSDT', times, closes) == 0
    assert tracker.stats('AUSDT')['n'] == 0


def test_the_gate_can_be_disabled_entirely():
    tracker = SymbolEdge(horizon_bars=1, min_samples=100, enabled=False)
    assert tracker.allows('ANYUSDT') == (True, 'edge_gate_disabled')


def test_snapshot_and_restore_round_trip_keeps_the_evidence():
    tracker = SymbolEdge(horizon_bars=1, window=10, min_samples=1, round_trip_cost=0.0006)
    times, closes = series(10, start=100.0, step=1.0)
    for index in range(5):
        tracker.observe('AUSDT', 'LONG', times[index], closes[index])
    tracker.resolve('AUSDT', times, closes)
    before = tracker.stats('AUSDT')

    restored = SymbolEdge(horizon_bars=1, window=10, min_samples=1).restore(tracker.snapshot())
    assert restored.round_trip_cost == pytest.approx(0.0006)
    assert restored.stats('AUSDT')['mean_net_bps'] == pytest.approx(before['mean_net_bps'])


def test_the_table_is_ordered_worst_first():
    tracker = SymbolEdge(horizon_bars=1, min_samples=1, round_trip_cost=0.0)
    times, closes = series(10, start=100.0, step=1.0)
    tracker.observe('WINUSDT', 'LONG', times[0], closes[0])
    tracker.resolve('WINUSDT', times, closes)
    down_times, down_closes = series(10, start=100.0, step=-1.0)
    tracker.observe('LOSEUSDT', 'LONG', down_times[0], down_closes[0])
    tracker.resolve('LOSEUSDT', down_times, down_closes)
    names = [row['symbol'] for row in tracker.table()]
    assert names == ['LOSEUSDT', 'WINUSDT']
