"""Triple-barrier labels and the weights that stop one event counting a dozen times.

The old label was the return from the next bar's open to the close twelve bars later. That
describes a strategy nobody runs: the live system has a stop, a target, a trailing stop and
a time stop, and it leaves when one of them is touched. The audit found the label
distribution and the realised trade distribution to be different distributions, and this is
the fix -- the label is now the outcome the exit policy would actually have produced.

The second half is the overlapping-sample problem. With a twelve-bar horizon, adjacent rows
share eleven twelfths of their outcome window, so they are nearly the same observation
counted twice. The audit measured label lag-1 autocorrelation at 0.557: an honest t of
about 5.15 looks like 14.0 when the overlap is ignored.
"""
import pytest

from app.models.labels import (LOWER, UPPER, VERTICAL, sample_weights, triple_barrier,
                               uniqueness_weights)


def bar(open_, high, low, close, index):
    base = index * 300_000
    return {'open_time': base, 'close_time': base + 299_999, 'open': open_
            , 'high': high, 'low': low, 'close': close, 'volume': 1.0}


def series(closes, highs=None, lows=None):
    highs = highs or [c * 1.001 for c in closes]
    lows = lows or [c * 0.999 for c in closes]
    return [bar(closes[i], highs[i], lows[i], closes[i], i) for i in range(len(closes))]


# ------------------------------------------------------------ the barriers
def test_the_upper_barrier_resolves_the_label():
    """A touched target ends the trade there, not at the end of the horizon."""
    bars = series([100, 100, 100, 100, 100, 100],
                  highs=[100, 101, 105, 106, 106, 106],
                  lows=[100, 99, 99, 99, 99, 99])
    outcome = triple_barrier(bars, 0, 2.0, 4.0, 5)
    assert outcome['barrier'] == UPPER
    assert outcome['future_return'] == pytest.approx(0.04)
    assert outcome['bars_held'] == 2


def test_the_lower_barrier_resolves_the_label():
    """A touched stop ends it too, with the loss actually taken."""
    bars = series([100, 100, 100, 100, 100, 100],
                  highs=[100, 100, 100, 100, 100, 100],
                  lows=[100, 97, 97, 97, 97, 97])
    outcome = triple_barrier(bars, 0, 2.0, 4.0, 5)
    assert outcome['barrier'] == LOWER
    assert outcome['future_return'] == pytest.approx(-0.02)
    assert outcome['bars_held'] == 1


def test_no_touch_falls_through_to_the_vertical_barrier():
    """The time stop, which is what the old fixed horizon was approximating."""
    bars = series([100, 100, 100, 100, 100, 101],
                  highs=[100, 100.5, 100.5, 100.5, 100.5, 101.2],
                  lows=[100, 99.5, 99.5, 99.5, 99.5, 100.8])
    outcome = triple_barrier(bars, 0, 2.0, 4.0, 5)
    assert outcome['barrier'] == VERTICAL
    assert outcome['bars_held'] == 5
    assert outcome['future_return'] == pytest.approx(0.01)


def test_the_label_is_measured_from_the_entry_price_given():
    """The entry is the next bar's open, not the close that produced the signal."""
    bars = series([100, 110, 110, 110, 110, 110],
                  highs=[100, 110, 110, 110, 110, 110],
                  lows=[100, 110, 110, 110, 110, 110])
    outcome = triple_barrier(bars, 0, 2.0, 4.0, 5, entry_price=110.0)
    assert outcome['future_return'] == pytest.approx(0.0)


def test_a_tighter_stop_is_hit_sooner():
    """The barrier distances are what the exit policy decides, so they have to matter."""
    bars = series([100] * 6, highs=[100] * 6, lows=[100, 99, 99, 99, 99, 99])
    wide = triple_barrier(bars, 0, 5.0, 4.0, 5)
    tight = triple_barrier(bars, 0, 0.5, 4.0, 5)
    assert wide['barrier'] == VERTICAL
    assert tight['barrier'] == LOWER


def test_invalid_inputs_are_refused():
    """A zero-distance barrier would resolve every row on the first bar."""
    bars = series([100] * 5)
    with pytest.raises(ValueError):
        triple_barrier(bars, 0, 0.0, 4.0, 3)
    with pytest.raises(ValueError):
        triple_barrier(bars, 0, 2.0, 4.0, 0)


def test_the_walk_never_runs_past_the_available_bars():
    """The last rows of a series have a truncated horizon, not an exception."""
    bars = series([100, 100, 100])
    outcome = triple_barrier(bars, 1, 2.0, 4.0, 50)
    assert outcome['barrier'] == VERTICAL
    assert outcome['bars_held'] <= 1


# ------------------------------------------------------------ uniqueness
def test_overlapping_labels_get_smaller_weights_than_disjoint_ones():
    """The whole point: an observation shared with a neighbour is worth less."""
    overlapping = uniqueness_weights([(0, 3), (1, 4), (2, 5), (3, 6)], 10)
    disjoint = uniqueness_weights([(0, 1), (3, 4), (6, 7), (9, 10)], 10)
    assert all(weight < 1.0 for weight in overlapping)
    assert disjoint == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert sum(overlapping) < sum(disjoint)


def test_weights_are_symmetric_when_the_overlap_is():
    """No positional artefact: the same shape at either end weights the same."""
    weights = uniqueness_weights([(0, 3), (1, 4), (2, 5), (3, 6)], 10)
    assert weights[0] == pytest.approx(weights[-1])
    assert weights[1] == pytest.approx(weights[-2])


def test_a_single_label_covering_everything_weighs_one():
    """Nothing overlaps it, so it is entirely its own observation."""
    assert uniqueness_weights([(0, 9)], 10) == pytest.approx([1.0])


def test_sample_weights_are_normalised_to_a_mean_of_one():
    """A weighted fit must not also be trained at a different scale."""
    rows = [{'timestamp': i, 'bars_held': 3} for i in range(20)]
    weights = sample_weights(rows)
    assert len(weights) == len(rows)
    assert sum(weights) / len(weights) == pytest.approx(1.0)


def test_sample_weights_read_the_recorded_label_span():
    """label_start/label_end are written by the builder and preferred over a guess."""
    rows = [{'timestamp': 0, 'label_start': 0, 'label_end': 2},
            {'timestamp': 1, 'label_start': 1, 'label_end': 3}]
    weights = sample_weights(rows)
    assert all(0 < weight <= 2 for weight in weights)


def test_sample_weights_of_nothing_is_nothing():
    """The caller passes None through, rather than an empty array to the library."""
    assert sample_weights([]) == []

def test_weights_are_linear_in_rows_not_in_milliseconds():
    """Intervals are bar ordinals, never raw timestamps.

    The first version passed the recorded millisecond spans straight to the concurrency
    sweep, so it allocated one slot per millisecond: 587 real rows spanning 176 million ms
    took 134 seconds and most of a gigabyte, and a real 1.2M-row dataset would never have
    finished. A regression here is invisible in the weights themselves -- they come out
    correct, just impossibly slowly -- so it needs its own test.
    """
    import time

    rows = [{'timestamp': 1_700_000_000_000 + index * 300_000, 'bars_held': 12}
            for index in range(2_000)]
    started = time.time()
    weights = sample_weights(rows)
    elapsed = time.time() - started
    assert len(weights) == len(rows)
    assert elapsed < 5.0, "%d rows took %.1fs; intervals are probably in ms again" % (
        len(rows), elapsed)
    assert all(weight > 0 for weight in weights)


def test_a_realistic_timestamp_spread_stays_cheap():
    """The shape that broke it: dense 5-minute stamps over months."""
    import time

    rows = [{'timestamp': 1_600_000_000_000 + index * 300_000, 'bars_held': 12}
            for index in range(5_000)]
    started = time.time()
    sample_weights(rows)
    assert time.time() - started < 10.0

