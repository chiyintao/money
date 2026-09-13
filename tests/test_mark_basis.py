"""The basis needs a mark price observed at the bar, not at the last settlement.

Measured on the shipped training set before this was fixed: mark_basis sat on its +/-2%
clip for 122,201 of 1,258,344 rows -- 9.711% -- while every other feature saturated at
0.046% or less. The cause was that the mark price was read from the funding row, and
funding settles every eight hours, so "basis" was the price change since the previous
settlement. A tenth of the training set fed the model a constant.
"""
from app.market.funding import MAX_BASIS_AGE_MS, funding_features, mark_basis, mark_series
from app.storage.storage import Store

HOUR = 3_600_000


def test_a_fresh_mark_produces_a_real_basis():
    # A mark 20 bps above the close is a basis of 20 bps, not a rounding artefact.
    times, rates, marks = [1000], [0.0001], [100.0]
    values = funding_features(times, rates, marks, 1000, price=100.0,
                              mark_series=([1000], [100.2]))
    assert abs(values["mark_basis"] - 0.002) < 1e-12


def test_a_settlement_old_mark_is_refused():
    # Eight hours is the funding cadence: the mark on that row describes a different
    # market, and comparing it to the current close measures the price move, not a basis.
    times, rates, marks = [0], [0.0001], [80.0]
    values = funding_features(times, rates, marks, 8 * HOUR, price=100.0,
                              mark_series=([0], [80.0]))
    assert values["mark_basis"] is None


def test_the_bound_is_where_it_says_it_is():
    fresh = funding_features([0], [0.0001], [100.0], MAX_BASIS_AGE_MS, price=100.0,
                             mark_series=([0], [100.0]))
    assert fresh["mark_basis"] == 0.0
    stale = funding_features([0], [0.0001], [100.0], MAX_BASIS_AGE_MS + 1, price=100.0,
                             mark_series=([0], [100.0]))
    assert stale["mark_basis"] is None


def test_the_funding_cadence_does_not_constrain_the_mark():
    # The whole point of the split: the funding lookup still bisects the eight-hourly
    # series, while the basis bisects the five-minute one.
    funding_times = [0, 8 * HOUR]
    rates = [0.0001, 0.0002]
    marks = [100.0, 100.0]
    mark_times = [0, 300_000, 600_000, 900_000]
    mark_values = [100.0, 101.0, 102.0, 103.0]
    values = funding_features(funding_times, rates, marks, 900_000, price=103.0,
                              mark_series=(mark_times, mark_values))
    assert values["funding_rate"] == 0.0001        # still the first settlement
    assert values["mark_basis"] == 0.0             # the 900s mark, not the t=0 one


def test_a_mark_axis_with_the_wrong_values_is_not_silently_accepted():
    # Pairing a mark time axis with the funding values looks right and is not; the values
    # have to travel with their own axis.
    values = funding_features([0], [0.0001], [50.0], 0, price=100.0,
                              mark_series=([0], [50.0]))
    assert values["mark_basis"] == -0.5


def test_mark_basis_is_none_for_every_unusable_input():
    assert mark_basis([], [], 1000, 100.0) is None
    assert mark_basis([1000], [100.0], 500, 100.0) is None      # nothing visible yet
    assert mark_basis([1000], [100.0], 1000, None) is None      # no price to compare
    assert mark_basis([1000], [100.0], 1000, 0.0) is None
    assert mark_basis([1000], [0.0], 1000, 100.0) is None       # no mark to compare
    assert mark_basis([1000], [], 1000, 100.0) is None


def test_mark_series_reads_the_five_minute_detail_rows(tmp_path):
    store = Store(tmp_path)
    try:
        store.record_derivatives([{"symbol": "X", "event_time": 0, "funding_rate": 0.0001,
                                   "mark_price": 90.0, "open_interest": 0.0}])
        store.record_derivatives_detail([
            {"symbol": "X", "event_time": 300, "open_interest": 10.0,
             "open_interest_value": 1010.0},
            {"symbol": "X", "event_time": 600, "open_interest": 10.0,
             "open_interest_value": 1020.0}])
        times, marks = mark_series(store, "X")
        assert times == [300, 600]
        assert marks == [101.0, 102.0]
    finally:
        store.close()


def test_mark_series_falls_back_to_the_funding_rows(tmp_path):
    # An older database has only the eight-hourly rows. It must behave exactly as it did
    # before rather than losing the feature outright; the age bound is what refuses it.
    store = Store(tmp_path)
    try:
        store.record_derivatives([{"symbol": "X", "event_time": 0, "funding_rate": 0.0001,
                                   "mark_price": 90.0, "open_interest": 0.0}])
        times, marks = mark_series(store, "X")
        assert times == [0] and marks == [90.0]
    finally:
        store.close()


def test_mark_series_ignores_rows_with_no_usable_price(tmp_path):
    store = Store(tmp_path)
    try:
        store.record_derivatives_detail([
            {"symbol": "X", "event_time": 300, "open_interest": 0.0,
             "open_interest_value": 0.0},
            {"symbol": "X", "event_time": 600, "open_interest": 10.0,
             "open_interest_value": 500.0}])
        times, marks = mark_series(store, "X")
        assert times == [600] and marks == [50.0]
    finally:
        store.close()
