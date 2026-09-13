"""Funding history: parsing, coverage and the no-look-ahead guarantee."""
import pytest

from app.market.funding import RECORDS_PER_DAY, funding_features, funding_series, parse_funding
from app.storage.storage import Store


def test_parse_funding_reads_the_endpoint_shape():
    rows = parse_funding([{"symbol": "BTCUSDT", "fundingTime": 1700000000000,
                          "fundingRate": "0.00010000", "markPrice": "78000.5"}])
    assert rows == [{"symbol": "BTCUSDT", "event_time": 1700000000000,
                     "funding_rate": 0.0001, "mark_price": 78000.5}]


def test_parse_funding_skips_malformed_records():
    assert parse_funding([{"nope": 1}, None, {"symbol": "X", "fundingTime": "bad"}]) == []
    assert parse_funding(None) == []


def test_funding_features_never_use_a_later_publication():
    # The settlement at 3000 is a large jump. A bar before it must not see it, which is
    # exactly the look-ahead that would make a backtest look profitable.
    times = [1000, 2000, 3000]
    rates = [0.0001, 0.0002, 0.0009]
    marks = [100.0, 101.0, 130.0]
    assert funding_features(times, rates, marks, 2000, price=101.0)["funding_rate"] == 0.0002
    assert funding_features(times, rates, marks, 2500, price=105.0)["funding_rate"] == 0.0002
    assert funding_features(times, rates, marks, 2999, price=105.0)["funding_rate"] == 0.0002


def test_a_bar_exactly_on_a_settlement_sees_it():
    times, rates, marks = [1000, 2000], [0.0001, 0.0005], [100.0, 102.0]
    assert funding_features(times, rates, marks, 2000, price=102.0)["funding_rate"] == 0.0005


def test_before_the_first_settlement_nothing_is_known():
    values = funding_features([1000], [0.001], [100.0], 500, price=99.0)
    # All four are None rather than 0.0: no publication is visible, so none of them is
    # known, and a filled-in zero sits inside the training bounds where nothing downstream
    # can tell it from a real reading of neutral funding.
    assert values == {"funding_rate": None, "funding_z": None, "funding_carry_24h": None,
                      "mark_basis": None}


def test_carry_sums_three_settlements():
    times = [1000, 2000, 3000, 4000]
    rates = [0.0001, 0.0002, 0.0003, 0.0004]
    marks = [100.0] * 4
    values = funding_features(times, rates, marks, 4000, price=100.0)
    assert values["funding_carry_24h"] == pytest.approx(0.0002 + 0.0003 + 0.0004)
    assert RECORDS_PER_DAY == 3


def test_basis_compares_the_mark_to_the_bar_close():
    values = funding_features([1000], [0.0], [102.0], 1000, price=100.0)
    assert values["mark_basis"] == pytest.approx(0.02)


def test_z_score_is_zero_when_funding_never_moves():
    times = list(range(0, 40000, 1000))
    rates = [0.0001] * len(times)
    marks = [100.0] * len(times)
    values = funding_features(times, rates, marks, times[-1], price=100.0)
    assert values["funding_z"] == 0.0


def test_z_score_reacts_to_a_deviation():
    times = list(range(0, 40000, 1000))
    rates = [0.0001] * (len(times) - 1) + [0.001]
    marks = [100.0] * len(times)
    values = funding_features(times, rates, marks, times[-1], price=100.0)
    assert values["funding_z"] > 1.0


def test_missing_price_leaves_the_basis_unfilled():
    values = funding_features([1000], [0.0], [102.0], 1000, price=None)
    assert values["mark_basis"] is None


def test_funding_series_round_trips_through_the_store(tmp_path):
    store = Store(str(tmp_path))
    store.record_derivatives([{"symbol": "BTCUSDT", "event_time": 2000,
                              "funding_rate": 0.0002, "mark_price": 101.0},
                             {"symbol": "BTCUSDT", "event_time": 1000,
                              "funding_rate": 0.0001, "mark_price": 100.0}])
    times, rates, marks = funding_series(store, "BTCUSDT")
    assert times == [1000, 2000]
    assert rates == [0.0001, 0.0002]
    assert marks == [100.0, 101.0]
    store.close()


def test_a_zero_funding_rate_is_not_treated_as_missing(tmp_path):
    # Neutral funding settles at exactly zero; filtering those out understated coverage.
    store = Store(str(tmp_path))
    store.record_derivatives([{"symbol": "X", "event_time": 1000, "funding_rate": 0.0,
                              "mark_price": 100.0},
                             {"symbol": "X", "event_time": 2000, "funding_rate": 0.0001,
                              "mark_price": 100.0}])
    from app.market.funding import coverage
    assert coverage(store, ["X"])[0]["records"] == 2
    store.close()
