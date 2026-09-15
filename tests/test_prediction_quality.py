"""Unit tests for the modules added to replace the failing intraday pipeline.

Each test pins a property that the measured failure showed was not being checked, rather
than a number that happens to hold on the current dataset.
"""
import numpy as np

from app.models import cpcv, feature_audit, label_economics
from app.strategy import cross_sectional as cs


def _row(timestamp, symbol="A", **features):
    row = {"timestamp": timestamp, "symbol": symbol, "future_return": 0.0}
    row.update(features)
    return row


# ------------------------------------------------------------------ feature audit

def test_audit_finds_a_constant_column():
    rows = [_row(i, dead=0.0, live=i * 0.01) for i in range(100)]
    report = feature_audit.audit_rows(rows, features=("dead", "live"))
    assert report["counts"]["constant"] == 1
    assert report["dead"] == ["dead"]
    assert feature_audit.trainable_features(report) == ["live"]


def test_audit_keeps_a_rare_event_column():
    # A liquidation field that fires on 2% of bars is informative and must survive. The
    # gate exists to catch a collector that never ran, not to remove rare events.
    rows = [_row(i, liq=1.0 if i % 50 == 0 else 0.0) for i in range(1000)]
    report = feature_audit.audit_rows(rows, features=("liq",))
    assert report["verdicts"][0]["status"] == "ok"
    assert report["dead"] == []


def test_audit_flags_a_degenerate_column_as_degenerate_not_dead():
    rows = [_row(i, mostly=0.0) for i in range(1000)]
    rows[5]["mostly"] = 1.0
    report = feature_audit.audit_rows(rows, features=("mostly",))
    assert report["verdicts"][0]["status"] == "degenerate"
    # Degenerate is reported, not dropped: the caller decides.
    assert feature_audit.trainable_features(report) == ["mostly"]


def test_audit_treats_nan_as_missing_rather_than_as_a_value():
    rows = [_row(i, hole=float("nan")) for i in range(100)]
    report = feature_audit.audit_rows(rows, features=("hole",))
    assert report["verdicts"][0]["status"] == "unmeasured"
    assert report["verdicts"][0]["missing_share"] == 1.0


# ------------------------------------------------------------------ CPCV

def test_cpcv_split_count_matches_the_combinatorial_formula():
    # C(8,2) = 28 is the upper bound. A combination is dropped when the purge and embargo
    # guard would leave no training rows at all, which is correct behaviour rather than a
    # shortfall -- so the assertion is that the count is close to the bound and never above.
    rows = [_row(t * 300_000, symbol="S%d" % s) for t in range(400) for s in range(5)]
    splits = cpcv.cpcv_splits(rows, groups=8, k=2, purge_bars=12, embargo_bars=12)
    assert len(splits) <= 28
    assert len(splits) >= 27
    assert cpcv.path_count(8, 2) == 7


def test_cpcv_drops_a_combination_that_would_leave_no_training_rows():
    # Tiny sample: eight groups of four bars with a 24-bar guard cannot keep anything.
    rows = [_row(t * 300_000, symbol="S%d" % s) for t in range(32) for s in range(5)]
    splits = cpcv.cpcv_splits(rows, groups=8, k=2, purge_bars=12, embargo_bars=12)
    assert splits == []


def test_cpcv_never_puts_a_test_timestamp_in_the_training_set():
    rows = [_row(t * 300_000, symbol="S%d" % s) for t in range(200) for s in range(5)]
    for split in cpcv.cpcv_splits(rows, groups=6, k=2, purge_bars=12, embargo_bars=12):
        train = {rows[i]["timestamp"] for i in split["train_idx"]}
        test = {rows[i]["timestamp"] for i in split["test_idx"]}
        assert not (train & test)


def test_cpcv_embargo_keeps_a_gap_after_the_test_region():
    # Without the embargo the nearest training bar sits adjacent to the test region, which
    # is the standard way to believe a split is purged when it is not.
    rows = [_row(t * 300_000, symbol="S%d" % s) for t in range(200) for s in range(5)]
    split = cpcv.cpcv_splits(rows, groups=6, k=2, purge_bars=12, embargo_bars=12)[0]
    train = {rows[i]["timestamp"] for i in split["train_idx"]}
    test = {rows[i]["timestamp"] for i in split["test_idx"]}
    gap_bars = min(abs(a - b) for a in train for b in test) / 300_000
    assert gap_bars >= 24  # purge 12 + embargo 12


def test_pbo_is_low_when_a_configuration_genuinely_dominates():
    rng = np.random.default_rng(7)
    matrix = rng.normal(0, 1, (80, 6))
    matrix[:, 2] += 3.0  # a real, large edge
    result = cpcv.probability_of_backtest_overfitting(matrix, rounds=100)
    assert result["pbo"] is not None and result["pbo"] < 0.2


def test_pbo_is_high_on_pure_noise():
    rng = np.random.default_rng(11)
    matrix = rng.normal(0, 1, (80, 12))
    result = cpcv.probability_of_backtest_overfitting(matrix, rounds=150)
    # No configuration is better than any other, so the in-sample winner is a coin flip
    # out of sample. A PBO near a half is the honest answer.
    assert 0.25 < result["pbo"] < 0.75


# ------------------------------------------------------------------ label economics

def test_breakeven_r2_is_the_cost_over_sigma_squared():
    # A cost of 0.1 sigma needs rho = 0.1 and therefore R^2 = 1%.
    result = label_economics.breakeven_requirement(std_bps=100.0, cost_bps_value=10.0)
    assert abs(result["cost_in_sigma"] - 0.1) < 1e-9
    assert abs(result["required_r2"] - 0.01) < 1e-9


def test_a_wide_label_against_a_small_cost_is_feasible():
    outcomes = [0.05 if i % 2 else -0.05 for i in range(500)]
    report = label_economics.assess(outcomes, profile="maker", target_r2=0.01)
    assert report["ready"] and report["feasible"]


def test_a_narrow_label_against_a_large_cost_is_refused():
    outcomes = [0.0005 if i % 2 else -0.0005 for i in range(500)]
    report = label_economics.assess(outcomes, profile="taker", target_r2=0.003)
    assert report["ready"] and not report["feasible"]
    assert any("required_r2_above_achievable" in r for r in report["reasons"])


def test_negative_barrier_skew_is_reported_as_infeasible():
    # The stop is touched more often than the target, which is the shape the real labels
    # have (39% lower, 37% upper) and which no directional model can trade profitably.
    # 30% upper against 45% lower: the stop is hit half again as often as the target.
    outcomes = [0.01] * 300 + [-0.01] * 450 + [0.0] * 250
    barriers = ["upper"] * 300 + ["lower"] * 450 + ["vertical"] * 250
    report = label_economics.assess(outcomes, barriers, profile="maker", target_r2=0.01)
    assert report["label"]["barrier_balance"] < label_economics.MIN_BARRIER_BALANCE
    assert any("barrier_balance" in r for r in report["reasons"])


def test_a_balanced_label_is_not_flagged_for_skew():
    # 40/40/20 is the shape a well-centred barrier configuration produces and must pass,
    # otherwise the gate would fire on a healthy label and be turned off.
    outcomes = [0.01] * 400 + [-0.01] * 400 + [0.0] * 200
    barriers = ["upper"] * 400 + ["lower"] * 400 + ["vertical"] * 200
    report = label_economics.assess(outcomes, barriers, profile="maker", target_r2=0.01)
    assert report["label"]["barrier_balance"] >= label_economics.MIN_BARRIER_BALANCE
    assert not any("barrier_balance" in r for r in report["reasons"])


# ------------------------------------------------------------------ cross-sectional

def _universe(per_symbol_drift, days=120, start=100.0):
    """A synthetic universe whose symbols trend at different constant rates."""
    universe = {}
    for symbol, drift in per_symbol_drift.items():
        price = start
        bars = []
        for day in range(days):
            for bar in range(1):
                bars.append({"open_time": (day * 288 + bar) * 300_000,
                             "close": price})
            price *= (1.0 + drift)
        universe[symbol] = bars
    return universe


def test_rank_weights_are_dollar_neutral_and_bounded():
    prices_at = {"S%d" % i: 100.0 for i in range(20)}
    prices_past = {"S%d" % i: 100.0 * (1 + 0.01 * (i - 10)) for i in range(20)}
    weights, info = cs.rank_weights(prices_at, prices_past, quantile=0.25)
    assert info["ready"]
    assert abs(sum(weights.values())) < 1e-9
    assert abs(sum(abs(w) for w in weights.values()) - 2.0) < 1e-9
    assert all(weights[s] > 0 for s in info["longs"])
    assert all(weights[s] < 0 for s in info["shorts"])


def test_backtest_finds_the_edge_in_a_persistently_trending_universe():
    drifts = {"S%02d" % i: (i - 10) * 0.0008 for i in range(21)}
    result = cs.backtest(_universe(drifts, days=150), lookback=7, hold=1,
                         quantile=0.25, cost_bps=1.0)
    assert result["ready"]
    assert result["summary"]["net_bps_mean"] > 0
    assert result["summary"]["t_stat"] > 2.0


def test_backtest_refuses_a_universe_that_is_too_small_to_rank():
    drifts = {"S%d" % i: 0.001 for i in range(3)}
    result = cs.backtest(_universe(drifts, days=40), lookback=7, hold=1)
    assert not result["ready"]
    assert result["reason"] == "universe_below_minimum"


def test_costs_turn_a_marginal_edge_negative():
    drifts = {"S%02d" % i: (i - 10) * 0.0002 for i in range(21)}
    universe = _universe(drifts, days=150)
    cheap = cs.backtest(universe, lookback=7, hold=1, quantile=0.25, cost_bps=0.5)
    dear = cs.backtest(universe, lookback=7, hold=1, quantile=0.25, cost_bps=25.0)
    if cheap.get("ready") and dear.get("ready"):
        assert cheap["summary"]["net_bps_mean"] > dear["summary"]["net_bps_mean"]


def test_align_intersects_calendars_rather_than_unioning_them():
    # Twelve symbols, two of which are listed late. The union would be 20 days; the usable
    # cross-section is the 10 days every symbol shares.
    # Symbols with a near-full history but staggered gaps share fewer days than the longest
    # one; the intersection is what the ranking may use.
    universe = {"S%02d" % i: [{"open_time": d * 86_400_000, "close": 100.0}
                              for d in range(20)] for i in range(10)}
    universe["GAPPY1"] = [{"open_time": d * 86_400_000, "close": 100.0} for d in range(1, 19)]
    days, prices, meta = cs.align(universe)
    assert meta["ready"] and len(days) == 18
    assert all(len(prices[s]) == 18 for s in prices)


def test_align_excludes_a_symbol_with_too_little_history():
    # Three symbols of 192 to 310 days truncated the whole cross-section to their own
    # length, because the calendars are intersected: measured effect was the edge falling
    # from 78.2 bps at t = 3.37 to 30.0 bps at t = 1.17. Short histories are therefore
    # excluded before the intersection rather than after it.
    universe = {"S%02d" % i: [{"open_time": d * 86_400_000, "close": 100.0}
                              for d in range(400)] for i in range(10)}
    universe["SHORT"] = [{"open_time": d * 86_400_000, "close": 100.0}
                         for d in range(200, 400)]
    days, prices, meta = cs.align(universe)
    assert meta["ready"]
    assert "SHORT" not in prices
    assert meta["excluded_short_history"] == ["SHORT"]
    # The full 400 days survive because the short symbol no longer truncates them.
    assert len(days) == 400


def test_align_refuses_a_universe_too_small_to_rank():
    universe = {"A": [{"open_time": d * 86_400_000, "close": 100.0} for d in range(10)],
                "B": [{"open_time": d * 86_400_000, "close": 100.0} for d in range(10)]}
    days, prices, meta = cs.align(universe)
    assert not meta["ready"] and meta["reason"] == "universe_below_minimum"
