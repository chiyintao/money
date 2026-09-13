"""Walk-forward evaluation: fold construction and the bootstrap interval."""
import numpy as np
import pytest

from app.models.walkforward import aggregate, bootstrap_edge, time_blocks, walk_forward_tabular
from helpers import row


def test_time_blocks_split_on_time_not_rows():
    rows = [row(i, symbol=s) for s in ("A", "B") for i in range(40)]
    blocks = time_blocks(rows, 4)
    assert len(blocks) == 4
    assert blocks[0][0] == 0
    # Every block holds the same timestamps regardless of symbol count.
    assert all(len(block) == 10 for block in blocks)
    assert blocks[0][-1] < blocks[1][0]


def test_too_few_timestamps_for_the_requested_folds_is_rejected():
    rows = [row(i) for i in range(10)]
    with pytest.raises(ValueError, match="too_few_time_groups"):
        time_blocks(rows, 12)


def test_invalid_fold_count_is_rejected():
    with pytest.raises(ValueError, match="invalid_fold_count"):
        time_blocks([row(0)], 0)


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.004, size=600)
    result = bootstrap_edge(returns, iterations=400)
    assert result["samples"] == 600
    assert result["low_bps"] < result["mean_bps"] < result["high_bps"]
    assert 0.0 <= result["prob_positive"] <= 1.0


def test_bootstrap_reports_a_negative_edge_when_returns_are_negative():
    returns = np.full(400, -0.002) + np.linspace(-0.0001, 0.0001, 400)
    result = bootstrap_edge(returns, iterations=300)
    assert result["prob_positive"] == 0.0
    assert result["high_bps"] < 0


def test_block_bootstrap_is_wider_than_an_iid_one_when_returns_are_persistent():
    # Overlapping labels make neighbouring trades dependent; the block interval must not
    # pretend the sample is larger than it is.
    rng = np.random.default_rng(7)
    persistent = np.repeat(rng.normal(0, 0.005, size=60), 10)
    blocked = bootstrap_edge(persistent, iterations=400)
    iid = bootstrap_edge(persistent, iterations=400, block=1)
    assert (blocked["high_bps"] - blocked["low_bps"]) > (iid["high_bps"] - iid["low_bps"])


def test_bootstrap_handles_too_few_samples():
    result = bootstrap_edge(np.array([0.01]))
    assert result["samples"] == 1
    assert result["iterations"] == 0
    assert bootstrap_edge(np.array([]))["mean_bps"] == 0.0


def test_aggregate_weights_the_edge_by_active_trades():
    folds = [
        {"rows": 100, "active_samples": 90, "active_net_edge_bps": 100.0,
         "directional_accuracy_pct": 55.0},
        {"rows": 100, "active_samples": 10, "active_net_edge_bps": -100.0,
         "directional_accuracy_pct": 45.0},
    ]
    result = aggregate(folds)
    assert result["active_samples"] == 100
    assert result["net_edge_bps"] == pytest.approx(80.0)
    assert result["folds_positive_edge"] == 1
    assert result["folds_negative_edge"] == 1
    assert aggregate([]) == {"folds": 0}


def test_walk_forward_produces_out_of_sample_folds():
    pytest.importorskip("lightgbm")
    rows = [row(i, symbol="A", label_end_time=i + 12,
                future_return=(0.01 if i % 2 else -0.01)) for i in range(600)]
    result = walk_forward_tabular(rows, "lightgbm", folds=5, cost_bps=1, rounds=5,
                                  horizon=12)
    assert result["folds"] >= 3
    assert result["purge_bars"] == 12
    for fold in result["per_fold"]:
        assert fold["test_rows"] > 0 and fold["train_rows"] > 0
    assert "net_edge_bps" in result["aggregate"]
    assert "prob_positive" in result["bootstrap"]


def test_walk_forward_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unsupported_backend"):
        walk_forward_tabular([row(0)], "transformer", folds=2)
