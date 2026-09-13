"""Universe classification and the pass/fail reading of a training run."""
import os

from app.models.training_job import DEFAULTS, candidates_root, dataset_path, live_candidates_root, passes, promote_candidate, verdict_for
from app.features.universe import RULES, TIER_EXCLUDED, TIER_MAINSTREAM, TIER_SPECULATIVE, classify, classify_one, describe, summarise, symbols_for

DAY = 86_400_000
NOW = 1_800_000_000_000


def market(symbol, volume, count, price, age_days, span_pct):
    high = price * (1 + span_pct / 200.0)
    low = price * (1 - span_pct / 200.0)
    return {"symbol": symbol, "volume": volume, "count": count, "price": price,
            "high": high, "low": low, "open_time": NOW - int(age_days * DAY)}


def mainstream_row(symbol="BTCUSDT"):
    return market(symbol, 5e9, 2_000_000, 78000.0, 2500, 3.0)


def test_deep_established_and_calm_is_mainstream():
    tier, reasons = classify_one(describe(mainstream_row(), NOW))
    assert tier == TIER_MAINSTREAM


def test_high_volatility_moves_a_symbol_out_of_mainstream():
    # "Fast money" is mostly defined by the violence of the move, not by the name.
    row = market("IOSTUSDT", 1.3e9, 20_000_000, 0.001, 2400, 121.6)
    facts = describe(row, NOW)
    assert facts["daily_range_pct"] > 100
    tier, reasons = classify_one(facts)
    assert tier == TIER_SPECULATIVE
    assert any("daily range" in reason for reason in reasons)
    assert any("price below" in reason for reason in reasons)


def test_a_recent_listing_is_speculative_not_mainstream():
    row = market("HYPEUSDT", 6.8e8, 2_000_000, 83.0, 468, 5.6)
    tier, reasons = classify_one(describe(row, NOW))
    assert tier == TIER_SPECULATIVE
    assert any("listed only" in reason for reason in reasons)


def test_thin_or_too_new_is_excluded():
    thin = market("THINUSDT", 1e6, 500, 1.0, 900, 20.0)
    fresh = market("NEWUSDT", 5e8, 900_000, 5.0, 11, 30.0)
    assert classify_one(describe(thin, NOW))[0] == TIER_EXCLUDED
    assert classify_one(describe(fresh, NOW))[0] == TIER_EXCLUDED


def test_every_symbol_gets_a_reason():
    graded = classify([mainstream_row(), market("X", 1e6, 1, 1.0, 5, 50.0)], now_ms=NOW)
    assert all(item["reasons"] for item in graded)


def test_classify_sorts_by_liquidity_and_counts_tiers():
    rows = [market("SMALLUSDT", 2e7, 50_000, 1.0, 400, 20.0), mainstream_row(),
            market("THINUSDT", 1e5, 10, 1.0, 400, 20.0)]
    graded = classify(rows, now_ms=NOW)
    assert [item["symbol"] for item in graded] == ["BTCUSDT", "SMALLUSDT", "THINUSDT"]
    counts = summarise(graded)
    assert counts == {TIER_MAINSTREAM: 1, TIER_SPECULATIVE: 1, TIER_EXCLUDED: 1}


def test_symbols_for_respects_the_limit():
    rows = [mainstream_row("A" * 4 + "USDT"), mainstream_row("B" * 4 + "USDT")]
    assert len(symbols_for(TIER_MAINSTREAM, rows, limit=1, now_ms=NOW)) == 1


def test_missing_range_does_not_crash_and_falls_through():
    row = {"symbol": "X", "volume": 5e9, "count": 2e6, "price": 10.0, "open_time": 0}
    facts = describe(row, NOW)
    assert facts["daily_range_pct"] is None
    assert classify_one(facts)[0] in (TIER_SPECULATIVE, TIER_EXCLUDED)


def test_rules_are_configurable():
    strict = {TIER_MAINSTREAM: {**RULES[TIER_MAINSTREAM], "min_quote_volume": 1e12},
              TIER_SPECULATIVE: RULES[TIER_SPECULATIVE]}
    assert classify_one(describe(mainstream_row(), NOW), strict)[0] != TIER_MAINSTREAM


def walk_forward_result(active=5000, edge=20.0, prob=0.99, low=5.0, high=40.0,
                        positive=8, negative=0):
    return {"aggregate": {"active_samples": active, "net_edge_bps": edge,
                          "folds_positive_edge": positive, "folds_negative_edge": negative},
            "bootstrap": {"prob_positive": prob, "low_bps": low, "high_bps": high}}


def test_a_clean_result_passes():
    result = walk_forward_result()
    assert passes(result, DEFAULTS)
    assert "passed every check" in verdict_for(result, DEFAULTS)[0]


def test_too_few_trades_fails_even_with_a_positive_edge():
    # This is the trap the earlier single-split number walked into: a large edge on a
    # handful of trades is not evidence.
    result = walk_forward_result(active=111, edge=35.8, prob=0.69)
    assert not passes(result, DEFAULTS)
    lines = verdict_for(result, DEFAULTS)
    assert any("too few" in line for line in lines)


def test_a_negative_edge_never_passes():
    result = walk_forward_result(edge=-17.1, prob=0.02, low=-34.4, high=-0.1)
    assert not passes(result, DEFAULTS)
    assert any("would lose money" in line for line in verdict_for(result, DEFAULTS))


def test_an_interval_straddling_zero_is_flagged():
    result = walk_forward_result(edge=6.9, prob=0.69, low=-15.6, high=35.9)
    assert not passes(result, DEFAULTS)
    lines = verdict_for(result, DEFAULTS)
    assert any("straddles zero" in line for line in lines)


def test_more_losing_folds_than_winning_ones_is_flagged():
    result = walk_forward_result(edge=10.0, prob=0.97, positive=4, negative=5)
    assert any("more losing folds" in line for line in verdict_for(result, DEFAULTS))


def test_dataset_and_candidate_paths_are_plain_strings():
    # Regression: the inline expression formatted the Path instead of the filename, so
    # every run died with "unsupported operand type(s) for %: WindowsPath and str".
    for tier in (TIER_MAINSTREAM, TIER_SPECULATIVE):
        path = dataset_path("data", tier)
        assert isinstance(path, str)
        assert path.endswith("training_dataset_%s.jsonl" % tier)
        assert "research_v3" in path
        root = candidates_root("data", tier)
        assert isinstance(root, str) and root.endswith("candidates_%s" % tier)


def test_the_two_tiers_write_to_different_files():
    assert dataset_path("data", TIER_MAINSTREAM) != dataset_path("data", TIER_SPECULATIVE)


def test_promotion_copies_a_candidate_into_the_live_folder(tmp_path):
    # Training writes per-tier folders, which the live runtime never reads. Promotion is
    # the only bridge, so this path has to work.
    source = tmp_path / "research_v3" / "candidates_mainstream" / "lightgbm-abc"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    (source / "model.txt").write_text("tree", encoding="utf-8")
    live = promote_candidate(str(tmp_path), str(source), "lightgbm")
    import pathlib
    assert pathlib.Path(live) == pathlib.Path(live_candidates_root(str(tmp_path))) / "lightgbm-abc"
    assert (tmp_path / "research_v3" / "candidates" / "lightgbm-abc" / "model.txt").is_file()


def test_promotion_leaves_the_source_in_place(tmp_path):
    source = tmp_path / "research_v3" / "candidates_speculative" / "catboost-x"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    promote_candidate(str(tmp_path), str(source), "catboost")
    assert source.is_dir()


def test_promotion_replaces_a_previous_artifact_of_the_same_name(tmp_path):
    source = tmp_path / "research_v3" / "candidates_mainstream" / "lightgbm-same"
    source.mkdir(parents=True)
    (source / "model.txt").write_text("first", encoding="utf-8")
    promote_candidate(str(tmp_path), str(source), "lightgbm")
    (source / "model.txt").write_text("second", encoding="utf-8")
    promote_candidate(str(tmp_path), str(source), "lightgbm")
    live = tmp_path / "research_v3" / "candidates" / "lightgbm-same" / "model.txt"
    assert live.read_text(encoding="utf-8") == "second"


def test_promotion_rejects_a_missing_candidate(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="candidate_missing"):
        promote_candidate(str(tmp_path), str(tmp_path / "nope"), "lightgbm")


def test_live_folder_is_separate_from_the_per_tier_folders():
    for tier in (TIER_MAINSTREAM, TIER_SPECULATIVE):
        assert candidates_root("data", tier) != live_candidates_root("data")


def test_a_failing_run_is_never_promoted():
    # The gate is the whole point: -41bp out of sample must not reach the agent.
    losing = walk_forward_result(active=24100, edge=-41.1, prob=0.07, low=-96.5,
                                 high=12.5, positive=4, negative=7)
    assert not passes(losing, DEFAULTS)


def test_no_result_is_reported_not_passed():
    assert not passes(None, DEFAULTS)
    assert verdict_for(None, DEFAULTS) == ["no walk-forward result"]
