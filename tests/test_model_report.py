"""The report must flag an unsupported edge rather than present one."""
from scripts.model_report import load_manifest, summarise, verdict


def manifest(**test_overrides):
    test = {"rows": 1000, "directional_accuracy_pct": 49.0, "active_samples": 10,
            "active_share_pct": 1.0, "active_net_edge_bps": 30.0, "active_hit_rate_pct": 60.0}
    test.update(test_overrides)
    return {"backend": "lightgbm", "feature_version": "features-v2", "features": ["rsi"],
            "dataset_rows": 5000, "dataset_symbols": ["A", "B"], "cost_bps": 12,
            "split": {"train": 1, "validation": 2, "test": 3, "purged": 0},
            "metrics": {"train": dict(test), "validation": dict(test), "test": test},
            "metrics_by_symbol": {"A": dict(test), "B": dict(test)}}


def test_below_coin_flip_accuracy_is_flagged():
    assert any("below 50%" in line for line in verdict(manifest(directional_accuracy_pct=48.2)))


def test_a_positive_accuracy_is_not_flagged_for_direction():
    assert not any("below 50%" in line for line in verdict(manifest(directional_accuracy_pct=52.0)))


def test_a_handful_of_active_samples_is_flagged():
    # A large positive net edge on a hundred samples is not evidence of anything.
    lines = verdict(manifest(active_samples=111, active_net_edge_bps=35.8))
    assert any("too few" in line for line in lines)


def test_enough_active_samples_are_not_flagged():
    assert not any("too few" in line for line in verdict(manifest(active_samples=5000)))


def test_a_sign_flip_across_symbols_is_flagged():
    data = manifest()
    data["metrics_by_symbol"] = {
        "A": {"active_samples": 10, "active_net_edge_bps": 40.0},
        "B": {"active_samples": 10, "active_net_edge_bps": -30.0},
    }
    assert any("changes sign" in line for line in verdict(data))


def test_a_consistent_sign_is_not_flagged():
    data = manifest()
    data["metrics_by_symbol"] = {
        "A": {"active_samples": 10, "active_net_edge_bps": 40.0},
        "B": {"active_samples": 10, "active_net_edge_bps": 12.0},
    }
    assert not any("changes sign" in line for line in verdict(data))


def test_symbols_with_no_activity_are_ignored_by_the_sign_check():
    data = manifest()
    data["metrics_by_symbol"] = {
        "A": {"active_samples": 10, "active_net_edge_bps": 40.0},
        "B": {"active_samples": 0, "active_net_edge_bps": 0.0},
    }
    assert not any("changes sign" in line for line in verdict(data))


def test_summarise_prints_every_partition(capsys):
    summarise(manifest())
    out = capsys.readouterr().out
    for token in ("train", "validation", "test", "net_edge", "active"):
        assert token in out


def test_load_manifest_accepts_a_folder_or_a_file(tmp_path):
    import json
    folder = tmp_path / "lightgbm-x"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
    assert load_manifest(str(folder))["backend"] == "lightgbm"
    assert load_manifest(str(folder / "manifest.json"))["backend"] == "lightgbm"
