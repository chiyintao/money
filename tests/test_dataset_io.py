"""The streaming dataset profile must reproduce the historical digest exactly."""
import hashlib
import json
import os
import time

from app.models.dataset_io import cached_profile, profile_cache_path, rows_digest, stream_digest, stream_profile
from app.features.feature_spec import FEATURES
from helpers import row


def write_dataset(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item) + "\n")
    return path


def legacy_digest(rows):
    """The algorithm the live loader and the trainer used before streaming."""
    ordered = sorted(rows, key=lambda item: (item["timestamp"], item["symbol"]))
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample(count=40):
    rows = [row(i, symbol=s, label_end_time=i + 1) for s in ("BTC", "ETH") for i in range(count)]
    rows.sort(key=lambda item: (item["timestamp"], item["symbol"]))
    return rows


def test_streamed_digest_matches_the_legacy_algorithm(tmp_path):
    # Existing model manifests record the legacy digest. If streaming changed it, every
    # candidate would silently fail verification and the agent would stop trading.
    rows = sample()
    path = write_dataset(tmp_path / "d.jsonl", rows)
    assert stream_digest(str(path)) == legacy_digest(rows)


def test_digest_is_stable_when_the_file_is_written_with_unsorted_keys(tmp_path):
    # The builder writes sorted keys, but the digest must not depend on that.
    rows = sample(5)
    shuffled_keys = [{k: item[k] for k in reversed(list(item))} for item in rows]
    path = write_dataset(tmp_path / "d.jsonl", shuffled_keys)
    assert stream_digest(str(path)) == legacy_digest(rows)


def test_rows_digest_matches_the_file_digest(tmp_path):
    rows = sample(7)
    path = write_dataset(tmp_path / "d.jsonl", rows)
    assert rows_digest(rows) == stream_digest(str(path))


def test_profile_reports_rows_bounds_and_symbols(tmp_path):
    rows = sample(10)
    path = write_dataset(tmp_path / "d.jsonl", rows)
    profile = stream_profile(str(path))
    assert profile["rows"] == len(rows)
    assert profile["symbols"] == ["BTC", "ETH"]
    assert set(profile["bounds"]) == set(FEATURES)
    for name in FEATURES:
        values = [item[name] for item in rows]
        # The bounds are tail quantiles of the observed distribution rather than its
        # extremes, and the extremes are reported separately so the difference is visible.
        assert profile["extremes"][name] == [min(values), max(values)]
        low, high = profile["bounds"][name]
        assert min(values) <= low <= high <= max(values)


def test_bounds_narrow_inside_the_clip_range_when_the_data_saturates_it(tmp_path):
    # Regression. The bounds used to be the observed extremes of features that had already
    # been clipped, so a feature that touched its clip limit -- which four of the ten did
    # in the stored dataset -- produced a bound equal to the clip range. The serving gate
    # then tested values that its own pipeline had clipped into that range, and could not
    # fire. Here two bars out of four thousand are flash prints at return_10's limit.
    rows = sample(2000)
    rows[0] = dict(rows[0], return_10=0.5)
    rows[1] = dict(rows[1], return_10=-0.5)
    path = write_dataset(tmp_path / "d.jsonl", rows)
    profile = stream_profile(str(path))
    low, high = profile["bounds"]["return_10"]
    assert profile["extremes"]["return_10"] == [-0.5, 0.5]
    assert (low, high) != (-0.5, 0.5), "a saturated bound is not a bound"
    assert low > -0.5 and high < 0.5


def test_a_tail_the_dataset_is_too_small_to_resolve_falls_back_to_the_extremes(tmp_path):
    # The tail trim needs rows to trim. With sixty of four hundred bars at the limit the
    # 0.1% quantile is still inside that limit, and the honest answer is the extreme --
    # which is exactly the case the quality report exists to surface.
    from app.models.dataset_io import bounds_are_informative

    rows = sample(200)
    for index in range(60):
        rows[index] = dict(rows[index], return_10=0.5 if index % 2 else -0.5)
    path = write_dataset(tmp_path / "d.jsonl", rows)
    profile = stream_profile(str(path))
    assert profile["bounds"]["return_10"] == [-0.5, 0.5]
    quality = bounds_are_informative(profile["bounds"])
    assert "return_10" in quality["degenerate"]
    assert "ema20_gap" not in quality["degenerate"]


def test_a_bound_equal_to_the_clip_range_is_reported_as_uninformative():
    from app.models.dataset_io import bounds_are_informative
    from app.features.feature_spec import FEATURE_CLIP

    saturated = {name: list(limits) for name, limits in FEATURE_CLIP.items()}
    quality = bounds_are_informative(saturated)
    assert quality["informative"] is False
    assert set(quality["degenerate"]) == set(FEATURE_CLIP)

    inside = {name: [limits[0] * 0.5, limits[1] * 0.5] for name, limits in FEATURE_CLIP.items()}
    assert bounds_are_informative(inside)["informative"] is True


def test_blank_lines_are_ignored(tmp_path):
    rows = sample(3)
    path = tmp_path / "d.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n")
        for item in rows:
            handle.write(json.dumps(item) + "\n\n")
    assert stream_profile(str(path))["rows"] == len(rows)


def test_cache_hit_returns_the_same_profile_without_rehashing(tmp_path):
    rows = sample(5)
    path = write_dataset(tmp_path / "d.jsonl", rows)
    first = cached_profile(str(path))
    assert profile_cache_path(path).is_file()
    second = cached_profile(str(path))
    assert second == first


def test_cache_is_discarded_when_the_file_changes(tmp_path):
    # Bounds must never be served from a dataset that has been replaced.
    path = write_dataset(tmp_path / "d.jsonl", sample(5))
    cached_profile(str(path))
    time.sleep(0.01)
    rows = sample(20)
    rows[0] = dict(rows[0], atr_pct=0.09)
    write_dataset(path, rows)
    os.utime(path, (time.time() + 1, time.time() + 1))
    refreshed = cached_profile(str(path))
    assert refreshed["rows"] == len(rows)
    # The upper bound is the top of the bin holding the largest value, so it covers 0.09
    # rather than equalling it exactly.
    upper = refreshed["bounds"]["atr_pct"][1]
    assert 0.09 <= upper <= 0.0901
    assert refreshed["extremes"]["atr_pct"][1] == 0.09


def test_cache_can_be_bypassed(tmp_path):
    path = write_dataset(tmp_path / "d.jsonl", sample(4))
    cached_profile(str(path))
    fresh = cached_profile(str(path), use_cache=False)
    assert fresh["rows"] == 8
