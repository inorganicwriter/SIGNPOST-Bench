import json
from unittest.mock import patch

import pytest

from analysis.compute_tfr import (
    equal_dataset_macro,
    geocode_text,
    load_prediction_map,
    prepare_geocode_cache,
    summarize_trap_distance_reductions,
)


def test_transient_geocoding_failure_is_not_cached():
    cache = {}
    with (
        patch("analysis.compute_tfr.time.sleep"),
        patch("analysis.compute_tfr.urlopen", side_effect=RuntimeError("temporary failure")),
    ):
        lat, lon = geocode_text("Test Place", cache)

    assert lat is None
    assert lon is None
    assert "test place" not in cache


def test_load_prediction_map_keeps_latest_valid_prediction(tmp_path):
    result_file = tmp_path / "results_Blank_model.jsonl"
    rows = [
        {"original_source": "a", "pred_lat": 1.0, "pred_lon": 2.0},
        {"original_source": "outside", "pred_lat": 3.0, "pred_lon": 4.0},
        {"original_source": "a", "pred_lat": 5.0, "pred_lon": 6.0},
        {"original_source": "invalid", "pred_lat": None, "pred_lon": None},
    ]
    result_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_prediction_map(result_file, {"a", "invalid"}) == {"a": (5.0, 6.0)}


def test_frozen_cache_excludes_missing_queries_without_network_or_write():
    cache = {"known": {"lat": 1.0, "lon": 2.0}}
    with (
        patch("analysis.compute_tfr.geocode_text") as geocode,
        patch("analysis.compute_tfr.save_geocode_cache") as save,
    ):
        missing = prepare_geocode_cache({"known", "missing"}, cache, geocode_missing=False)

    assert missing == ["missing"]
    assert cache == {"known": {"lat": 1.0, "lon": 2.0}}
    geocode.assert_not_called()
    save.assert_not_called()


def test_tdr_sign_and_attraction_rate():
    # Positive means Adversarial is closer to the trap than the paired Blank output.
    summary = summarize_trap_distance_reductions([80.0, -20.0, 0.0])

    assert summary["paired_count"] == 3
    assert summary["mean_km"] == 20.0
    assert summary["median_km"] == 0.0
    assert summary["attraction_rate_percent"] == pytest.approx(100 / 3)


def test_tdr_empty_pair_set_is_explicit():
    assert summarize_trap_distance_reductions([]) == {
        "paired_count": 0,
        "mean_km": None,
        "median_km": None,
        "attraction_rate_percent": None,
    }


def test_macro_average_is_equal_across_datasets_not_sample_weighted():
    datasets = [
        {"mean_trap_distance_reduction_km": 100.0, "paired_count": 10},
        {"mean_trap_distance_reduction_km": 0.0, "paired_count": 1000},
    ]

    assert equal_dataset_macro(datasets, "mean_trap_distance_reduction_km") == 50.0
