import json
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from run_stunet_campaign import failures, historical_active_seconds, phase_succeeded, report

from stunet_campaign import (
    aggregate_rows,
    sample_targets,
    segmentation_metrics,
    select_evaluation_cases,
)


def test_evaluation_selection_is_deterministic_and_excludes_used():
    candidates = [f"case_{value:05d}" for value in range(12)]
    first = select_evaluation_cases(candidates, {"case_00003"}, seed=20260915, count=6)
    second = select_evaluation_cases(reversed(candidates), {"case_00003"}, seed=20260915, count=6)
    assert first == second
    assert len(first) == len(set(first)) == 6
    assert "case_00003" not in first


def test_sampling_quota_and_missing_class_substitution():
    full = sample_targets([0, 1, 2, 3], "case_00001", 0)
    assert full.count(2) == 4
    assert full.count(1) == 2
    assert full.count(3) == 1
    assert full.count(0) == 1
    missing = sample_targets([0, 1, 2], "case_00001", 0)
    assert 3 not in missing
    assert missing.count(2) == 5


def test_raw_and_hec_metrics_are_distinct():
    target = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    prediction = np.array([[[0, 2, 2, 2]]], dtype=np.uint8)
    result = segmentation_metrics(prediction, target)
    assert result["classes"]["1"]["dice"] == 0.0
    assert result["classes"]["2"]["dice"] == pytest.approx(2 / 4)
    assert result["hec"]["kidney_and_masses"]["dice"] == 1.0
    assert result["hec"]["kidney_mass"]["dice"] == pytest.approx(4 / 5)
    assert result["hec"]["tumor"]["dice"] == pytest.approx(2 / 4)


def test_aggregate_rows_reports_macro_and_micro():
    first = segmentation_metrics(np.array([0, 2]), np.array([0, 2]))
    second = segmentation_metrics(np.array([0, 0]), np.array([0, 2]))
    result = aggregate_rows([first, second])
    assert result["classes"]["2"]["mean_dice"] == pytest.approx(0.5)
    assert result["classes"]["2"]["micro_dice"] == pytest.approx(2 / 3)


def test_failed_phase_is_retried_and_only_success_is_skipped():
    state = {"phases": {"pilot": {"status": "failed"}}}
    assert not phase_succeeded(state, "pilot")
    state["phases"]["pilot"]["status"] = "success"
    assert phase_succeeded(state, "pilot")


def test_historical_active_time_ignores_external_wall_pause():
    state = {"started_at": 1.0, "finished_at": 10001.0, "phases": {
        "evaluation_staging": {"status": "success", "seconds": 173.0},
        "pilot": {"status": "failed", "seconds": 106.0},
    }}
    assert historical_active_seconds(state) == 279.0
    state["active_seconds"] = 300.0
    assert historical_active_seconds(state) == 300.0


def test_evaluation_swap_guard_is_stricter_than_training_guard():
    mib = 2 ** 20
    gib = 2 ** 30
    state = {
        "disk_free_bytes": 30 * gib, "available_ram_bytes": 6 * gib,
        "swap_used_bytes": 200 * mib, "process_tree_rss_bytes": 0,
    }
    assert "swap_limit" not in failures(state, swap_limit=256 * mib)
    assert "swap_limit" in failures(state, swap_limit=192 * mib)


def test_report_is_partial_until_all_six_evaluations_exist(tmp_path):
    result = report(tmp_path)
    assert result["status"] == "partial"
    assert result["rows"] == []


def test_five_gib_ram_floor_applies_only_to_initial_start():
    gib = 2 ** 30
    state = {
        "disk_free_bytes": 30 * gib, "available_ram_bytes": 4 * gib,
        "swap_used_bytes": 0, "process_tree_rss_bytes": 0,
    }
    assert "available_ram_floor" in failures(state, startup=True)
    assert "available_ram_floor" not in failures(state, startup=False)


def test_startup_disk_reserve_can_follow_a_smaller_campaign_ceiling():
    gib = 2 ** 30
    state = {
        "disk_free_bytes": 23 * gib, "available_ram_bytes": 6 * gib,
        "swap_used_bytes": 0, "process_tree_rss_bytes": 0,
    }
    assert "physical_disk_floor" in failures(state, startup=True)
    assert "physical_disk_floor" not in failures(
        state, startup=True, startup_reserve_bytes=2 * gib)
