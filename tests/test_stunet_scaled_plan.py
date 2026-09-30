import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from run_stunet_train_scaled import make_plan, plan_digest
from stunet_campaign import csv_case_ids


def config():
    return json.loads((ROOT / "research/stunet_campaign_config.json").read_text())


def test_scaled_plan_is_frozen_disjoint_and_nested():
    setting = config()
    p96 = make_plan(setting, 96)
    p128 = make_plan(setting, 128)
    original = csv_case_ids(Path(setting["train_csv"]))
    validation = set(csv_case_ids(Path(setting["validation_csv"])))
    assert p96["train_cases"][:64] == original
    assert p128["train_cases"][:96] == p96["train_cases"]
    assert len(set(p128["train_cases"])) == 128
    assert not set(p128["train_cases"]) & validation
    assert not set(p128["train_cases"]) & set(p128["development_cases"])
    assert p96["evaluation_cases"] == []
    assert p96["expected_updates"] == 1536
    assert p128["expected_updates"] == 2048
    assert p128["max_pinned_plus_case_bytes"] <= p128["cache_budget_bytes"]
    assert plan_digest(make_plan(setting, 96)) == plan_digest(p96)


def test_scaled_plan_supports_the_full_eligible_cohort():
    setting = config()
    reference = make_plan(setting, 96)
    full_count = reference["original_train_cases"] + reference["eligible_additional_count"]
    full = make_plan(setting, full_count)
    validation = set(csv_case_ids(Path(setting["validation_csv"])))
    assert len(full["train_cases"]) == full_count
    assert len(set(full["train_cases"])) == full_count
    assert not set(full["train_cases"]) & validation
    assert full["max_pinned_plus_case_bytes"] <= full["cache_budget_bytes"]
    assert full["expected_updates"] == full_count * 2 * setting["patches_per_case"]


def test_scaled_plan_rejects_sizes_outside_frozen_eligibility():
    setting = config()
    reference = make_plan(setting, 96)
    full_count = reference["original_train_cases"] + reference["eligible_additional_count"]
    with pytest.raises(ValueError, match="at least the frozen 64"):
        make_plan(setting, 63)
    with pytest.raises(ValueError, match="only .* fit the frozen protocol"):
        make_plan(setting, full_count + 1)
