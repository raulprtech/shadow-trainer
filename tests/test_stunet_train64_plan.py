import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from run_stunet_train64 import confirmed_updates, make_plan, plan_digest


def test_real_frozen_plan_is_disjoint_and_bounded():
    config = json.loads((ROOT / "research/stunet_campaign_config.json").read_text())
    plan = make_plan(config)
    assert len(plan["train_cases"]) == 64
    assert len(plan["development_cases"]) == 4
    assert not set(plan["train_cases"]) & set(plan["development_cases"])
    assert plan["evaluation_cases"] == []
    assert plan["expected_updates"] == 1024
    assert plan["max_pinned_plus_case_bytes"] <= plan["cache_budget_bytes"]
    assert len(plan_digest(plan)) == 64


def test_epoch_choice_is_frozen():
    config = json.loads((ROOT / "research/stunet_campaign_config.json").read_text())
    with pytest.raises(ValueError, match="two epochs"):
        make_plan(config, epochs=3)


def test_interrupted_receipt_counts_only_complete_case_boundaries(tmp_path):
    arm = tmp_path / "arm_A"
    arm.mkdir()
    (arm / "metrics.jsonl").write_text(
        json.dumps({"kind": "case_boundary", "global_step": 8}) + "\n"
        + json.dumps({"kind": "train_step", "global_step": 9}) + "\n"
        + json.dumps({"kind": "case_boundary", "global_step": 16}) + "\n"
        + '{"kind": "case_boundary",'
    )
    assert confirmed_updates(tmp_path) == 16
