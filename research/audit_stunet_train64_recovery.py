#!/usr/bin/env python3
"""Audit the 64-case training boundary and paired development recovery.

The interrupted supervisor receipt is preserved. This writes a separate
post-guard receipt and makes no independent-efficacy claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_stunet_train64 import make_plan, plan_digest
from stunet_campaign import atomic_json, sha256_file
from stunet_campaign_evaluate import load_checkpoint_safely


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(session: Path) -> dict:
    config = load_json(session / "resolved-config.json")
    plan = make_plan(config)
    frozen = load_json(session / "plan.json")
    if frozen != {"plan": plan, "sha256": plan_digest(plan)}:
        raise ValueError("frozen plan changed")
    original = load_json(session / "supervisor.json")
    if original.get("status") != "guard_stopped" or "swap_limit" not in original.get("phase", {}).get("reasons", []):
        raise ValueError("unexpected original supervisor status")
    events = [json.loads(line) for line in (session / "arm_A/metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = [row for row in events if row.get("kind") == "train_step"]
    boundaries = [row for row in events if row.get("kind") == "case_boundary"]
    expected = plan["expected_updates"]
    if len(steps) != expected or len(boundaries) != len(plan["train_cases"]) * plan["epochs"]:
        raise ValueError("training event count mismatch")
    if [row["global_step"] for row in boundaries] != list(range(plan["patches_per_case"], expected + 1, plan["patches_per_case"])):
        raise ValueError("case boundaries are not sequential")
    if not all(math.isfinite(float(row["loss"])) for row in steps):
        raise ValueError("nonfinite training loss")
    checkpoint = session / "arm_A/latest.checkpoint.pt"
    payload = load_checkpoint_safely(checkpoint)
    expected_schedule = {
        "train_cases": plan["train_cases"],
        "development_cases": plan["development_cases"],
        "seed": config["training_seed"],
        "patches_per_case": plan["patches_per_case"],
        "epochs": plan["epochs"],
    }
    if (payload.get("schema_version") != "shadowtrainer.stunet-checkpoint/v1"
            or payload.get("schedule") != expected_schedule
            or payload.get("base_sha256") != plan["base_checkpoint_sha256"]
            or (payload.get("global_step"), payload.get("next_epoch"), payload.get("next_case_index")) != (expected, plan["epochs"], 0)):
        raise ValueError("last checkpoint does not match confirmed boundary")
    b0 = load_json(session / "evaluation/development/B0/summary.json")
    candidate = load_json(session / "evaluation/development/A/summary.json")
    base_sha = sha256_file(Path(config["base_checkpoint"]))
    checkpoint_sha = sha256_file(checkpoint)
    if b0.get("status") != "success" or candidate.get("status") != "success":
        raise ValueError("paired development evaluation incomplete")
    if b0.get("checkpoint_sha256") != base_sha or candidate.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("evaluated checkpoint identity mismatch")
    if b0.get("cases_requested") != plan["development_cases"] or candidate.get("cases_requested") != plan["development_cases"]:
        raise ValueError("development cohort mismatch")
    if b0.get("protocol") != candidate.get("protocol") or b0["protocol"].get("surface_voxel_limit") != 0:
        raise ValueError("paired protocol mismatch")
    b0_rows, a_rows = b0["cases"], candidate["cases"]
    if len(b0_rows) != len(a_rows) or len(a_rows) != len(plan["development_cases"]):
        raise ValueError("paired case count mismatch")
    pairs = []
    for left, right in zip(b0_rows, a_rows):
        if left["case_id"] != right["case_id"] or not left.get("provenance_verified") or not right.get("provenance_verified"):
            raise ValueError("paired case provenance mismatch")
        pairs.append({
            "case_id": left["case_id"],
            "b0_kidney_dice": left["classes"]["1"]["dice"],
            "a_kidney_dice": right["classes"]["1"]["dice"],
            "b0_tumor_dice": left["classes"]["2"]["dice"],
            "a_tumor_dice": right["classes"]["2"]["dice"],
            "b0_prediction_sha256": left["prediction_sha256"],
            "a_prediction_sha256": right["prediction_sha256"],
        })
    baseline_kidney = b0["aggregate"]["classes"]["1"]["mean_dice"]
    baseline_tumor = b0["aggregate"]["classes"]["2"]["mean_dice"]
    candidate_kidney = candidate["aggregate"]["classes"]["1"]["mean_dice"]
    candidate_tumor = candidate["aggregate"]["classes"]["2"]["mean_dice"]
    del payload  # Avoid retaining the model alongside the audit output.
    return {
        "schema_version": "shadowtrainer.stunet-train64-recovery/v1",
        "status": "training_complete_development_recovered",
        "original_supervisor_status": original["status"],
        "original_guard_reasons": original["phase"]["reasons"],
        "plan_sha256": frozen["sha256"],
        "base_checkpoint_sha256": base_sha,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "train_cases": len(plan["train_cases"]),
        "development_cases": len(plan["development_cases"]),
        "epochs": plan["epochs"],
        "confirmed_updates": expected,
        "finite_training_losses": True,
        "development_protocol": b0["protocol"],
        "b0_mean_kidney_dice": baseline_kidney,
        "a_mean_kidney_dice": candidate_kidney,
        "delta_mean_kidney_dice": candidate_kidney - baseline_kidney,
        "b0_mean_tumor_dice": baseline_tumor,
        "a_mean_tumor_dice": candidate_tumor,
        "delta_mean_tumor_dice": candidate_tumor - baseline_tumor,
        "tumor_improved_cases": sum(row["a_tumor_dice"] > row["b0_tumor_dice"] for row in pairs),
        "tumor_worsened_cases": sum(row["a_tumor_dice"] < row["b0_tumor_dice"] for row in pairs),
        "candidate_eligible_by_mean_development_rule": candidate_kidney >= baseline_kidney - 0.02 and candidate_tumor > baseline_tumor,
        "paired_cases": pairs,
        "limitations": [
            "four model-selection development cases only",
            "tumor mean is dominated by one improved case",
            "no independent or clinical validation",
            "HD95 skipped for the bounded recovery evaluation",
            "original supervisor stopped on swap guard after training boundary",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.session.resolve())
    output = args.session / "recovery-audit.json"
    atomic_json(output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "confirmed_updates", "development_cases",
        "delta_mean_kidney_dice", "delta_mean_tumor_dice",
        "tumor_improved_cases", "tumor_worsened_cases",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
