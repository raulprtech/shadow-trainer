#!/usr/bin/env python3
"""Offline integrity audit of the post-hoc, equal-update 64-case arm."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_stunet_fixed_budget64 import make_plan
from run_stunet_train64 import plan_digest
from stunet_campaign import atomic_json, sha256_file
from stunet_campaign_evaluate import load_checkpoint_safely


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(session: Path) -> dict:
    config = read(session / "resolved-config.json")
    frozen = read(session / "plan.json")
    plan = make_plan(config)
    digest = plan_digest(plan)
    if frozen != {"plan": plan, "sha256": digest}:
        raise ValueError("frozen plan changed")
    receipt = read(session / "supervisor.json")
    expected = plan["expected_updates"]
    if (receipt.get("status") != "success" or
            receipt.get("plan_sha256") != digest or
            receipt.get("confirmed_updates") != expected):
        raise ValueError("supervisor did not complete the frozen plan")
    with (session / "arm_A/metrics.jsonl").open() as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    steps = [row for row in events if row.get("kind") == "train_step"]
    boundaries = [row for row in events if row.get("kind") == "case_boundary"]
    epochs = [row for row in events if row.get("kind") == "epoch"]
    patches = plan["patches_per_case"]
    if len(steps) != expected or len(boundaries) != len(plan["train_cases"]) * plan["epochs"]:
        raise ValueError("missing or duplicate training events")
    if [row["global_step"] for row in boundaries] != list(range(patches, expected + 1, patches)):
        raise ValueError("case boundaries out of order")
    if [row["case_id"] for row in boundaries] != plan["train_cases"] * plan["epochs"]:
        raise ValueError("case order changed")
    if [row["epoch"] for row in epochs] != list(range(1, plan["epochs"] + 1)):
        raise ValueError("development evaluation incomplete")
    if not all(math.isfinite(float(row["loss"])) for row in steps):
        raise ValueError("nonfinite loss")
    checkpoint = session / "arm_A/latest.checkpoint.pt"
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != receipt["last_checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    payload = load_checkpoint_safely(checkpoint)
    schedule = {
        "train_cases": plan["train_cases"],
        "development_cases": plan["development_cases"],
        "seed": config["training_seed"],
        "patches_per_case": patches,
        "epochs": plan["epochs"],
    }
    if (payload.get("schema_version") != "shadowtrainer.stunet-checkpoint/v1" or
            payload.get("schedule") != schedule or
            payload.get("base_sha256") != plan["base_checkpoint_sha256"] or
            (payload.get("global_step"), payload.get("next_epoch"),
             payload.get("next_case_index")) != (expected, plan["epochs"], 0)):
        raise ValueError("checkpoint boundary or provenance mismatch")
    return {
        "schema_version": "shadowtrainer.stunet-fixed-budget64-audit/v1",
        "status": "complete_integrity_only",
        "plan_sha256": digest,
        "last_checkpoint_sha256": checkpoint_sha,
        "train_cases": len(plan["train_cases"]),
        "epochs": plan["epochs"],
        "confirmed_updates": expected,
        "finite_training_losses": True,
        "epoch_development": [{
            "epoch": row["epoch"], "global_step": row["global_step"],
            "kidney_mean": row["development"]["kidney_mean"],
            "tumor_mean": row["development"]["tumor_mean"],
            "eligible": row["eligible"],
        } for row in epochs],
        "limitations": [
            "post-hoc: 96-case result inspected before comparator design",
            "equal updates, unequal case exposure",
            "development only; no independent efficacy test",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    result = audit(session)
    output = session / "training-audit.json"
    if output.exists() and read(output) != result:
        raise ValueError("existing audit differs")
    atomic_json(output, result)
    print(json.dumps({key: result[key] for key in
                      ("status", "train_cases", "confirmed_updates", "epoch_development")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
