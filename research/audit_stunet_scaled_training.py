#!/usr/bin/env python3
"""Offline integrity audit of a complete 96/128-case STU-Net training session."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_stunet_train_scaled import make_plan, plan_digest
from stunet_campaign import atomic_json, sha256_file
from stunet_campaign_evaluate import load_checkpoint_safely


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(session: Path) -> dict:
    config = read(session / "resolved-config.json")
    frozen = read(session / "plan.json")
    count = len(frozen["plan"]["train_cases"])
    plan = make_plan(config, count)
    digest = plan_digest(plan)
    if frozen != {"plan": plan, "sha256": digest}:
        raise ValueError("frozen plan changed")
    receipt = read(session / "supervisor.json")
    expected = plan["expected_updates"]
    if (receipt.get("status") != "success" or
            receipt.get("plan_sha256") != digest or
            receipt.get("confirmed_updates") != expected):
        raise ValueError("supervisor did not complete the frozen plan")
    events = []
    with (session / "arm_A/metrics.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                events.append(json.loads(line))
    steps = [row for row in events if row.get("kind") == "train_step"]
    boundaries = [row for row in events if row.get("kind") == "case_boundary"]
    if len(steps) != expected or len(boundaries) != count * plan["epochs"]:
        raise ValueError("missing or duplicate training events")
    if [row["global_step"] for row in boundaries] != list(range(
            plan["patches_per_case"], expected + 1, plan["patches_per_case"])):
        raise ValueError("case boundaries out of order")
    wanted = plan["train_cases"] * plan["epochs"]
    if [row["case_id"] for row in boundaries] != wanted:
        raise ValueError("case order changed")
    if not all(math.isfinite(float(row["loss"])) for row in steps):
        raise ValueError("nonfinite training loss")
    checkpoint = session / "arm_A/latest.checkpoint.pt"
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != receipt["last_checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    payload = load_checkpoint_safely(checkpoint)
    schedule = {
        "train_cases": plan["train_cases"],
        "development_cases": plan["development_cases"],
        "seed": config["training_seed"],
        "patches_per_case": plan["patches_per_case"],
        "epochs": plan["epochs"],
    }
    if (payload.get("schema_version") != "shadowtrainer.stunet-checkpoint/v1" or
            payload.get("schedule") != schedule or
            payload.get("base_sha256") != plan["base_checkpoint_sha256"] or
            (payload.get("global_step"), payload.get("next_epoch"),
             payload.get("next_case_index")) != (expected, plan["epochs"], 0)):
        raise ValueError("checkpoint boundary or provenance mismatch")
    epoch_rows = [row for row in events if row.get("kind") == "epoch"]
    if [row["epoch"] for row in epoch_rows] != [1, 2]:
        raise ValueError("development evaluation incomplete")
    return {
        "schema_version": "shadowtrainer.stunet-scaled-training-audit/v1",
        "status": "complete_integrity_only",
        "plan_sha256": digest,
        "last_checkpoint_sha256": checkpoint_sha,
        "train_cases": count,
        "confirmed_updates": expected,
        "finite_training_losses": True,
        "epoch_development": [{
            "epoch": row["epoch"], "global_step": row["global_step"],
            "kidney_mean": row["development"]["kidney_mean"],
            "tumor_mean": row["development"]["tumor_mean"],
            "eligible": row["eligible"],
        } for row in epoch_rows],
        "limitations": ["integrity and development only; no independent efficacy test",
                        "sync staging; no long prefetch equivalence claim"],
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
