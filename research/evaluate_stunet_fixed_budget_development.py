#!/usr/bin/env python3
"""Bounded, paired development evaluation for the equal-update 64/96 arms.

Uses the four previously selected development patients only. The 96-case
latest checkpoint remains diagnostic; this does not unlock held-out data.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from prediction_provenance import binding, verify
from run_stunet_campaign import GIB, HERE, LAB, MIB, run_guarded, snapshot
from stunet_campaign import atomic_json, sha256_file

TRAIN = {
    "64": HERE / "workspace/stunet-fixed-budget64-a-v1",
    "96": HERE / "workspace/stunet-train96-a-v1",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def plan_for(trains: dict[str, Path]) -> dict:
    receipts = {name: read(root / "supervisor.json") for name, root in trains.items()}
    cohorts = {name: read(root / "cohort.json") for name, root in trains.items()}
    configs = {name: read(root / "resolved-config.json") for name, root in trains.items()}
    a, b = cohorts["64"], cohorts["96"]
    if (any(receipts[name]["status"] != "success" or
            receipts[name]["confirmed_updates"] != 1536 for name in trains) or
            a["train_cases"] != b["train_cases"][:64] or
            a["development_cases"] != b["development_cases"] or
            len(a["development_cases"]) != 4 or
            set(a["development_cases"]) & set(b["train_cases"]) or
            configs["64"]["training_seed"] != configs["96"]["training_seed"] or
            a["base_checkpoint_sha256"] != b["base_checkpoint_sha256"]):
        raise ValueError("equal-update development protocol mismatch")
    checkpoints = {name: root / "arm_A" /
                   ("best.checkpoint.pt" if name == "64" else "latest.checkpoint.pt")
                   for name, root in trains.items()}
    for name, checkpoint in checkpoints.items():
        expected = (receipts[name]["best_checkpoint_sha256"] if name == "64"
                    else receipts[name]["last_checkpoint_sha256"])
        if sha256_file(checkpoint) != expected:
            raise ValueError(f"{name} checkpoint changed")
    for case in a["development_cases"]:
        for filename in ("imaging.nii.gz", "segmentation.nii.gz"):
            left = Path(configs["64"]["training_cache"]) / case / filename
            right = Path(configs["96"]["training_cache"]) / case / filename
            if sha256_file(left) != sha256_file(right):
                raise ValueError(f"development source mismatch: {case}/{filename}")
    return {
        "schema_version": "shadowtrainer.stunet-fixed-budget-development-plan/v1",
        "status": "development_only",
        "cases": a["development_cases"],
        "updates_per_arm": 1536,
        "train_plan_sha256": {name: receipts[name]["plan_sha256"] for name in trains},
        "checkpoint_sha256": {name: sha256_file(checkpoints[name]) for name in trains},
        "checkpoint_path": {name: str(checkpoints[name]) for name in trains},
        "limitations": ["four reused development cases", "96 latest is not a selected model",
                        "post-hoc contrast, not independent clinical validation"],
    }


def compare(output: Path, trains: dict[str, Path], plan: dict) -> dict:
    summaries = {name: read(output / name / "evaluation/development/A/summary.json")
                 for name in trains}
    a, b = summaries["64"], summaries["96"]
    if (a["status"] != b["status"] or a["status"] != "success" or
            a["protocol"] != b["protocol"] or
            a["cases_requested"] != b["cases_requested"] or
            a["cases_requested"] != plan["cases"] or
            not (len(a["cases"]) == len(b["cases"]) == 4)):
        raise ValueError("development evaluations not comparable")
    for name, summary in summaries.items():
        if summary["checkpoint_sha256"] != plan["checkpoint_sha256"][name]:
            raise ValueError("checkpoint provenance changed")
        cache = Path(read(trains[name] / "resolved-config.json")["training_cache"])
        for case, row in zip(plan["cases"], summary["cases"]):
            if row["case_id"] != case:
                raise ValueError("development case order changed")
            prediction = output / name / "evaluation/development/A/predictions" / f"{case}.nii.gz"
            if sha256_file(prediction) != row["prediction_sha256"]:
                raise ValueError("prediction hash mismatch")
            verify(prediction, binding(summary["checkpoint_sha256"],
                                       cache / case / "imaging.nii.gz",
                                       cache / case / "segmentation.nii.gz",
                                       summary["protocol"]))
    metrics = {}
    for group, keys in (("classes", ("1", "2", "3")),
                        ("hec", ("kidney_and_masses", "kidney_mass", "tumor"))):
        metrics[group] = {}
        for key in keys:
            left = [row[group][key]["dice"] for row in a["cases"]]
            right = [row[group][key]["dice"] for row in b["cases"]]
            pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
            if any(not math.isfinite(x) or not math.isfinite(y) for x, y in pairs):
                raise ValueError("nonfinite Dice")
            metrics[group][key] = {
                "valid_pairs": len(pairs),
                "mean_64": statistics.mean(x for x, _ in pairs) if pairs else None,
                "mean_96": statistics.mean(y for _, y in pairs) if pairs else None,
                "mean_delta_64_minus_96": statistics.mean(x-y for x, y in pairs) if pairs else None,
                "better_64": sum(x > y for x, y in pairs),
                "better_96": sum(y > x for x, y in pairs),
                "ties": sum(x == y for x, y in pairs),
            }
    return {
        "schema_version": "shadowtrainer.stunet-fixed-budget-development-audit/v1",
        "status": "complete_development_only", "case_count": 4,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "metrics": metrics, "limitations": plan["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "workspace/stunet-fixed-budget-development-r1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    trains = {name: root.resolve() for name, root in TRAIN.items()}
    state = snapshot()
    issues = []
    if state["disk_free_bytes"] < 20 * GIB + 512 * MIB:
        issues.append("physical_disk_reservation")
    if state["available_ram_bytes"] < int(1.5 * GIB):
        issues.append("available_ram_floor")
    if state["swap_used_bytes"] > 192 * MIB:
        issues.append("swap_limit")
    elif state["swap_used_bytes"] > (192 - 48) * MIB:
        # The first case increased host swap by about 41 MiB before the
        # unchanged 192-MiB run guard stopped the session. Reserve headroom.
        issues.append("swap_headroom_below_48_mib")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], text=True,
                         capture_output=True, timeout=10)
    if gpu.returncode or int(gpu.stdout.splitlines()[0]) < 2500:
        issues.append("gpu_free_below_2500_mib")
    plan = None
    if not issues:
        try:
            plan = plan_for(trains)
            if output.exists() and read(output / "plan.json") != plan:
                issues.append("existing_plan_mismatch")
        except Exception as exc:
            issues.append(type(exc).__name__ + ": " + str(exc))
    if args.preflight_only or issues:
        print(json.dumps({"status": "pass" if not issues else "fail", "issues": issues,
                          "free_disk_gib": round(state["disk_free_bytes"] / GIB, 2)}))
        return 0 if not issues else 2
    with (LAB / ".stage23_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not output.exists():
            output.mkdir(parents=True)
            atomic_json(output / "plan.json", plan)
            for name in trains:
                arm = output / name
                arm.mkdir()
                atomic_json(arm / "cohort.json", read(trains[name] / "cohort.json"))
                atomic_json(arm / "resolved-config.json",
                            read(trains[name] / "resolved-config.json"))
        for name in trains:
            arm = output / name
            summary = arm / "evaluation/development/A/summary.json"
            if summary.exists() and read(summary).get("status") == "success":
                continue
            config = read(arm / "resolved-config.json")
            command = [config["python"], str(HERE / "stunet_campaign_evaluate.py"),
                       "--config", str(arm / "resolved-config.json"),
                       "--session", str(arm), "--model", "A", "--development",
                       "--checkpoint", plan["checkpoint_path"][name],
                       "--surface-voxel-limit", "0"]
            deadline = time.time() + 30 * 60
            phase = run_guarded(command, LAB, arm / "eval_A.log",
                                arm / "eval_A.resources.jsonl", deadline, deadline,
                                swap_limit=192 * MIB)
            atomic_json(arm / "phase.json", phase)
            if phase["status"] != "success":
                print(json.dumps({"status": phase["status"], "arm": name,
                                  "reasons": phase["reasons"]}))
                return 2
        result = compare(output, trains, plan)
        atomic_json(output / "paired-audit.json", result)
        print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
