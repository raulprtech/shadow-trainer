#!/usr/bin/env python3
"""Bounded, resumable STU-Net A training on the frozen 64-case KiTS train split.

This is a development experiment. It never evaluates the reused six-case final
cohort and never claims that 64 cases are all of KiTS23.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

from run_stunet_campaign import GIB, LAB, HERE, failures, preflight, run_guarded, snapshot
from stunet_campaign import atomic_json, csv_case_ids, manifest_records, record_bytes, sha256_file


def make_plan(config: dict, epochs: int = 2) -> dict:
    if epochs != 2:
        raise ValueError("train64 protocol fixes two epochs")
    records = manifest_records(Path(config["manifest"]))
    train = csv_case_ids(Path(config["train_csv"]))
    stage34 = json.loads(Path(config["stage34_schedule"]).read_text())
    development = list(stage34["validation_cases"])
    val = set(csv_case_ids(Path(config["validation_csv"])))
    if len(train) != 64 or len(set(train)) != 64 or len(development) != 4:
        raise ValueError("unexpected frozen cohort size or duplicates")
    if not set(stage34["train_cases"]).issubset(train):
        raise ValueError("Stage34 training cases absent from 64-case split")
    if set(train) & val or not set(development).issubset(val):
        raise ValueError("train/validation leakage")
    missing = (set(train) | set(development)) - set(records)
    if missing:
        raise ValueError(f"cases absent from manifest: {len(missing)}")
    budget = int(config["training_cache_bytes"])
    pinned = sum(record_bytes(records[case]) for case in development)
    largest = max(record_bytes(records[case]) for case in train)
    if pinned + largest > budget:
        raise ValueError("development pin plus largest train case exceeds cache budget")
    return {
        "schema_version": "shadowtrainer.stunet-train64-plan/v1",
        "purpose": "development_training_only",
        "arm": "A",
        "loss": "stage20",
        "epochs": epochs,
        "patches_per_case": int(config["patches_per_case"]),
        "expected_updates": len(train) * epochs * int(config["patches_per_case"]),
        "train_cases": train,
        "development_cases": development,
        "evaluation_cases": [],
        "selected_cases_are_entire_frozen_train_csv": True,
        "max_pinned_plus_case_bytes": pinned + largest,
        "cache_budget_bytes": budget,
        "manifest_sha256": sha256_file(Path(config["manifest"])),
        "train_csv_sha256": sha256_file(Path(config["train_csv"])),
        "validation_csv_sha256": sha256_file(Path(config["validation_csv"])),
        "stage34_schedule_sha256": sha256_file(Path(config["stage34_schedule"])),
        "base_checkpoint_sha256": config["base_checkpoint_sha256"],
        "limitations": [
            "64 cases are the frozen local training split, not all KiTS23",
            "four development cases may select the best checkpoint",
            "no independent final or clinical validation",
            "sync staging only; long prefetch equivalence remains pending",
        ],
    }


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def confirmed_updates(session: Path) -> int:
    """Count only fsynced case boundaries, including interrupted runs."""
    metrics = session / "arm_A/metrics.jsonl"
    if not metrics.exists():
        return 0
    step = 0
    with metrics.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A non-fsynced trailing fragment is not a commit.
            if event.get("kind") == "case_boundary":
                step = max(step, int(event["global_step"]))
    return step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "stunet_campaign_config.json")
    parser.add_argument("--output", type=Path, default=HERE / "workspace/stunet-train64-a-v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=6.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 8:
        raise ValueError("max-hours must be in (0, 8]")
    config = json.loads(args.config.read_text())
    config["training_cache"] = str(args.output.resolve() / "training_cache")
    config["training_seed"] = 20260921
    plan = make_plan(config)
    digest = plan_digest(plan)
    existing = args.output.exists()
    check = preflight(config, require_remote=True, startup=not existing)
    check["plan_sha256"] = digest
    check["train_cases"] = len(plan["train_cases"])
    check["development_cases"] = len(plan["development_cases"])
    check["expected_updates"] = plan["expected_updates"]
    check["output_exists"] = existing
    if not existing and check["status"] == "pass":
        try:
            gpu_line = check["gpu"]["stdout"].splitlines()[0]
            free_mib = int(gpu_line.rsplit(",", 1)[-1].strip())
            if free_mib < 2500:
                check["issues"].append("gpu_free_below_2500_mib")
        except (IndexError, ValueError):
            check["issues"].append("gpu_free_unreadable")
    if check["issues"]:
        check["status"] = "fail"
    if args.preflight_only:
        # Do not print remote paths, tokens or patient identifiers.
        print(json.dumps({key: check[key] for key in
                          ("status", "issues", "resources", "train_cases",
                           "development_cases", "expected_updates",
                           "output_exists", "plan_sha256")}, indent=2))
        return 0 if check["status"] == "pass" else 2
    if check["status"] != "pass":
        raise RuntimeError(f"preflight failed: {check['issues']}")
    with (LAB / ".stage23_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = args.output.resolve()
        if not existing:
            session.mkdir(parents=True)
            atomic_json(session / "cohort.json", plan)
            atomic_json(session / "resolved-config.json", config)
            atomic_json(session / "plan.json", {"plan": plan, "sha256": digest})
        else:
            frozen = json.loads((session / "plan.json").read_text())
            if frozen != {"plan": plan, "sha256": digest}:
                raise RuntimeError("resume plan mismatch")
            if json.loads((session / "resolved-config.json").read_text()) != config:
                raise RuntimeError("resume config mismatch")
        prior = json.loads((session / "supervisor.json").read_text()) if (
            session / "supervisor.json").exists() else {}
        prior_seconds = float(prior.get("active_seconds", 0.0))
        if prior.get("status") == "success":
            print(json.dumps({"status": "already_complete", "session": str(session)}))
            return 0
        if prior_seconds >= args.max_hours * 3600:
            raise RuntimeError("max active hours already exhausted; raise explicit bound to resume")
        start = time.time()
        deadline = start + args.max_hours * 3600 - prior_seconds
        result = run_guarded(
            [config["python"], str(HERE / "stunet_campaign_worker.py"),
             "--config", str(session / "resolved-config.json"),
             "--session", str(session), "--arm", "A", "--epochs", "2"],
            LAB, session / "train_A.log", session / "train_A.resources.jsonl",
            deadline, deadline,
        )
        active = prior_seconds + time.time() - start
        summary_path = session / "arm_A/summary.json"
        arm = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        confirmed = confirmed_updates(session)
        status = "success" if (result["status"] == "success" and
                               arm.get("status") == "success" and
                               arm.get("global_step") == plan["expected_updates"] and
                               confirmed == plan["expected_updates"]) else result["status"]
        receipt = {
            "schema_version": "shadowtrainer.stunet-train64-supervisor/v1",
            "status": status, "plan_sha256": digest,
            "active_seconds": active, "expected_updates": plan["expected_updates"],
            "confirmed_updates": confirmed,
            "phase": result, "resources_at_close": snapshot(),
            "best_checkpoint_sha256": sha256_file(session / "arm_A/best.checkpoint.pt")
                if (session / "arm_A/best.checkpoint.pt").exists() else None,
            "last_checkpoint_sha256": sha256_file(session / "arm_A/latest.checkpoint.pt")
                if (session / "arm_A/latest.checkpoint.pt").exists() else None,
        }
        atomic_json(session / "supervisor.json", receipt)
        print(json.dumps({key: receipt[key] for key in
                          ("status", "active_seconds", "expected_updates",
                           "confirmed_updates", "plan_sha256")}, indent=2))
        return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
