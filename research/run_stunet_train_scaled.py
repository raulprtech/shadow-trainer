#!/usr/bin/env python3
"""Bounded KiTS STU-Net development training with a frozen 64+N cohort.

The original 64-case campaign is immutable. Additional cases are selected by
hash from the manifest, excluding the entire local validation CSV. This is an
exploratory data-scale experiment, not an independent clinical evaluation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

from run_stunet_campaign import GIB, LAB, HERE, preflight, run_guarded, snapshot
from run_stunet_train64 import confirmed_updates, plan_digest
from stunet_campaign import atomic_json, csv_case_ids, manifest_records, record_bytes, sha256_file


def make_plan(config: dict, case_count: int = 96) -> dict:
    if case_count < 64:
        raise ValueError("scaled protocol requires at least the frozen 64 cases")
    records = manifest_records(Path(config["manifest"]))
    original = csv_case_ids(Path(config["train_csv"]))
    validation = csv_case_ids(Path(config["validation_csv"]))
    stage34 = json.loads(Path(config["stage34_schedule"]).read_text())
    development = list(stage34["validation_cases"])
    if len(original) != 64 or len(set(original)) != 64:
        raise ValueError("original 64-case split changed")
    if len(validation) != len(set(validation)) or len(development) != 4:
        raise ValueError("validation/development split changed")
    if set(original) & set(validation) or not set(development).issubset(validation):
        raise ValueError("train/validation leakage")
    if not set(stage34["train_cases"]).issubset(original):
        raise ValueError("Stage34 training provenance changed")
    if (set(original) | set(validation)) - set(records):
        raise ValueError("split case absent from manifest")
    budget = int(config["training_cache_bytes"])
    pinned = sum(record_bytes(records[case]) for case in development)
    if pinned + max(record_bytes(records[case]) for case in original) > budget:
        raise ValueError("original case exceeds cache with pinned development")
    excluded = set(original) | set(validation)
    eligible = [case for case in records if case not in excluded
                and pinned + record_bytes(records[case]) <= budget]
    seed = 20260922
    ranked = sorted(eligible, key=lambda case: (hashlib.sha256(
        f"{seed}:{case}".encode()).hexdigest(), case))
    maximum_case_count = len(original) + len(ranked)
    if case_count > maximum_case_count:
        raise ValueError(
            f"requested {case_count} cases but only {maximum_case_count} fit the frozen protocol"
        )
    extra = ranked[:case_count - len(original)]
    if len(extra) != case_count - len(original):
        raise ValueError("insufficient eligible additional cases")
    train = [*original, *extra]
    return {
        "schema_version": "shadowtrainer.stunet-scaled-plan/v1",
        "purpose": "development_training_data_scale_exploratory",
        "arm": "A", "loss": "stage20", "epochs": 2,
        "patches_per_case": int(config["patches_per_case"]),
        "expected_updates": case_count * 2 * int(config["patches_per_case"]),
        "train_cases": train, "original_train_cases": len(original),
        "additional_cases": extra, "selection_seed": seed,
        "selection": "sha256(seed:case_id), ascending; exclude entire validation CSV; fit cache",
        "eligible_additional_count": len(eligible),
        "development_cases": development, "evaluation_cases": [],
        "max_pinned_plus_case_bytes": pinned + max(record_bytes(records[case]) for case in train),
        "cache_budget_bytes": budget,
        "manifest_sha256": sha256_file(Path(config["manifest"])),
        "train_csv_sha256": sha256_file(Path(config["train_csv"])),
        "validation_csv_sha256": sha256_file(Path(config["validation_csv"])),
        "stage34_schedule_sha256": sha256_file(Path(config["stage34_schedule"])),
        "base_checkpoint_sha256": config["base_checkpoint_sha256"],
        "limitations": [
            "comparison with 64 cases is exploratory; total updates and training composition differ",
            "existing six-case heldout has been opened and cannot serve as a new independent test",
            "four development cases alone may select checkpoint",
            "sync staging only; long prefetch equivalence remains pending",
            "no clinical or prognostic claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "stunet_campaign_config.json")
    parser.add_argument(
        "--cases",
        type=int,
        default=96,
        help="Total frozen training cases (for example 96, 128, or the full eligible cohort).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 12:
        raise ValueError("max-hours must be in (0, 12]")
    output = args.output or HERE / f"workspace/stunet-train{args.cases}-a-v1"
    config = json.loads(args.config.read_text())
    config["training_cache"] = str(output.resolve() / "training_cache")
    config["training_seed"] = 20260922
    plan = make_plan(config, args.cases)
    digest = plan_digest(plan)
    existing = output.exists()
    check = preflight(config, require_remote=True, startup=not existing)
    if not existing and check["status"] == "pass":
        try:
            free_mib = int(check["gpu"]["stdout"].splitlines()[0].rsplit(",", 1)[-1].strip())
            if free_mib < 2500:
                check["issues"].append("gpu_free_below_2500_mib")
        except (IndexError, ValueError):
            check["issues"].append("gpu_free_unreadable")
    if check["issues"]:
        check["status"] = "fail"
    if args.preflight_only:
        print(json.dumps({
            "status": check["status"], "issues": check["issues"],
            "resources": check["resources"], "case_count": args.cases,
            "additional_cases": len(plan["additional_cases"]),
            "expected_updates": plan["expected_updates"],
            "output_exists": existing, "plan_sha256": digest,
        }, indent=2))
        return 0 if check["status"] == "pass" else 2
    if check["status"] != "pass":
        raise RuntimeError(f"preflight failed: {check['issues']}")
    with (LAB / ".stage23_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = output.resolve()
        if not existing:
            session.mkdir(parents=True)
            atomic_json(session / "cohort.json", plan)
            atomic_json(session / "resolved-config.json", config)
            atomic_json(session / "plan.json", {"plan": plan, "sha256": digest})
        else:
            if json.loads((session / "plan.json").read_text()) != {"plan": plan, "sha256": digest}:
                raise RuntimeError("resume plan mismatch")
            if json.loads((session / "resolved-config.json").read_text()) != config:
                raise RuntimeError("resume config mismatch")
        prior = json.loads((session / "supervisor.json").read_text()) if (
            session / "supervisor.json").exists() else {}
        if prior.get("status") == "success":
            print(json.dumps({"status": "already_complete", "session": str(session)}))
            return 0
        prior_seconds = float(prior.get("active_seconds", 0.0))
        if prior_seconds >= args.max_hours * 3600:
            raise RuntimeError("max active hours exhausted")
        start = time.time()
        deadline = start + args.max_hours * 3600 - prior_seconds
        result = run_guarded(
            [config["python"], str(HERE / "stunet_campaign_worker.py"),
             "--config", str(session / "resolved-config.json"),
             "--session", str(session), "--arm", "A", "--epochs", "2"],
            LAB, session / "train_A.log", session / "train_A.resources.jsonl",
            deadline, deadline,
            combined_artifact_root=session,
            combined_artifact_ceiling=int(config["campaign_artifact_ceiling_bytes"]),
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
            "schema_version": "shadowtrainer.stunet-scaled-supervisor/v1",
            "status": status, "plan_sha256": digest,
            "active_seconds": active, "expected_updates": plan["expected_updates"],
            "confirmed_updates": confirmed, "phase": result,
            "resources_at_close": snapshot(),
            "best_checkpoint_sha256": sha256_file(session / "arm_A/best.checkpoint.pt")
                if (session / "arm_A/best.checkpoint.pt").exists() else None,
            "last_checkpoint_sha256": sha256_file(session / "arm_A/latest.checkpoint.pt")
                if (session / "arm_A/latest.checkpoint.pt").exists() else None,
        }
        atomic_json(session / "supervisor.json", receipt)
        print(json.dumps({key: receipt[key] for key in
            ("status", "active_seconds", "expected_updates", "confirmed_updates", "plan_sha256")}, indent=2))
        return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
