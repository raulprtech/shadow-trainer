#!/usr/bin/env python3
"""Post-hoc fixed-update 64-case comparator for the completed 96-case run.

Both arms start from Stage20 with seed 20260922 and receive 1536 updates:
64 cases x 3 epochs x 8 patches versus 96 cases x 2 epochs x 8 patches.
The six previously opened cases remain exploratory, not an independent test.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from run_stunet_campaign import LAB, HERE, preflight, run_guarded, snapshot
from run_stunet_train64 import confirmed_updates, plan_digest
from run_stunet_train_scaled import make_plan as make_scaled_plan
from stunet_campaign import atomic_json, sha256_file


def make_plan(config: dict) -> dict:
    scaled = make_scaled_plan(config, 96)
    train = scaled["train_cases"][:64]
    if train != scaled["train_cases"][:scaled["original_train_cases"]]:
        raise ValueError("64-case prefix changed")
    plan = dict(scaled)
    plan.update({
        "schema_version": "shadowtrainer.stunet-fixed-budget64-plan/v1",
        "purpose": "post_hoc_equal_update_data_scale_comparison",
        "epochs": 3,
        "expected_updates": 3 * len(train) * int(config["patches_per_case"]),
        "train_cases": train,
        "additional_cases": [],
        "reference_96_plan_sha256": plan_digest(scaled),
        "limitations": [
            "post-hoc comparator: the 96-case outcome was inspected before this arm was designed",
            "equal updates do not equalize case exposure: 64 cases repeat three times, 96 twice",
            "the six previously opened evaluation cases are exploratory, not independent",
            "only the four development cases may select a checkpoint",
            "sync staging only; the long prefetch equivalence gate is pending",
            "no clinical or prognostic claim",
        ],
    })
    if plan["expected_updates"] != scaled["expected_updates"]:
        raise ValueError("update budget differs from completed 96-case arm")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "stunet_campaign_config.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "workspace/stunet-fixed-budget64-a-v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 12:
        raise ValueError("max-hours must be in (0, 12]")
    session = args.output.resolve()
    config = json.loads(args.config.read_text())
    config["training_cache"] = str(session / "training_cache")
    config["training_seed"] = 20260922
    plan = make_plan(config)
    digest = plan_digest(plan)
    reference = HERE / "workspace/stunet-train96-a-v1"
    reference_plan = json.loads((reference / "plan.json").read_text())
    reference_receipt = json.loads((reference / "supervisor.json").read_text())
    if (reference_plan["sha256"] != plan["reference_96_plan_sha256"] or
            reference_receipt["status"] != "success" or
            reference_receipt["confirmed_updates"] != plan["expected_updates"]):
        raise RuntimeError("96-case reference provenance or update budget changed")
    existing = session.exists()
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
            "resources": check["resources"], "cases": len(plan["train_cases"]),
            "epochs": plan["epochs"], "expected_updates": plan["expected_updates"],
            "output_exists": existing, "plan_sha256": digest,
            "reference_96_plan_sha256": plan["reference_96_plan_sha256"],
        }, indent=2))
        return 0 if check["status"] == "pass" else 2
    if check["status"] != "pass":
        raise RuntimeError(f"preflight failed: {check['issues']}")
    with (LAB / ".stage23_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
             "--session", str(session), "--arm", "A", "--epochs", "3"],
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
            "schema_version": "shadowtrainer.stunet-fixed-budget64-supervisor/v1",
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
