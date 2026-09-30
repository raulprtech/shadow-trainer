#!/usr/bin/env python3
"""Second-seed, equal-update 64/96-case STU-Net development replication.

The cohort is identical to the completed first-seed comparison. This is a
development robustness check, not independent evaluation or clinical evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

from run_stunet_campaign import GIB, HERE, LAB, preflight, run_guarded, snapshot
from run_stunet_fixed_budget64 import make_plan as make_64_plan
from run_stunet_train64 import confirmed_updates, plan_digest
from run_stunet_train_scaled import make_plan as make_96_plan
from stunet_campaign import atomic_json, sha256_file


SEED = 20260923
ARMS = (("64", 3), ("96", 2))


def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def frozen_arm(config: dict, root: Path, name: str, reference_digest: str | None = None) -> tuple[dict, dict]:
    session = root / f"cases{name}"
    resolved = dict(config)
    resolved["training_seed"] = SEED
    resolved["training_cache"] = str(session / "training_cache")
    plan = make_64_plan(resolved) if name == "64" else make_96_plan(resolved, 96)
    plan["schema_version"] = "shadowtrainer.stunet-fixed-budget-replication-plan/v1"
    plan["purpose"] = "second_seed_equal_update_development_robustness"
    plan["replication_seed"] = SEED
    plan["reference_96_plan_sha256"] = reference_digest if name == "64" else None
    plan["limitations"] = [
        "development robustness check: the first-seed result was known before this replication",
        "same cases and 1536 updates; exposures differ (64x3 versus 96x2)",
        "four development patients may select checkpoints and are not an independent test",
        "previously opened six patients and locked test are excluded from this campaign",
        "sync staging only; long STU-Net prefetch gate remains pending",
        "no clinical, EDA, or universal acceleration claim",
    ]
    return plan, resolved


def prepare(root: Path, config: dict, write: bool) -> tuple[dict, dict]:
    plan96, config96 = frozen_arm(config, root, "96")
    digest96 = plan_digest(plan96)
    plan64, config64 = frozen_arm(config, root, "64", digest96)
    first96 = make_96_plan(config96, 96)
    first64 = make_64_plan(config64)
    if plan96["train_cases"] != first96["train_cases"] or plan64["train_cases"] != first64["train_cases"]:
        raise RuntimeError("cohort identity changed")
    if set(plan96["train_cases"]) & set(plan96["development_cases"]):
        raise RuntimeError("train/development overlap")
    if plan64["expected_updates"] != plan96["expected_updates"] != 1536:
        raise RuntimeError("unequal update budget")
    arms = {"64": (plan64, config64), "96": (plan96, config96)}
    if write:
        for name, (plan, resolved) in arms.items():
            session = root / f"cases{name}"
            if not session.exists():
                session.mkdir(parents=True)
                atomic_json(session / "cohort.json", plan)
                atomic_json(session / "resolved-config.json", resolved)
                atomic_json(session / "plan.json", {"plan": plan, "sha256": plan_digest(plan)})
            elif (json.loads((session / "plan.json").read_text()) != {"plan": plan, "sha256": plan_digest(plan)}
                  or json.loads((session / "resolved-config.json").read_text()) != resolved):
                raise RuntimeError(f"resume provenance mismatch: {name}")
    return arms, {"seed": SEED, "cases64_plan_sha256": plan_digest(plan64),
                  "cases96_plan_sha256": digest96, "expected_updates_per_arm": 1536,
                  "order": ["64", "96"], "combined_artifact_ceiling_bytes": 5 * GIB}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "stunet_campaign_config.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "workspace/stunet-fixed-budget-replication-seed20260923")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 12:
        raise ValueError("max-hours must be in (0,12]")
    root = args.output.resolve()
    config = json.loads(args.config.read_text())
    arms, frozen = prepare(root, config, write=False)
    check = preflight(arms["64"][1], require_remote=True, startup=not root.exists())
    if check["status"] == "pass":
        try:
            free_mib = int(check["gpu"]["stdout"].splitlines()[0].rsplit(",", 1)[-1].strip())
            if free_mib < 2500:
                check["issues"].append("gpu_free_below_2500_mib")
        except (IndexError, ValueError):
            check["issues"].append("gpu_free_unreadable")
    if tree_bytes(root) > 5 * GIB:
        check["issues"].append("combined_artifact_ceiling")
    check["status"] = "fail" if check["issues"] else "pass"
    if args.preflight_only:
        print(json.dumps({"status": check["status"], "issues": check["issues"],
                          "resources": check["resources"], "frozen": frozen,
                          "output_exists": root.exists(), "current_artifact_bytes": tree_bytes(root)}, indent=2))
        return 0 if check["status"] == "pass" else 2
    if check["status"] != "pass":
        raise RuntimeError(f"preflight failed: {check['issues']}")
    with (LAB / ".stage23_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, frozen = prepare(root, config, write=True)
        receipt_path = root / "supervisor.json"
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if prior.get("status") == "success":
            print(json.dumps({"status": "already_complete", "output": str(root)}))
            return 0
        prior_seconds = float(prior.get("active_seconds", 0))
        if prior_seconds >= args.max_hours * 3600:
            raise RuntimeError("active-hour budget exhausted")
        start = time.time()
        deadline = start + args.max_hours * 3600 - prior_seconds
        phases = dict(prior.get("phases", {}))
        for name, epochs in ARMS:
            session = root / f"cases{name}"
            plan = arms[name][0]
            arm_summary = session / "arm_A/summary.json"
            summary = json.loads(arm_summary.read_text()) if arm_summary.exists() else {}
            if summary.get("status") == "success" and confirmed_updates(session) == 1536:
                phases[name] = {"status": "success", "confirmed_updates": 1536,
                                "best_checkpoint_sha256": sha256_file(session / "arm_A/best.checkpoint.pt"),
                                "latest_checkpoint_sha256": sha256_file(session / "arm_A/latest.checkpoint.pt")}
                continue
            if tree_bytes(root) >= 5 * GIB:
                phases[name] = {"status": "combined_artifact_ceiling"}
                break
            before = preflight(arms[name][1], require_remote=True, startup=False)
            if before["issues"]:
                phases[name] = {"status": "preflight_failed", "issues": before["issues"]}
                break
            result = run_guarded(
                [arms[name][1]["python"], str(HERE / "stunet_campaign_worker.py"),
                 "--config", str(session / "resolved-config.json"), "--session", str(session),
                 "--arm", "A", "--epochs", str(epochs)],
                LAB, session / "train_A.log", session / "train_A.resources.jsonl", deadline, deadline,
                combined_artifact_root=root, combined_artifact_ceiling=5 * GIB)
            summary = json.loads(arm_summary.read_text()) if arm_summary.exists() else {}
            confirmed = confirmed_updates(session)
            complete = (result["status"] == "success" and summary.get("status") == "success"
                        and summary.get("global_step") == plan["expected_updates"] == confirmed)
            phases[name] = {"status": "success" if complete else result["status"],
                            "confirmed_updates": confirmed, "worker": result,
                            "best_checkpoint_sha256": sha256_file(session / "arm_A/best.checkpoint.pt")
                                if (session / "arm_A/best.checkpoint.pt").exists() else None,
                            "latest_checkpoint_sha256": sha256_file(session / "arm_A/latest.checkpoint.pt")
                                if (session / "arm_A/latest.checkpoint.pt").exists() else None}
            atomic_json(receipt_path, {"schema_version": "shadowtrainer.stunet-fixed-budget-replication-supervisor/v1",
                                       "status": "running" if complete else phases[name]["status"],
                                       "frozen": frozen, "phases": phases,
                                       "active_seconds": prior_seconds + time.time() - start,
                                       "artifact_bytes": tree_bytes(root), "resources_at_close": snapshot()})
            if not complete:
                break
        status = "success" if all(phases.get(name, {}).get("status") == "success" for name, _ in ARMS) else "incomplete"
        receipt = {"schema_version": "shadowtrainer.stunet-fixed-budget-replication-supervisor/v1",
                   "status": status, "frozen": frozen, "phases": phases,
                   "active_seconds": prior_seconds + time.time() - start,
                   "artifact_bytes": tree_bytes(root), "resources_at_close": snapshot()}
        atomic_json(receipt_path, receipt)
        print(json.dumps({"status": status, "output": str(root), "active_seconds": receipt["active_seconds"],
                          "artifact_bytes": receipt["artifact_bytes"],
                          "phases": {name: phase["status"] for name, phase in phases.items()}}, indent=2))
        return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
