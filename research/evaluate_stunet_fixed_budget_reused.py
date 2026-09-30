#!/usr/bin/env python3
"""Post-hoc volumetric contrast of equal-update 64/96 arms on six reused cases.

These cases were opened before this comparison. The result is diagnostic, not
an independent test, and must not select a checkpoint or support clinical use.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from prediction_provenance import binding, verify
from run_stunet_campaign import GIB, HERE, LAB, MIB, run_guarded, snapshot
from stunet_campaign import atomic_json, sha256_file

TRAIN64 = HERE / "workspace/stunet-fixed-budget64-a-v1"
TRAIN96 = HERE / "workspace/stunet-train96-a-v1"
SOURCE96 = HERE / "workspace/stunet-train96-reused-six-r1"
ORIGIN = HERE / "workspace/stunet-train64-heldout-r1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def frozen_plan(train64: Path, train96: Path, source96: Path, origin: Path) -> dict:
    a = read(train64 / "supervisor.json")
    b = read(train96 / "supervisor.json")
    c64 = read(train64 / "cohort.json")
    c96 = read(train96 / "cohort.json")
    cfg64 = read(train64 / "resolved-config.json")
    cfg96 = read(train96 / "resolved-config.json")
    old = read(source96 / "evaluation/evaluation/A/summary.json")
    cases = read(origin / "selection.json")["evaluation_cases"]
    if (a["status"] != "success" or b["status"] != "success" or
            not (a["confirmed_updates"] == b["confirmed_updates"] == 1536)):
        raise ValueError("equal-update training not complete")
    if (len(c64["train_cases"]) != 64 or len(c96["train_cases"]) != 96 or
            c64["train_cases"] != c96["train_cases"][:64] or
            c64["epochs"] != 3 or c96["epochs"] != 2 or
            cfg64["training_seed"] != cfg96["training_seed"] or
            c64["base_checkpoint_sha256"] != c96["base_checkpoint_sha256"] or
            c64["reference_96_plan_sha256"] != b["plan_sha256"]):
        raise ValueError("64/96 protocol mismatch")
    if (len(cases) != 6 or len(set(cases)) != 6 or
            set(cases) & (set(c96["train_cases"]) | set(c96["development_cases"]))):
        raise ValueError("reused evaluation cohort mismatch or leakage")
    candidate = train64 / "arm_A/latest.checkpoint.pt"
    if (sha256_file(candidate) != a["last_checkpoint_sha256"] or
            old["status"] != "success" or old["cases_requested"] != cases or
            old["checkpoint_sha256"] != b["last_checkpoint_sha256"] or
            len(old["cases"]) != 6):
        raise ValueError("checkpoint or 96-case evaluation provenance changed")
    cache = origin / "evaluation_cache"
    for case in cases:
        for name in ("imaging.nii.gz", "segmentation.nii.gz"):
            if not (cache / case / name).is_file():
                raise FileNotFoundError(cache / case / name)
    return {
        "schema_version": "shadowtrainer.stunet-equal-updates-reused-plan/v1",
        "status": "post_hoc_exploratory",
        "cases": cases,
        "train64_plan_sha256": a["plan_sha256"],
        "train96_plan_sha256": b["plan_sha256"],
        "updates_per_arm": 1536,
        "checkpoint64_sha256": a["last_checkpoint_sha256"],
        "checkpoint96_sha256": b["last_checkpoint_sha256"],
        "source96_summary_sha256": sha256_file(source96 / "evaluation/evaluation/A/summary.json"),
        "origin_selection_sha256": sha256_file(origin / "selection.json"),
        "limitations": [
            "six cases previously opened; not independent",
            "comparison designed after observing 96-case outcome",
            "equal updates but unequal patient exposure",
            "development alone selects checkpoints; latest checkpoints here are diagnostic",
        ],
    }


def seal(output: Path, train64: Path, origin: Path, plan: dict) -> None:
    if output.exists():
        if read(output / "plan.json") != plan:
            raise ValueError("existing evaluation plan differs")
        return
    output.mkdir(parents=True)
    cohort = read(train64 / "cohort.json")
    cohort["evaluation_cases"] = plan["cases"]
    config = read(train64 / "resolved-config.json")
    config["evaluation_cache"] = str((origin / "evaluation_cache").resolve())
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "cohort.json", cohort)
    atomic_json(output / "resolved-config.json", config)


def audit(output: Path, source96: Path, origin: Path, plan: dict) -> dict:
    old = read(source96 / "evaluation/evaluation/A/summary.json")
    new = read(output / "evaluation/evaluation/A/summary.json")
    cases = plan["cases"]
    if (old["status"] != "success" or new["status"] != "success" or
            old["protocol"] != new["protocol"] or
            old["cases_requested"] != cases or new["cases_requested"] != cases or
            old["checkpoint_sha256"] != plan["checkpoint96_sha256"] or
            new["checkpoint_sha256"] != plan["checkpoint64_sha256"] or
            sha256_file(source96 / "evaluation/evaluation/A/summary.json") !=
            plan["source96_summary_sha256"] or
            len(old["cases"]) != 6 or len(new["cases"]) != 6):
        raise ValueError("paired evaluation provenance mismatch")
    pairs = []
    cache = origin / "evaluation_cache"
    for case, row96, row64 in zip(cases, old["cases"], new["cases"]):
        if row96["case_id"] != case or row64["case_id"] != case:
            raise ValueError("case order mismatch")
        image = cache / case / "imaging.nii.gz"
        label = cache / case / "segmentation.nii.gz"
        for row, root, digest in (
            (row96, source96, old["checkpoint_sha256"]),
            (row64, output, new["checkpoint_sha256"]),
        ):
            prediction = root / "evaluation/evaluation/A/predictions" / f"{case}.nii.gz"
            if sha256_file(prediction) != row["prediction_sha256"]:
                raise ValueError("prediction digest mismatch")
            verify(prediction, binding(digest, image, label, old["protocol"]))
        metrics = {}
        for key in ("1", "2", "3"):
            d64 = row64["classes"][key]["dice"]
            d96 = row96["classes"][key]["dice"]
            if any(x is not None and (not math.isfinite(x) or not 0 <= x <= 1)
                   for x in (d64, d96)):
                raise ValueError("invalid Dice")
            metrics[key] = {"dice_64": d64, "dice_96": d96,
                            "delta_96_minus_64": d96-d64 if d64 is not None and d96 is not None else None}
        pairs.append({"alias": f"P{len(pairs)+1}", "metrics": metrics})
    aggregate = {}
    for key, name in (("1", "kidney"), ("2", "tumor"), ("3", "cyst")):
        left = [p["metrics"][key]["dice_64"] for p in pairs]
        right = [p["metrics"][key]["dice_96"] for p in pairs]
        valid = [p["metrics"][key]["delta_96_minus_64"] for p in pairs
                 if p["metrics"][key]["delta_96_minus_64"] is not None]
        aggregate[name] = {
            "valid_pairs": len(valid),
            "dice_64_mean": statistics.mean(x for x in left if x is not None) if any(x is not None for x in left) else None,
            "dice_96_mean": statistics.mean(x for x in right if x is not None) if any(x is not None for x in right) else None,
            "mean_delta_96_minus_64": statistics.mean(valid) if valid else None,
            "median_delta_96_minus_64": statistics.median(valid) if valid else None,
            "improved_96": sum(x > 0 for x in valid),
            "worsened_96": sum(x < 0 for x in valid),
            "unchanged": sum(x == 0 for x in valid),
        }
    return {
        "schema_version": "shadowtrainer.stunet-equal-updates-reused-audit/v1",
        "status": "complete_post_hoc_exploratory",
        "case_count": 6, "updates_per_arm": 1536,
        "metrics": aggregate, "paired_cases": pairs,
        "checkpoint_sha256": {"64": new["checkpoint_sha256"],
                                  "96": old["checkpoint_sha256"]},
        "limitations": plan["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train64", type=Path, default=TRAIN64)
    parser.add_argument("--train96", type=Path, default=TRAIN96)
    parser.add_argument("--source96", type=Path, default=SOURCE96)
    parser.add_argument("--origin", type=Path, default=ORIGIN)
    parser.add_argument("--output", type=Path,
                        default=HERE / "workspace/stunet-fixed-budget64-reused-six-r1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train64, train96, source96, origin, output = (
        x.resolve() for x in (args.train64, args.train96, args.source96,
                              args.origin, args.output))
    state = snapshot()
    issues = []
    if state["disk_free_bytes"] < 20 * GIB + 512 * MIB:
        issues.append("physical_disk_reservation")
    if state["available_ram_bytes"] < int(1.5 * GIB):
        issues.append("available_ram_floor")
    if state["swap_used_bytes"] > 192 * MIB:
        issues.append("swap_limit")
    plan = None
    if not issues:
        try:
            plan = frozen_plan(train64, train96, source96, origin)
            if output.exists() and read(output / "plan.json") != plan:
                issues.append("existing_plan_mismatch")
        except Exception as exc:
            issues.append(type(exc).__name__ + ": " + str(exc))
    if args.preflight_only or issues:
        print(json.dumps({"status": "pass" if not issues else "fail", "issues": issues,
                          "free_disk_gib": round(state["disk_free_bytes"] / GIB, 2)}))
        return 0 if not issues else 2
    seal(output, train64, origin, plan)
    config = read(output / "resolved-config.json")
    command = [config["python"], str(HERE / "stunet_campaign_evaluate.py"),
               "--config", str(output / "resolved-config.json"),
               "--session", str(output), "--model", "A", "--checkpoint",
               str(train64 / "arm_A/latest.checkpoint.pt"),
               "--surface-voxel-limit", "0"]
    deadline = time.time() + 45 * 60
    phase = run_guarded(command, LAB, output / "eval_A.log",
                        output / "eval_A.resources.jsonl", deadline, deadline,
                        swap_limit=192 * MIB)
    atomic_json(output / "phase.json", phase)
    if phase["status"] != "success":
        print(json.dumps({"status": phase["status"], "reasons": phase["reasons"]}))
        return 2
    result = audit(output, source96, origin, plan)
    atomic_json(output / "paired-audit.json", result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
