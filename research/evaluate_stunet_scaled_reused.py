#!/usr/bin/env python3
"""Compare a complete scaled STU-Net run on six already-opened KiTS cases.

This is exploratory: these six cases were opened for the 64-case result. Never
overwrite its sealed predictions or call this a fresh independent evaluation.
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

SOURCE = HERE / "workspace/stunet-train64-heldout-r1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def frozen_plan(train: Path, source: Path, candidate: Path) -> dict:
    supervisor = read(train / "supervisor.json")
    cohort = read(train / "cohort.json")
    config = read(train / "resolved-config.json")
    selection = read(source / "selection.json")
    old = read(source / "evaluation/evaluation/A/summary.json")
    cases = selection["evaluation_cases"]
    if (supervisor["status"] != "success" or
            supervisor["confirmed_updates"] != supervisor["expected_updates"] or
            supervisor["expected_updates"] != cohort["expected_updates"]):
        raise ValueError("scaled training is incomplete")
    if sha256_file(candidate) != supervisor["last_checkpoint_sha256"]:
        raise ValueError("scaled latest checkpoint changed")
    if (len(cases) != 6 or len(set(cases)) != 6 or
            set(cases) & (set(cohort["train_cases"]) | set(cohort["development_cases"]))):
        raise ValueError("cohort leakage or size mismatch")
    if (old["status"] != "success" or old["cases_requested"] != cases or
            len(old["cases"]) != 6 or
            old["checkpoint_sha256"] != selection["candidate_checkpoint_sha256"]):
        raise ValueError("64-case reference mismatch")
    if (sha256_file(Path(config["manifest"])) != cohort["manifest_sha256"] or
            sha256_file(Path(config["validation_csv"])) != cohort["validation_csv_sha256"]):
        raise ValueError("source manifest or validation split changed")
    cache = source / "evaluation_cache"
    for case in cases:
        for name in ("imaging.nii.gz", "segmentation.nii.gz"):
            if not (cache / case / name).is_file():
                raise FileNotFoundError(cache / case / name)
    return {
        "schema_version": "shadowtrainer.stunet-scaled-reused-plan/v1",
        "status": "exploratory_reused_six",
        "cases": cases,
        "training_plan_sha256": supervisor["plan_sha256"],
        "scaled_checkpoint_sha256": supervisor["last_checkpoint_sha256"],
        "reference_checkpoint_sha256": old["checkpoint_sha256"],
        "source_selection_sha256": sha256_file(source / "selection.json"),
        "source_summary_sha256": sha256_file(source / "evaluation/evaluation/A/summary.json"),
        "limitation": "already-opened six cases; 64 and scaled have different numbers of updates",
    }


def seal_session(output: Path, train: Path, source: Path, plan: dict) -> None:
    if output.exists():
        if read(output / "plan.json") != plan:
            raise ValueError("existing evaluation plan differs")
        return
    output.mkdir(parents=True)
    cohort = read(train / "cohort.json")
    cohort["evaluation_cases"] = plan["cases"]
    config = read(train / "resolved-config.json")
    config["evaluation_cache"] = str((source / "evaluation_cache").resolve())
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "cohort.json", cohort)
    atomic_json(output / "resolved-config.json", config)


def audit(output: Path, source: Path, plan: dict) -> dict:
    old = read(source / "evaluation/evaluation/A/summary.json")
    new = read(output / "evaluation/evaluation/A/summary.json")
    cases = plan["cases"]
    if (old["status"] != "success" or new["status"] != "success" or
            old["protocol"] != new["protocol"] or
            old["cases_requested"] != cases or new["cases_requested"] != cases or
            old["checkpoint_sha256"] != plan["reference_checkpoint_sha256"] or
            new["checkpoint_sha256"] != plan["scaled_checkpoint_sha256"] or
            sha256_file(source / "evaluation/evaluation/A/summary.json") !=
            plan["source_summary_sha256"] or
            len(old["cases"]) != 6 or len(new["cases"]) != 6):
        raise ValueError("paired evaluation provenance mismatch")
    pairs = []
    cache = source / "evaluation_cache"
    for case, left, right in zip(cases, old["cases"], new["cases"]):
        if left["case_id"] != case or right["case_id"] != case:
            raise ValueError("case order mismatch")
        image = cache / case / "imaging.nii.gz"
        label = cache / case / "segmentation.nii.gz"
        for row, root, digest in (
            (left, source, old["checkpoint_sha256"]),
            (right, output, new["checkpoint_sha256"]),
        ):
            prediction = root / "evaluation/evaluation/A/predictions" / f"{case}.nii.gz"
            if sha256_file(prediction) != row["prediction_sha256"]:
                raise ValueError("prediction digest mismatch")
            verify(prediction, binding(digest, image, label, old["protocol"]))
        metrics = {}
        for key in ("1", "2", "3"):
            baseline = left["classes"][key]["dice"]
            scaled = right["classes"][key]["dice"]
            if any(x is not None and (not math.isfinite(x) or not 0 <= x <= 1)
                   for x in (baseline, scaled)):
                raise ValueError("invalid Dice")
            metrics[key] = {"dice_64": baseline, "dice_scaled": scaled,
                            "delta": scaled-baseline if baseline is not None and scaled is not None else None}
        pairs.append({"alias": f"P{len(pairs)+1}", "metrics": metrics})
    aggregate = {}
    for key, name in (("1", "kidney"), ("2", "tumor"), ("3", "cyst")):
        left = [p["metrics"][key]["dice_64"] for p in pairs]
        right = [p["metrics"][key]["dice_scaled"] for p in pairs]
        differences = [p["metrics"][key]["delta"] for p in pairs]
        valid = [x for x in differences if x is not None]
        aggregate[name] = {
            "valid_pairs": len(valid),
            "dice_64_mean": statistics.mean(x for x in left if x is not None) if any(x is not None for x in left) else None,
            "dice_scaled_mean": statistics.mean(x for x in right if x is not None) if any(x is not None for x in right) else None,
            "mean_delta": statistics.mean(valid) if valid else None,
            "median_delta": statistics.median(valid) if valid else None,
            "improved": sum(x > 0 for x in valid),
            "worsened": sum(x < 0 for x in valid),
            "unchanged": sum(x == 0 for x in valid),
        }
    return {
        "schema_version": "shadowtrainer.stunet-scaled-reused-audit/v1",
        "status": "complete_exploratory_reused_six",
        "case_count": 6, "metrics": aggregate, "paired_cases": pairs,
        "checkpoint_sha256": {"64": old["checkpoint_sha256"],
                                  "scaled": new["checkpoint_sha256"]},
        "limitations": ["six cases previously opened for the 64-case experiment",
                        "training cases and update counts both changed",
                        "not independent clinical or prognostic validation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-session", type=Path,
                        default=HERE / "workspace/stunet-train96-a-v1")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path,
                        default=HERE / "workspace/stunet-train96-reused-six-r1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train, source, output = (path.resolve() for path in
                             (args.train_session, args.source, args.output))
    candidate = train / "arm_A/latest.checkpoint.pt"
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
            plan = frozen_plan(train, source, candidate)
            if output.exists() and read(output / "plan.json") != plan:
                issues.append("existing_plan_mismatch")
        except Exception as exc:
            issues.append(type(exc).__name__ + ": " + str(exc))
    if args.preflight_only or issues:
        print(json.dumps({"status": "pass" if not issues else "fail", "issues": issues,
                          "free_disk_gib": round(state["disk_free_bytes"] / GIB, 2)}))
        return 0 if not issues else 2
    seal_session(output, train, source, plan)
    config = read(output / "resolved-config.json")
    command = [config["python"], str(HERE / "stunet_campaign_evaluate.py"),
               "--config", str(output / "resolved-config.json"),
               "--session", str(output), "--model", "A", "--checkpoint",
               str(candidate), "--surface-voxel-limit", "0"]
    deadline = time.time() + 45 * 60
    phase = run_guarded(command, LAB, output / "eval_A.log",
                        output / "eval_A.resources.jsonl", deadline, deadline,
                        swap_limit=192 * MIB)
    atomic_json(output / "phase.json", phase)
    if phase["status"] != "success":
        print(json.dumps({"status": phase["status"], "reasons": phase["reasons"]}))
        return 2
    result = audit(output, source, plan)
    atomic_json(output / "paired-audit.json", result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
