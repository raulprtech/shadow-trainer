#!/usr/bin/env python3
"""Resume the frozen six-case A evaluation and audit the complete B0/A pair.

Never changes the cohort, checkpoint, protocol or resource guards. Incomplete
summaries remain incomplete and no paired result is published in that state.
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

SESSION = HERE / "workspace/stunet-train64-heldout-r1"
CANDIDATE = HERE / "workspace/stunet-train64-a-v1/arm_A/latest.checkpoint.pt"
LIMIT_SWAP = 192 * MIB
MIN_FREE_DISK = 20 * GIB + 512 * MIB


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def frozen_inputs(session: Path, candidate: Path) -> tuple[dict, dict, dict]:
    selection = read_json(session / "selection.json")
    cohort = read_json(session / "cohort.json")
    config = read_json(session / "resolved-config.json")
    cases = selection["evaluation_cases"]
    if (selection["status"] != "frozen_before_labels_or_predictions"
            or len(cases) != 6 or len(set(cases)) != 6
            or cases != cohort["evaluation_cases"]
            or set(cases) & (set(cohort["train_cases"]) | set(cohort["development_cases"]))):
        raise ValueError("frozen evaluation cohort mismatch")
    checks = (
        (Path(config["manifest"]), selection["source_manifest_sha256"]),
        (Path(config["train_csv"]), selection["train_csv_sha256"]),
        (Path(config["validation_csv"]), selection["validation_csv_sha256"]),
        (Path(config["base_checkpoint"]), config["base_checkpoint_sha256"]),
        (candidate, selection["candidate_checkpoint_sha256"]),
        (candidate.parents[1] / "recovery-audit.json", selection["recovery_audit_sha256"]),
    )
    if any(sha256_file(path) != digest for path, digest in checks):
        raise ValueError("frozen source or checkpoint digest mismatch")
    if Path(config["evaluation_cache"]).resolve() != (session / "evaluation_cache").resolve():
        raise ValueError("evaluation cache moved")
    expected_entries = {(case_id, kind) for case_id in cases for kind in ("image", "label")}
    actual_entries = [(entry["case_id"], entry["kind"]) for entry in selection["files"]]
    if len(actual_entries) != len(expected_entries) or set(actual_entries) != expected_entries:
        raise ValueError("frozen source inventory incomplete or duplicated")
    for entry in selection["files"]:
        case_id, kind = entry["case_id"], entry["kind"]
        if case_id not in cases or kind not in ("image", "label"):
            raise ValueError("unexpected frozen source entry")
        name = "imaging.nii.gz" if kind == "image" else "segmentation.nii.gz"
        path = session / "evaluation_cache" / case_id / name
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError("staged input size mismatch")
    return selection, cohort, config


def preflight(session: Path, candidate: Path) -> dict:
    selection, _cohort, config = frozen_inputs(session, candidate)
    state = snapshot()
    issues = []
    if state["disk_free_bytes"] < MIN_FREE_DISK:
        issues.append("physical_disk_reservation")
    if state["available_ram_bytes"] < int(1.5 * GIB):
        issues.append("available_ram_floor")
    if state["swap_used_bytes"] > LIMIT_SWAP:
        issues.append("swap_limit")
    if (session / "evaluation/evaluation/B0/summary.json").exists():
        baseline = read_json(session / "evaluation/evaluation/B0/summary.json")
        if (baseline.get("status") != "success"
                or baseline.get("cases_requested") != selection["evaluation_cases"]
                or len(baseline.get("cases", [])) != 6):
            issues.append("baseline_incomplete")
    else:
        issues.append("baseline_missing")
    return {"status": "pass" if not issues else "fail", "issues": issues,
            "free_disk_gib": round(state["disk_free_bytes"] / GIB, 2),
            "available_ram_gib": round(state["available_ram_bytes"] / GIB, 2),
            "swap_used_mib": round(state["swap_used_bytes"] / MIB, 1),
            "case_count": len(selection["evaluation_cases"]),
            "candidate_checkpoint": str(candidate), "python": config["python"]}


def checked_score(row: dict, group: str, key: str) -> float | None:
    value = row[group][key]["dice"]
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("invalid Dice value")
    return number


def audit(session: Path, candidate: Path, *, verify_receipts: bool = True) -> dict:
    selection, _cohort, config = frozen_inputs(session, candidate)
    cases = selection["evaluation_cases"]
    summaries = {
        model: read_json(session / "evaluation/evaluation" / model / "summary.json")
        for model in ("B0", "A")
    }
    expected_hashes = {"B0": config["base_checkpoint_sha256"],
                       "A": selection["candidate_checkpoint_sha256"]}
    for model, summary in summaries.items():
        if (summary.get("status") != "success" or summary.get("model") != model
                or summary.get("cohort") != "evaluation"
                or summary.get("cases_requested") != cases
                or summary.get("checkpoint_sha256") != expected_hashes[model]
                or len(summary.get("cases", [])) != len(cases)
                or [row["case_id"] for row in summary["cases"]] != cases):
            raise ValueError(f"{model}: incomplete or mismatched evaluation")
    protocol = summaries["B0"]["protocol"]
    if (protocol != summaries["A"]["protocol"]
            or protocol.get("surface_voxel_limit") != 0
            or protocol.get("patch") != 128 or protocol.get("stride") != 64):
        raise ValueError("evaluation protocols differ")
    measurements = [("classes", str(number), name)
                    for number, name in ((1, "kidney"), (2, "tumor"), (3, "cyst"))]
    measurements += [("hec", name, name)
                     for name in ("kidney_and_masses", "kidney_mass", "tumor")]
    pairs = []
    for index, case_id in enumerate(cases):
        result = {"case_id": case_id, "metrics": {}}
        image = session / "evaluation_cache" / case_id / "imaging.nii.gz"
        label = session / "evaluation_cache" / case_id / "segmentation.nii.gz"
        for model, summary in summaries.items():
            row = summary["cases"][index]
            prediction = session / "evaluation/evaluation" / model / "predictions" / f"{case_id}.nii.gz"
            if (not row.get("provenance_verified") or Path(row["prediction"]).resolve() != prediction.resolve()
                    or sha256_file(prediction) != row["prediction_sha256"]):
                raise ValueError(f"{model}: prediction provenance mismatch")
            if verify_receipts:
                verify(prediction, binding(expected_hashes[model], image, label, protocol))
            result[f"{model.lower()}_prediction_sha256"] = row["prediction_sha256"]
        left, right = (summaries[model]["cases"][index] for model in ("B0", "A"))
        for group, key, name in measurements:
            baseline, candidate_score = checked_score(left, group, key), checked_score(right, group, key)
            delta = candidate_score - baseline if baseline is not None and candidate_score is not None else None
            result["metrics"][f"{group}.{name}"] = {
                "b0_dice": baseline, "a_dice": candidate_score, "delta": delta,
            }
        pairs.append(result)
    aggregates = {}
    for group, _key, name in measurements:
        label = f"{group}.{name}"
        deltas = [row["metrics"][label]["delta"] for row in pairs]
        valid = [value for value in deltas if value is not None]
        b0 = [row["metrics"][label]["b0_dice"] for row in pairs]
        a = [row["metrics"][label]["a_dice"] for row in pairs]
        aggregates[label] = {
            "valid_pairs": len(valid),
            "b0_mean_dice": statistics.mean(value for value in b0 if value is not None) if any(value is not None for value in b0) else None,
            "a_mean_dice": statistics.mean(value for value in a if value is not None) if any(value is not None for value in a) else None,
            "mean_paired_delta": statistics.mean(valid) if valid else None,
            "median_paired_delta": statistics.median(valid) if valid else None,
            "improved_cases": sum(value > 0 for value in valid),
            "worsened_cases": sum(value < 0 for value in valid),
            "unchanged_cases": sum(value == 0 for value in valid),
        }
    return {
        "schema_version": "shadowtrainer.stunet-train64-heldout-audit/v1",
        "status": "complete_paired_six_case_technical_evaluation",
        "selection_sha256": sha256_file(session / "selection.json"),
        "checkpoint_sha256": expected_hashes,
        "case_count": len(cases), "protocol": protocol,
        "metrics": aggregates, "paired_cases": pairs,
        "limitations": [
            "small KiTS23 held-out cohort; not clinical validation",
            "local schedule/history audit cannot rule out use in deleted or external artifacts",
            "candidate selected on four development cases before this evaluation",
            "HD95 explicitly skipped under the bounded memory protocol",
            "no ResNet prognostic comparison or universal acceleration claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=SESSION)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    session, candidate = args.session.resolve(), args.candidate.resolve()
    if args.preflight_only and args.audit_only:
        parser.error("choose preflight-only or audit-only")
    if args.audit_only:
        result = audit(session, candidate)
    else:
        result = preflight(session, candidate)
        if args.preflight_only or result["status"] != "pass":
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "pass" else 2
        config = read_json(session / "resolved-config.json")
        command = [config["python"], str(HERE / "stunet_campaign_evaluate.py"),
                   "--config", str(session / "resolved-config.json"),
                   "--session", str(session), "--model", "A",
                   "--checkpoint", str(candidate), "--surface-voxel-limit", "0"]
        deadline = time.time() + 45 * 60
        phase = run_guarded(command, LAB, session / "eval_A.log",
                            session / "eval_A.resources.jsonl", deadline, deadline,
                            swap_limit=LIMIT_SWAP)
        atomic_json(session / "eval_A_resume.json", phase)
        if phase["status"] != "success":
            print(json.dumps({"status": phase["status"], "reasons": phase["reasons"]}))
            return 2
        result = audit(session, candidate)
    output = session / "paired-audit.json"
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "case_count": result["case_count"],
                      "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
