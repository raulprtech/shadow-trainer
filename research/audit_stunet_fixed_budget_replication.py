#!/usr/bin/env python3
"""Offline, patient-paired audit of the two 64/96 STU-Net development seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from run_stunet_train64 import confirmed_updates, plan_digest
from stunet_campaign import atomic_json, sha256_file


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "workspace"
RUNS = {
    "20260922": {
        "64": WORKSPACE / "stunet-fixed-budget64-a-v1",
        "96": WORKSPACE / "stunet-train96-a-v1",
    },
    "20260923": {
        "64": WORKSPACE / "stunet-fixed-budget-replication-seed20260923/cases64",
        "96": WORKSPACE / "stunet-fixed-budget-replication-seed20260923/cases96",
    },
}
LIMITATIONS = [
    "The first comparison was designed after inspecting the 96-case result; the second seed was chosen after the first result.",
    "The same four development patients select checkpoints and supply all metrics here; they are not an independent test.",
    "The second-seed 64-case selected checkpoint is at step 1024, not the equal-budget step 1536.",
    "Equal update counts do not equalize distinct-patient exposure or case repetitions.",
    "The already opened six-case cohort and locked test were not used in this audit.",
    "Neither clinical usefulness nor an optimal case count follows from two seeds and four development patients.",
]


def load_run(seed: str, arm: str, directory: Path) -> dict:
    envelope = json.loads((directory / "plan.json").read_text())
    plan = envelope["plan"]
    if envelope["sha256"] != plan_digest(plan):
        raise ValueError(f"plan digest mismatch: {seed}/{arm}")
    if len(plan["train_cases"]) != int(arm) or len(set(plan["train_cases"])) != int(arm):
        raise ValueError(f"train cohort mismatch: {seed}/{arm}")
    development = plan["development_cases"]
    if len(development) != 4 or set(development) & set(plan["train_cases"]):
        raise ValueError(f"train/development leakage: {seed}/{arm}")
    if plan["expected_updates"] != 1536 or plan.get("evaluation_cases"):
        raise ValueError(f"budget or evaluation scope changed: {seed}/{arm}")
    config = json.loads((directory / "resolved-config.json").read_text())
    if config["training_seed"] != int(seed):
        raise ValueError(f"training seed mismatch: {seed}/{arm}")
    summary = json.loads((directory / "arm_A/summary.json").read_text())
    if (summary.get("status") != "success" or summary.get("global_step") != 1536
            or confirmed_updates(directory) != 1536):
        raise ValueError(f"training incomplete: {seed}/{arm}")
    events = [json.loads(line) for line in (directory / "arm_A/metrics.jsonl").read_text().splitlines()]
    steps = [event for event in events if event.get("kind") == "train_step"]
    if len(steps) != 1536 or any(not math.isfinite(event["loss"]) for event in steps):
        raise ValueError(f"step count or finite-loss audit failed: {seed}/{arm}")
    boundaries = [event["global_step"] for event in events if event.get("kind") == "case_boundary"]
    if boundaries != list(range(8, 1537, 8)):
        raise ValueError(f"case-boundary sequence changed: {seed}/{arm}")
    epochs = {event["epoch"]: event for event in events if event.get("kind") == "epoch"}
    if max(epochs) != plan["epochs"] or epochs[max(epochs)]["global_step"] != 1536:
        raise ValueError(f"last epoch mismatch: {seed}/{arm}")
    best_epoch = int(summary["best_epoch"])
    if best_epoch and (best_epoch not in epochs or not epochs[best_epoch]["eligible"]):
        raise ValueError(f"selected epoch mismatch: {seed}/{arm}")
    if best_epoch == 0:
        selected_development = summary["baseline_development"]
        selected_steps = 0
    else:
        selected_development = epochs[best_epoch]["development"]
        selected_steps = epochs[best_epoch]["global_step"]
    last_development = epochs[max(epochs)]["development"]
    for development_result in (selected_development, last_development):
        cases = development_result["cases"]
        if [case["case_id"] for case in cases] != development:
            raise ValueError(f"development case order changed: {seed}/{arm}")
        for metric, label in (("kidney_mean", "1"), ("tumor_mean", "2")):
            values = [case["per_class"][label]["dice_hard"] for case in cases]
            if not all(value is not None and math.isfinite(value) for value in values):
                raise ValueError(f"undefined metric: {seed}/{arm}/{metric}")
            if not math.isclose(statistics.mean(values), development_result[metric], abs_tol=1e-10):
                raise ValueError(f"mean mismatch: {seed}/{arm}/{metric}")
    return {"seed": seed, "arm": arm, "directory": str(directory), "plan": plan,
            "plan_sha256": envelope["sha256"], "summary": summary,
            "events": events, "steps": steps, "selected_epoch": best_epoch,
            "selected_steps": selected_steps, "selected_development": selected_development,
            "last_development": last_development}


def paired(seed: str, a: dict, b: dict, endpoint: str) -> tuple[dict, list[dict]]:
    development64 = a[f"{endpoint}_development"]
    development96 = b[f"{endpoint}_development"]
    if a["plan"]["development_cases"] != b["plan"]["development_cases"]:
        raise ValueError(f"development mismatch: {seed}")
    rows = []
    for case64, case96 in zip(development64["cases"], development96["cases"]):
        if case64["case_id"] != case96["case_id"]:
            raise ValueError(f"pair mismatch: {seed}")
        row = {"seed": seed, "endpoint": endpoint, "case_id": case64["case_id"]}
        for metric, label in (("kidney", "1"), ("tumor", "2")):
            v64 = case64["per_class"][label]["dice_hard"]
            v96 = case96["per_class"][label]["dice_hard"]
            row.update({f"{metric}_64": v64, f"{metric}_96": v96,
                        f"{metric}_delta_64_minus_96": v64 - v96})
        rows.append(row)
    result = {"seed": seed, "endpoint": endpoint, "steps_64": 1536 if endpoint == "last" else a["selected_steps"],
              "steps_96": 1536 if endpoint == "last" else b["selected_steps"]}
    for metric in ("kidney", "tumor"):
        deltas = [row[f"{metric}_delta_64_minus_96"] for row in rows]
        result[metric] = {"mean_64": development64[f"{metric}_mean"],
                          "mean_96": development96[f"{metric}_mean"],
                          "mean_delta_64_minus_96": statistics.mean(deltas),
                          "median_delta_64_minus_96": statistics.median(deltas),
                          "64_better": sum(value > 1e-12 for value in deltas),
                          "96_better": sum(value < -1e-12 for value in deltas),
                          "tie": sum(abs(value) <= 1e-12 for value in deltas)}
    return result, rows


def audit() -> dict:
    runs = {seed: {arm: load_run(seed, arm, directory) for arm, directory in arms.items()}
            for seed, arms in RUNS.items()}
    reference = runs["20260922"]
    replicate = runs["20260923"]
    for arm, run in reference.items():
        receipt = json.loads((Path(run["directory"]) / "supervisor.json").read_text())
        if (receipt.get("status") != "success" or receipt.get("confirmed_updates") != 1536
                or receipt.get("plan_sha256") != run["plan_sha256"]):
            raise ValueError(f"first-seed supervisor mismatch: {arm}")
    for seed, arms in runs.items():
        a, b = arms["64"], arms["96"]
        if a["plan"]["train_cases"] != b["plan"]["train_cases"][:64]:
            raise ValueError(f"64-case prefix differs: {seed}")
        if a["plan"]["base_checkpoint_sha256"] != b["plan"]["base_checkpoint_sha256"]:
            raise ValueError(f"base model differs: {seed}")
        for left, right in zip(a["steps"][:512], b["steps"][:512]):
            for key in ("case_id", "center", "flips", "target_class", "patch_index"):
                if left[key] != right[key]:
                    raise ValueError(f"first shared epoch differs: {seed}/{key}")
    for arm in ("64", "96"):
        if replicate[arm]["plan"]["train_cases"] != reference[arm]["plan"]["train_cases"]:
            raise ValueError(f"case identity differs between seeds: {arm}")
    second_receipt = json.loads((WORKSPACE / "stunet-fixed-budget-replication-seed20260923/supervisor.json").read_text())
    if second_receipt["status"] != "success":
        raise ValueError("replication supervisor incomplete")
    for arm in ("64", "96"):
        if second_receipt["phases"][arm]["confirmed_updates"] != 1536:
            raise ValueError(f"replication supervisor step mismatch: {arm}")
        if second_receipt["frozen"][f"cases{arm}_plan_sha256"] != replicate[arm]["plan_sha256"]:
            raise ValueError(f"replication plan receipt mismatch: {arm}")
        best = Path(replicate[arm]["directory"]) / "arm_A/best.checkpoint.pt"
        if sha256_file(best) != second_receipt["phases"][arm]["best_checkpoint_sha256"]:
            raise ValueError(f"best checkpoint digest mismatch: {arm}")
    comparisons, patient_rows = [], []
    for seed, arms in runs.items():
        for endpoint in ("last", "selected"):
            comparison, rows = paired(seed, arms["64"], arms["96"], endpoint)
            comparisons.append(comparison)
            patient_rows.extend(rows)
    return {"schema_version": "shadowtrainer.stunet-fixed-budget-two-seed-audit/v1",
            "status": "verified", "scope": "four reused development patients only",
            "run_provenance": [{"seed": seed, "arm": arm,
                                "plan_sha256": run["plan_sha256"],
                                "selected_epoch": run["selected_epoch"],
                                "selected_steps": run["selected_steps"],
                                "last_steps": 1536}
                               for seed, arms in runs.items() for arm, run in arms.items()],
            "comparisons": comparisons, "patients": patient_rows,
            "limitations": LIMITATIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "stunet-fixed-budget-replication-seed20260923/audit")
    args = parser.parse_args()
    result = audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "two-seed-audit.json", result)
    with (args.output_dir / "development-pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["seed", "endpoint", "case_id", "kidney_64", "kidney_96",
                   "kidney_delta_64_minus_96", "tumor_64", "tumor_96", "tumor_delta_64_minus_96"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["patients"])
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
