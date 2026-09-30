#!/usr/bin/env python3
"""Freeze the remaining locally unseen KiTS validation cases before inference."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from run_stunet_campaign import GIB, LAB, HERE, snapshot
from stunet_campaign import (atomic_json, csv_case_ids, manifest_records,
                             prior_used_cases, record_bytes, select_evaluation_cases,
                             sha256_file)

SHADOW = HERE.parent
SESSION = HERE / "workspace/stunet-train64-heldout-r1"


def previous_shadow_cases() -> set[str]:
    used: set[str] = set()
    for path in (HERE / "workspace").glob("*/cohort.json"):
        payload = json.loads(path.read_text())
        for key in ("train_cases", "development_cases", "evaluation_cases"):
            used.update(payload.get(key, []))
    return used


def history_paths(cases: list[str]) -> list[Path]:
    pattern = "|".join(re.escape(case_id) for case_id in cases)
    command = ["rg", "-l", "--no-ignore", "--hidden"]
    for glob in ("*.json", "*.csv", "*.md"):
        command.extend(["--glob", glob])
    command.extend([pattern, str(LAB), str(SHADOW)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("history inventory failed")
    return [Path(line) for line in result.stdout.splitlines() if line]


def freeze_plan(config: dict) -> dict:
    train = csv_case_ids(Path(config["train_csv"]))
    validation = csv_case_ids(Path(config["validation_csv"]))
    records = manifest_records(Path(config["manifest"]))
    used = prior_used_cases(LAB / "runs") | previous_shadow_cases() | set(train)
    candidates = select_evaluation_cases(validation, used, seed=20260922,
                                          count=len(validation))
    if len(candidates) < 4:
        raise RuntimeError("fewer than four locally unseen validation cases")
    candidates = candidates[:6]
    references = history_paths(candidates)
    allowed = [LAB / "manifests", LAB / "source_manifest"]
    unexpected = [path for path in references if not any(path.is_relative_to(root) for root in allowed)]
    if unexpected:
        raise RuntimeError(f"candidate history needs review: {len(unexpected)} unexpected files")
    total = sum(record_bytes(records[case_id]) for case_id in candidates)
    if total > int(1.5 * GIB):
        raise RuntimeError("frozen candidate set exceeds 1.5 GiB staging budget")
    if set(candidates) & (set(train) | previous_shadow_cases()):
        raise RuntimeError("frozen cohort overlaps prior local use")
    return {
        "schema_version": "shadowtrainer.stunet-train64-heldout-plan/v1",
        "status": "frozen_before_labels_or_predictions",
        "selection": "sha256(20260922:case_id), ascending among validation CSV cases absent from local schedules and shadow cohorts",
        "evaluation_cases": candidates,
        "case_count": len(candidates),
        "compressed_input_bytes": total,
        "source_inventory_reference_count": len(references),
        "unexpected_prior_text_references": 0,
        "source_manifest_sha256": sha256_file(Path(config["manifest"])),
        "train_csv_sha256": sha256_file(Path(config["train_csv"])),
        "validation_csv_sha256": sha256_file(Path(config["validation_csv"])),
        "candidate_checkpoint_sha256": sha256_file(HERE / "workspace/stunet-train64-a-v1/arm_A/latest.checkpoint.pt"),
        "recovery_audit_sha256": sha256_file(HERE / "workspace/stunet-train64-a-v1/recovery-audit.json"),
        "files": [
            {"case_id": case_id, "kind": kind, **records[case_id][kind]}
            for case_id in candidates for kind in ("image", "label")
        ],
        "limits": [
            "local textual inventory cannot prove absence on other machines or deleted artifacts",
            "candidate selection is fixed before downloading labels or running inference",
            "no clinical validation claim from a small KiTS cohort",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = HERE / "workspace/stunet-train64-a-v1/resolved-config.json"
    config = json.loads(source.read_text())
    if SESSION.exists():
        raise FileExistsError(f"heldout session already exists: {SESSION}")
    plan = freeze_plan(config)
    resources = snapshot()
    if (resources["disk_free_bytes"] < 23 * GIB
            or resources["available_ram_bytes"] < 5 * GIB
            or resources["swap_used_bytes"] > 192 * 2**20):
        raise RuntimeError("heldout physical resource preflight failed")
    display = {"status": "pass", "case_count": plan["case_count"],
               "compressed_input_gib": round(plan["compressed_input_bytes"] / GIB, 3),
               "disk_free_gib": round(resources["disk_free_bytes"] / GIB, 3),
               "source_inventory_reference_count": plan["source_inventory_reference_count"]}
    if args.preflight_only:
        print(json.dumps(display, indent=2))
        return 0
    SESSION.mkdir(parents=True)
    config["evaluation_cache"] = str(SESSION / "evaluation_cache")
    config["evaluation_cache_bytes"] = int(1.5 * GIB)
    prior = json.loads((HERE / "workspace/stunet-train64-a-v1/cohort.json").read_text())
    cohort = {"schema_version": "shadowtrainer.stunet-train64-heldout-cohort/v1",
              "train_cases": prior["train_cases"],
              "development_cases": prior["development_cases"],
              "evaluation_cases": plan["evaluation_cases"]}
    atomic_json(SESSION / "selection.json", plan)
    atomic_json(SESSION / "cohort.json", cohort)
    atomic_json(SESSION / "resolved-config.json", config)
    print(json.dumps({**display, "session": str(SESSION)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
