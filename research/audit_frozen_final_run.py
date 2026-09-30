#!/usr/bin/env python3
"""Read-only audit of a completed frozen STU-Net run; writes a sanitized receipt.

The six cases are aliases F1-F6 in the output. This tool does not train,
select checkpoints, promote a model, or interpret a reused cohort as independent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from prediction_provenance import sha256, verify


MODELS = ("B0", "A", "B")
METRICS = {
    "tumor": ("classes", "2"),
    "kidney": ("classes", "1"),
    "renal_union": ("hec", "kidney_and_masses"),
    "cyst": ("classes", "3"),
}


def file_hashes(path):
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def stats(values):
    if any(value is None for value in values):
        raise ValueError("missing Dice in complete cohort")
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def compare(left, right):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {
        **stats(differences),
        "improved": sum(value > 0 for value in differences),
        "worse": sum(value < 0 for value in differences),
        "tied": sum(value == 0 for value in differences),
    }


def audit(plan_path, session):
    plan = json.loads(plan_path.read_text())
    campaign = json.loads((session / "campaign-summary.json").read_text())
    digest = sha256(plan_path)
    if campaign.get("status") != "success" or campaign.get("plan_sha256") != digest:
        raise ValueError("campaign not successful or plan digest mismatch")
    if set(campaign.get("models", {})) != set(MODELS):
        raise ValueError("model set incomplete")
    ids = plan["cohort"]["case_ids_in_frozen_order"]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("invalid frozen cohort")
    verified = json.loads((session / "verified-inputs.json").read_text())
    by_key = {(row["case_id"], row["kind"]): row for row in verified}
    if len(by_key) != 12 or len(verified) != 12:
        raise ValueError("verified input set incomplete")
    for frozen in plan["inputs"]:
        row = by_key[(frozen["case_id"], frozen["kind"])]
        path = Path(frozen["path"])
        if row["path"] != str(path) or path.stat().st_size != frozen["frozen_size_bytes"]:
            raise ValueError("input path or size changed")
        md5, current_sha = file_hashes(path)
        if md5 != frozen["frozen_md5"] or current_sha != row["sha256"]:
            raise ValueError("final input content changed")
    for row in plan["checkpoints"].values():
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("checkpoint content changed")
    for path, expected in plan["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("frozen source changed")

    summaries = {}
    receipt_count = 0
    for model in MODELS:
        path = session / f"evaluation/evaluation/{model}/summary.json"
        row = campaign["models"][model]
        if sha256(path) != row["summary_sha256"] or row["cases"] != 6:
            raise ValueError("summary digest or count mismatch")
        summary = json.loads(path.read_text())
        cases = summary.get("cases", [])
        if summary.get("status") != "success" or [case["case_id"] for case in cases] != ids:
            raise ValueError("summary status or order mismatch")
        if summary["checkpoint_sha256"] != plan["checkpoints"][model]["sha256"]:
            raise ValueError("summary checkpoint mismatch")
        if summary["protocol"]["version"] != plan["work"]["evaluation_protocol"]:
            raise ValueError("evaluation protocol mismatch")
        for case in cases:
            cid = case["case_id"]
            expected = {
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "image_sha256": by_key[(cid, "image")]["sha256"],
                "label_sha256": by_key[(cid, "label")]["sha256"],
                "protocol": summary["protocol"],
            }
            receipt = verify(case["prediction"], expected)
            if not case.get("provenance_verified") or receipt["prediction_sha256"] != case["prediction_sha256"]:
                raise ValueError("prediction row not verified")
            receipt_count += 1
        summaries[model] = summary

    samples = [json.loads(line) for line in (session / "resources.jsonl").open()]
    if not samples or set(row["model"] for row in samples) != set(MODELS):
        raise ValueError("incomplete resource telemetry")
    limits = plan["resources"]
    minimum_disk = min(row["disk_free_bytes"] for row in samples)
    minimum_ram = min(row["ram_available_bytes"] for row in samples)
    maximum_swap = max(row["swap_used_bytes"] for row in samples)
    maximum_rss = max(row["tree_rss_bytes"] for row in samples)
    artifact_bytes = sum(path.stat().st_size for path in session.rglob("*")
                         if path.is_file() and not path.is_symlink())
    if (minimum_disk < limits["physical_disk_floor_bytes"]
            or minimum_ram < limits["min_available_ram_bytes"]
            or maximum_swap > limits["max_swap_bytes"]
            or maximum_rss > limits["max_tree_rss_bytes"]
            or artifact_bytes > limits["new_artifacts_and_temporaries_cap_bytes"]):
        raise ValueError("resource limit exceeded")
    if any((session / "decoded_cache").iterdir()):
        raise ValueError("decoded cache not released at completion")

    metric_rows = {}
    for name, path in METRICS.items():
        values = {
            model: [case[path[0]][path[1]]["dice"] for case in summaries[model]["cases"]]
            for model in MODELS
        }
        metric_rows[name] = {
            "models": {model: stats(values[model]) for model in MODELS},
            "comparisons": {
                "A-B0": compare(values["A"], values["B0"]),
                "B-B0": compare(values["B"], values["B0"]),
                "B-A": compare(values["B"], values["A"]),
            },
        }
    patients = []
    for index in range(6):
        rows = {model: summaries[model]["cases"][index] for model in MODELS}
        target_tumor = rows["B0"]["classes"]["2"]["target_voxels"]
        target_cyst = rows["B0"]["classes"]["3"]["target_voxels"]
        if any(row["classes"]["2"]["target_voxels"] != target_tumor or
               row["classes"]["3"]["target_voxels"] != target_cyst for row in rows.values()):
            raise ValueError("target counts differ across models")
        patients.append({
            "alias": f"F{index + 1}",
            "tumor_target_voxels": target_tumor,
            "cyst_target_voxels": target_cyst,
            "tumor": {
                model: {key: rows[model]["classes"]["2"][key]
                        for key in ("dice", "precision", "recall", "prediction_voxels")}
                for model in MODELS
            },
            "surface_status": {model: rows[model]["classes"]["2"]["surface_status"] for model in MODELS},
        })
    return {
        "schema": "shadowtrainer.frozen-final-paired-audit/v1",
        "status": "passed",
        "scope": "six same-cohort final cases reused from r1; descriptive exploratory replication, not independent or clinical validation",
        "plan_sha256": digest,
        "models": list(MODELS),
        "patients": patients,
        "metrics": metric_rows,
        "verified_prediction_receipts": receipt_count,
        "summary_sha256": {model: campaign["models"][model]["summary_sha256"] for model in MODELS},
        "duration_seconds": campaign["elapsed_seconds"],
        "model_duration_seconds": {model: summaries[model]["duration_seconds"] for model in MODELS},
        "resources": {
            "telemetry_samples": len(samples),
            "minimum_disk_free_bytes": minimum_disk,
            "minimum_ram_available_bytes": minimum_ram,
            "maximum_swap_bytes": maximum_swap,
            "maximum_tree_rss_bytes": maximum_rss,
            "new_artifact_bytes": artifact_bytes,
            "decoded_cache_final_bytes": 0,
        },
        "limitations": [
            "Same six cases as r1; historical independence not established.",
            "HD95 skipped on all six volumes by the frozen 20-million-voxel memory rule.",
            "Cyst segmentation remains exploratory; no clinical validation.",
            "No model selection or promotion based on these final results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be new")
    result = audit(args.plan, args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"],
        "verified_prediction_receipts": result["verified_prediction_receipts"],
        "plan_sha256": result["plan_sha256"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
