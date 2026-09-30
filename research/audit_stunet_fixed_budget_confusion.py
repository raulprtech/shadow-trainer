#!/usr/bin/env python3
"""Bounded offline four-class confusion audit of already sealed KiTS masks.

Reads gzip voxel payloads sequentially in 1-MiB blocks. No CT pixels, GPU,
new cases or held-out data are opened. Results use stable P1–P6 aliases.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from stunet_campaign import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
ORIGIN = HERE / "workspace/stunet-train64-heldout-r1"
RUN64 = HERE / "workspace/stunet-fixed-budget64-reused-six-r1"
RUN96 = HERE / "workspace/stunet-train96-reused-six-r1"
CHUNK_VOXELS = 1 << 20


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def md5_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def nifti_uint8(path: Path):
    image = nib.load(path, keep_file_open=False)
    if (image.get_data_dtype() != np.dtype("uint8") or
            image.dataobj.slope != 1.0 or image.dataobj.inter != 0.0):
        raise ValueError(f"not unscaled uint8: {path}")
    return image


def count_streams(label_path: Path, prediction_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    from contextlib import ExitStack

    label = nifti_uint8(label_path)
    predictions = {name: nifti_uint8(path) for name, path in prediction_paths.items()}
    for name, image in predictions.items():
        if image.shape != label.shape or not np.allclose(image.affine, label.affine):
            raise ValueError(f"prediction geometry mismatch: {name}")
    remaining = int(np.prod(label.shape))
    matrices = {name: np.zeros((4, 4), dtype=np.int64) for name in predictions}
    with ExitStack() as stack:
        target = stack.enter_context(gzip.open(label_path, "rb"))
        streams = {name: stack.enter_context(gzip.open(path, "rb"))
                   for name, path in prediction_paths.items()}
        target.seek(label.dataobj.offset)
        for name, stream in streams.items():
            stream.seek(predictions[name].dataobj.offset)
        while remaining:
            count = min(remaining, CHUNK_VOXELS)
            target_bytes = target.read(count)
            if len(target_bytes) != count:
                raise ValueError("truncated label payload")
            truth = np.frombuffer(target_bytes, dtype=np.uint8)
            if np.any(truth > 3):
                raise ValueError("invalid reference label")
            for name, stream in streams.items():
                prediction_bytes = stream.read(count)
                if len(prediction_bytes) != count:
                    raise ValueError(f"truncated prediction payload: {name}")
                predicted = np.frombuffer(prediction_bytes, dtype=np.uint8)
                if np.any(predicted > 3):
                    raise ValueError(f"invalid prediction label: {name}")
                encoded = truth * np.uint8(4) + predicted
                matrices[name] += np.bincount(encoded, minlength=16).reshape(4, 4)
            remaining -= count
    if any(int(matrix.sum()) != int(np.prod(label.shape)) for matrix in matrices.values()):
        raise ValueError("voxel count mismatch")
    return matrices


def dice(matrix: np.ndarray, labels: tuple[int, ...]) -> float | None:
    reference = int(matrix[list(labels), :].sum())
    predicted = int(matrix[:, list(labels)].sum())
    intersection = int(matrix[np.ix_(labels, labels)].sum())
    denominator = reference + predicted
    return 2 * intersection / denominator if denominator else None


def verify_metrics(matrix: np.ndarray, row: dict) -> None:
    groups = {
        "classes": {"1": (1,), "2": (2,), "3": (3,)},
        "hec": {"kidney_and_masses": (1, 2, 3),
                "kidney_mass": (2, 3), "tumor": (2,)},
    }
    for group, definitions in groups.items():
        for name, labels in definitions.items():
            measured = dice(matrix, labels)
            recorded = row[group][name]["dice"]
            if measured is None and recorded is None:
                continue
            if (measured is None or recorded is None or
                    not math.isclose(measured, recorded, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"recorded Dice differs: {group}/{name}")


def audit() -> dict:
    selection = read(ORIGIN / "selection.json")
    paired = read(RUN64 / "paired-audit.json")
    roots = {"64": RUN64, "96": RUN96}
    summaries = {name: read(root / "evaluation/evaluation/A/summary.json")
                 for name, root in roots.items()}
    cases = selection["evaluation_cases"]
    if (len(cases) != 6 or len(set(cases)) != 6 or
            paired["status"] != "complete_post_hoc_exploratory" or
            paired["case_count"] != 6 or
            any(summaries[name]["status"] != "success" or
                summaries[name]["cases_requested"] != cases or
                summaries[name]["checkpoint_sha256"] !=
                paired["checkpoint_sha256"][name] for name in roots)):
        raise ValueError("frozen paired cohort or checkpoint changed")
    label_md5 = {(row["case_id"], row["kind"]): row["md5"]
                 for row in selection["files"]}
    rows = []
    for index, case in enumerate(cases):
        label = ORIGIN / "evaluation_cache" / case / "segmentation.nii.gz"
        if md5_file(label) != label_md5[(case, "label")]:
            raise ValueError("reference label digest mismatch")
        paths = {name: root / "evaluation/evaluation/A/predictions" / f"{case}.nii.gz"
                 for name, root in roots.items()}
        for name in roots:
            recorded = summaries[name]["cases"][index]
            if (recorded["case_id"] != case or
                    sha256_file(paths[name]) != recorded["prediction_sha256"]):
                raise ValueError("sealed prediction digest mismatch")
        matrices = count_streams(label, paths)
        for name, matrix in matrices.items():
            verify_metrics(matrix, summaries[name]["cases"][index])
        rows.append({"alias": f"P{index+1}",
                     "confusion_target_rows_predicted_columns": {
                         name: matrix.tolist() for name, matrix in matrices.items()}})
    aggregate = {name: np.sum([np.asarray(row["confusion_target_rows_predicted_columns"][name],
                                          dtype=np.int64) for row in rows], axis=0)
                 for name in roots}
    return {
        "schema_version": "shadowtrainer.stunet-fixed-budget-confusion-audit/v1",
        "status": "complete_reused_six_only", "case_count": 6,
        "source_selection_sha256": sha256_file(ORIGIN / "selection.json"),
        "paired_audit_sha256": sha256_file(RUN64 / "paired-audit.json"),
        "checkpoint_sha256": paired["checkpoint_sha256"],
        "cases": rows,
        "aggregate_confusion_target_rows_predicted_columns": {
            name: matrix.tolist() for name, matrix in aggregate.items()},
        "limitations": ["same six previously opened cases", "post-hoc diagnostic",
                        "no independent efficacy or clinical claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=RUN64 / "confusion-audit.json")
    args = parser.parse_args()
    result = audit()
    if args.output.exists() and read(args.output) != result:
        raise ValueError("existing audit differs")
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "case_count": result["case_count"],
                      "aggregate": result["aggregate_confusion_target_rows_predicted_columns"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
