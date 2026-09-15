"""Scientific helpers for the bounded STU-Net campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


SEED = 20260915
CLASS_NAMES = {1: "kidney", 2: "tumor", 3: "cyst"}
HEC_CLASSES = {
    "kidney_and_masses": (1, 2, 3),
    "kidney_mass": (2, 3),
    "tumor": (2,),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_case_ids(path: Path) -> list[str]:
    with path.open(newline="") as stream:
        return [row["case_id"] for row in csv.DictReader(stream)]


def prior_used_cases(runs_dir: Path) -> set[str]:
    used: set[str] = set()
    for path in sorted(runs_dir.glob("*/schedule.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("train_cases", "validation_cases", "development_cases"):
            values = payload.get(key, [])
            if isinstance(values, list):
                used.update(value for value in values if isinstance(value, str))
    return used


def select_evaluation_cases(
    candidates: Iterable[str], used: set[str], *, seed: int = SEED, count: int = 6
) -> list[str]:
    eligible = sorted(set(candidates) - used)
    ranked = sorted(
        eligible,
        key=lambda case_id: (
            hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(), case_id
        ),
    )
    return ranked[:count]


def manifest_records(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {row["case_id"]: row for row in payload["cases"]}


def record_bytes(record: dict) -> int:
    return sum(int(record[key]["size_bytes"]) for key in ("image", "label"))


def sample_targets(labels_present: Iterable[int], case_id: str, epoch: int) -> list[int]:
    present = set(int(value) for value in labels_present)
    tumor = 2 if 2 in present else (1 if 1 in present else 0)
    renal = 1 if 1 in present else tumor
    cyst = 3 if 3 in present else tumor
    targets = [tumor, tumor, tumor, tumor, renal, renal, cyst, 0]
    offset = int(hashlib.sha256(f"{case_id}:{epoch}".encode()).hexdigest()[:8], 16)
    offset %= len(targets)
    return targets[offset:] + targets[:offset]


def binary_counts(prediction, target) -> dict[str, int | float | None]:
    import numpy as np

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    intersection = int(np.logical_and(prediction, target).sum())
    denominator = pred_count + target_count
    return {
        "prediction_voxels": pred_count,
        "target_voxels": target_count,
        "intersection_voxels": intersection,
        "dice": (2.0 * intersection / denominator) if denominator else None,
        "precision": intersection / pred_count if pred_count else (1.0 if not target_count else 0.0),
        "recall": intersection / target_count if target_count else (1.0 if not pred_count else None),
    }


def segmentation_metrics(prediction, target) -> dict:
    import numpy as np

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    classes = {
        str(class_id): {"name": name, **binary_counts(prediction == class_id, target == class_id)}
        for class_id, name in CLASS_NAMES.items()
    }
    hec = {
        name: binary_counts(np.isin(prediction, labels), np.isin(target, labels))
        for name, labels in HEC_CLASSES.items()
    }
    return {"classes": classes, "hec": hec}


def hierarchical_loss(logits, target):
    import torch
    import torch.nn.functional as functional

    weights = torch.tensor([1.0, 1.0, 2.0, 2.0], device=logits.device)
    cross_entropy = functional.cross_entropy(logits.float(), target, weight=weights)
    probabilities = torch.softmax(logits.float(), dim=1)
    renal_probability = probabilities[:, 1:].sum(dim=1)
    renal_target = (target > 0).float()
    tumor_probability = probabilities[:, 2]
    tumor_target = (target == 2).float()
    dimensions = tuple(range(1, target.ndim))

    def dice_loss(probability, truth):
        intersection = (probability * truth).sum(dimensions)
        denominator = probability.sum(dimensions) + truth.sum(dimensions)
        return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    renal = dice_loss(renal_probability, renal_target)
    tumor = dice_loss(tumor_probability, tumor_target)
    dice = 0.3 * renal + 0.7 * tumor
    total = 0.5 * cross_entropy + 0.5 * dice
    return total, cross_entropy, dice, renal, tumor


def baseline_loss(logits, target):
    import torch
    import torch.nn.functional as functional

    weights = torch.tensor([1.0, 1.0, 2.0, 2.0], device=logits.device)
    cross_entropy = functional.cross_entropy(logits.float(), target, weight=weights)
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = functional.one_hot(target, 4).permute(0, 4, 1, 2, 3).float()
    dimensions = (0, 2, 3, 4)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dimensions)
    denominator = probabilities[:, 1:].sum(dimensions) + one_hot[:, 1:].sum(dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = 0.8 * cross_entropy + 0.2 * dice
    return total, cross_entropy, dice, dice, dice


def aggregate_rows(rows: list[dict]) -> dict:
    import numpy as np

    result: dict[str, dict] = {"classes": {}, "hec": {}}
    for section in ("classes", "hec"):
        names = sorted({name for row in rows for name in row[section]})
        for name in names:
            items = [row[section][name] for row in rows]
            dice = [item["dice"] for item in items if item["dice"] is not None]
            denominator = sum(item["prediction_voxels"] + item["target_voxels"] for item in items)
            result[section][name] = {
                "mean_dice": float(np.mean(dice)) if dice else None,
                "median_dice": float(np.median(dice)) if dice else None,
                "micro_dice": (
                    2.0 * sum(item["intersection_voxels"] for item in items) / denominator
                    if denominator else None
                ),
                "cases": len(items),
            }
    return result
