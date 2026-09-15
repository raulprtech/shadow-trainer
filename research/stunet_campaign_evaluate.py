#!/usr/bin/env python3
"""Full-volume STU-Net evaluation with KiTS23 HEC metrics and exports."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import psutil
import torch
from scipy import ndimage

ROOT = Path("/home/raulprtech/stream-hot-kits-mini")
SHADOW = Path("/home/raulprtech/shadow-trainer")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHADOW / "research"))

from experiments.stage32.full_volume import normalize_patch, pad_to_patch, starts
from kits23_mini_worker import build_model, normalize_logits
from stunet_campaign import HEC_CLASSES, aggregate_rows, atomic_json, segmentation_metrics, sha256_file

GIB = 2**30
MIB = 2**20


def hd95(prediction, target, spacing) -> tuple[float | None, str]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return None, "both_empty"
    if not prediction.any() or not target.any():
        return None, "empty_mismatch"
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = np.logical_xor(prediction, ndimage.binary_erosion(prediction, structure=structure))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure))
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    to_target = target_distance[pred_surface]
    del target_distance
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    to_prediction = pred_distance[target_surface]
    del pred_distance
    distances = np.concatenate((to_target, to_prediction))
    return (float(np.percentile(distances, 95)) if len(distances) else 0.0), "ok"


def save_overlay(image, target, prediction, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tumor_per_slice = (target == 2).sum(axis=(0, 1))
    renal_per_slice = (target > 0).sum(axis=(0, 1))
    scores = tumor_per_slice if tumor_per_slice.max() else renal_per_slice
    index = int(scores.argmax())
    base = np.clip(image[:, :, index], -200, 300).T
    truth = target[:, :, index].T
    predicted = prediction[:, :, index].T
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, mask, label in (
        (axes[0], None, "CT"),
        (axes[1], truth, "Referencia"),
        (axes[2], predicted, "Predicción"),
    ):
        axis.imshow(base, cmap="gray", origin="lower")
        if mask is not None:
            if np.any(mask == 1):
                axis.contour(mask == 1, colors="#00d084", linewidths=0.8)
            if np.any(mask == 2):
                axis.contour(mask == 2, colors="#ff3b30", linewidths=1.1)
            if np.any(mask == 3):
                axis.contour(mask == 3, colors="#ffd60a", linewidths=0.8)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(f"{title} · corte {index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_case(model, device, cache: Path, case_id: str, output: Path) -> dict:
    process = psutil.Process()
    case_dir = cache / case_id
    image_nii = nib.load(case_dir / "imaging.nii.gz", keep_file_open=True)
    label_nii = nib.load(case_dir / "segmentation.nii.gz", keep_file_open=True)
    if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError(f"{case_id}: image/label spatial mismatch")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    target = np.asarray(label_nii.dataobj, dtype=np.uint8)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    padded, crop = pad_to_patch(image, 128)
    grid = [starts(length, 128, 64) for length in padded.shape]
    windows = [(d, h, w) for d in grid[0] for h in grid[1] for w in grid[2]]
    probabilities = np.zeros((4, *padded.shape), dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    peak_rss = process.memory_info().rss / MIB
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inference_seconds = 0.0
    with torch.inference_mode():
        for d, h, w in windows:
            normalized, _mean, _std = normalize_patch(padded[d:d + 128, h:h + 128, w:w + 128])
            x = torch.from_numpy(normalized)[None, None].to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                values = torch.softmax(normalize_logits(model(x)).float(), dim=1)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            cpu = values[0].cpu().numpy()
            probabilities[:, d:d + 128, h:h + 128, w:w + 128] += cpu
            counts[d:d + 128, h:h + 128, w:w + 128] += 1
            peak_rss = max(peak_rss, process.memory_info().rss / MIB)
            del x, values, cpu, normalized
    if np.any(counts == 0):
        raise RuntimeError(f"{case_id}: uncovered voxel")
    probabilities /= counts[None]
    prediction = np.empty(padded.shape, dtype=np.uint8)
    for start in range(0, padded.shape[0], 8):
        prediction[start:start + 8] = probabilities[:, start:start + 8].argmax(axis=0)
    prediction = prediction[crop]
    del probabilities, counts, padded
    metrics = segmentation_metrics(prediction, target)
    for class_id in (1, 2, 3):
        value, status = hd95(prediction == class_id, target == class_id, spacing)
        metrics["classes"][str(class_id)]["hd95_mm"] = value
        metrics["classes"][str(class_id)]["surface_status"] = status
    for name, labels in HEC_CLASSES.items():
        value, status = hd95(np.isin(prediction, labels), np.isin(target, labels), spacing)
        metrics["hec"][name]["hd95_mm"] = value
        metrics["hec"][name]["surface_status"] = status
    prediction_path = output / "predictions" / f"{case_id}.nii.gz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    header = label_nii.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction, label_nii.affine, header), prediction_path)
    overlay_path = output / "overlays" / f"{case_id}.png"
    save_overlay(image, target, prediction, overlay_path, case_id)
    result = {
        "case_id": case_id,
        **metrics,
        "shape": list(target.shape),
        "spacing": list(spacing),
        "windows": len(windows),
        "inference_seconds": inference_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "peak_rss_mib": peak_rss,
        "prediction": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "overlay": str(overlay_path),
    }
    del image, target, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--model", choices=["B0", "A", "B"], required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cohort = json.loads((args.session / "cohort.json").read_text())
    checkpoint = (
        Path(config["base_checkpoint"]) if args.model == "B0"
        else args.session / f"arm_{args.model}" / "best.checkpoint.pt"
    )
    label = "development" if args.development else "evaluation"
    cases = cohort["development_cases"] if args.development else cohort["evaluation_cases"]
    cache = Path(config["training_cache"] if args.development else config["evaluation_cache"])
    output = args.session / "evaluation" / label / args.model
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary = {
        "schema_version": "shadowtrainer.stunet-evaluation/v1",
        "status": "running", "model": args.model, "cohort": label,
        "cases_requested": cases, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "started_at": time.time(),
    }
    atomic_json(summary_path, summary)
    handles = []
    try:
        if shutil.disk_usage("/mnt/c").free < 20 * GIB + 512 * MIB:
            raise RuntimeError("evaluation disk reservation failed")
        memory = psutil.virtual_memory()
        if memory.available < int(1.5 * GIB) or psutil.swap_memory().used > 192 * MIB:
            raise RuntimeError("evaluation memory guard failed")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device("cuda")
        model, _stack, handles, stages = build_model("WORKSPACE_AWARE_HYBRID_V2", device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()
        rows = []
        for case_id in cases:
            row = evaluate_case(model, device, cache, case_id, output)
            rows.append(row)
            summary["cases"] = rows
            atomic_json(summary_path, summary)
        summary.update({
            "status": "success", "finished_at": time.time(),
            "duration_seconds": time.time() - summary["started_at"],
            "aggregate": aggregate_rows(rows), "structural_stages": stages,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rows),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in rows),
            "peak_rss_mib": max(row["peak_rss_mib"] for row in rows),
        })
    except Exception as exc:
        summary.update({
            "status": "error", "error": repr(exc),
            "traceback": traceback.format_exc(), "failed_at": time.time(),
        })
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
