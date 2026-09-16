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
    # scipy's full-volume distance transforms can require multiple extra
    # copies of a large KiTS volume. Keep the primary overlap metrics
    # complete and declare HD95 unavailable when the bounded budget cannot
    # safely accommodate the transform.
    if np.asarray(prediction).size > 64_000_000:
        return None, "skipped_memory_budget"
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
    if image.ndim == 2:
        base = np.clip(image, -200, 300).T
    else:
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


def materialize_image_memmap(proxy, shape: tuple[int, ...], case_id: str) -> np.memmap:
    """Decode a compressed NIfTI once into a reusable disk-backed float32 cache."""
    cache_path = Path("/tmp") / f"shadowtrainer-stunet-{case_id}.f32"
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if not cache_path.exists() or cache_path.stat().st_size != expected_bytes:
        partial = cache_path.with_suffix(cache_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        mapped = np.memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, shape[0], 8):
            stop = min(start + 8, shape[0])
            mapped[start:stop] = np.asarray(proxy[start:stop, :, :], dtype=np.float32)
        mapped.flush()
        del mapped
        os.replace(partial, cache_path)
    return np.memmap(cache_path, mode="r", dtype=np.float32, shape=shape)


def read_proxy_patch(proxy, starts_xyz, patch, pads, shape) -> np.ndarray:
    """Read one padded patch without materializing the full CT volume."""
    output = np.full((patch, patch, patch), -1024.0, dtype=np.float32)
    source_slices = []
    destination_slices = []
    for axis, start in enumerate(starts_xyz):
        before = pads[axis][0]
        source_start = max(0, start - before)
        source_end = min(shape[axis], start + patch - before)
        destination_start = source_start - (start - before)
        destination_end = destination_start + max(0, source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        destination_slices.append(slice(destination_start, destination_end))
    if all(section.start < section.stop for section in source_slices):
        output[tuple(destination_slices)] = np.asarray(proxy[tuple(source_slices)], dtype=np.float32)
    return output


def bounded_segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Compute overlap counts in shallow slabs to avoid full-volume bool masks."""
    def counts_for(mask_values):
        predicted_voxels = target_voxels = intersection_voxels = 0
        for start in range(0, prediction.shape[0], 8):
            stop = min(start + 8, prediction.shape[0])
            predicted = mask_values(prediction[start:stop])
            truth = mask_values(target[start:stop])
            predicted_voxels += int(predicted.sum())
            target_voxels += int(truth.sum())
            intersection_voxels += int(np.logical_and(predicted, truth).sum())
        denominator = predicted_voxels + target_voxels
        return {
            "prediction_voxels": predicted_voxels,
            "target_voxels": target_voxels,
            "intersection_voxels": intersection_voxels,
            "dice": (2.0 * intersection_voxels / denominator) if denominator else None,
            "precision": (intersection_voxels / predicted_voxels) if predicted_voxels else (1.0 if not target_voxels else 0.0),
            "recall": (intersection_voxels / target_voxels) if target_voxels else (1.0 if not predicted_voxels else None),
        }
    classes = {
        str(class_id): {"name": name, **counts_for(lambda values, class_id=class_id: values == class_id)}
        for class_id, name in HEC_CLASS_NAMES.items()
    }
    hec = {
        name: counts_for(lambda values, labels=labels: np.isin(values, labels))
        for name, labels in HEC_CLASSES.items()
    }
    return {"classes": classes, "hec": hec}


HEC_CLASS_NAMES = {1: "kidney", 2: "tumor", 3: "cyst"}

def evaluate_case(model, device, cache: Path, case_id: str, output: Path) -> dict:
    process = psutil.Process()
    case_dir = cache / case_id
    image_nii = nib.load(case_dir / "imaging.nii.gz", keep_file_open=True)
    label_nii = nib.load(case_dir / "segmentation.nii.gz", keep_file_open=True)
    if image_nii.shape != label_nii.shape or not np.allclose(image_nii.affine, label_nii.affine):
        raise ValueError(f"{case_id}: image/label spatial mismatch")
    target = np.asarray(label_nii.dataobj, dtype=np.uint8)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    image_shape = tuple(int(value) for value in image_nii.shape)
    pads = tuple((max(0, 128 - length) // 2, max(0, 128 - length) - max(0, 128 - length) // 2) for length in image_shape)
    image_source = materialize_image_memmap(image_nii.dataobj, image_shape, case_id)
    padded_shape = tuple(length + before + after for length, (before, after) in zip(image_shape, pads))
    crop = tuple(slice(before, before + length) for length, (before, _after) in zip(image_shape, pads))
    grid = [starts(length, 128, 64) for length in padded_shape]
    windows = [(d, h, w) for d in grid[0] for h in grid[1] for w in grid[2]]
    # Accumulate one depth slab at a time. A full 4-channel probability
    # volume for a large KiTS case is several GiB and violates the RSS guard.
    prediction_padded = np.empty(padded_shape, dtype=np.uint8)
    slab_depth = 128
    peak_rss = process.memory_info().rss / MIB
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inference_seconds = 0.0
    with torch.inference_mode():
        for slab_start in range(0, padded_shape[0], slab_depth):
            slab_end = min(slab_start + slab_depth, padded_shape[0])
            slab_len = slab_end - slab_start
            probabilities = np.zeros((4, slab_len, padded_shape[1], padded_shape[2]), dtype=np.float32)
            counts = np.zeros((slab_len, padded_shape[1], padded_shape[2]), dtype=np.uint8)
            for d, h, w in windows:
                overlap_start = max(d, slab_start)
                overlap_end = min(d + 128, slab_end)
                if overlap_start >= overlap_end:
                    continue
                patch_image = read_proxy_patch(image_source, (d, h, w), 128, pads, image_shape)
                normalized, _mean, _std = normalize_patch(patch_image)
                del patch_image
                x = torch.from_numpy(normalized)[None, None].to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    values = torch.softmax(normalize_logits(model(x)).float(), dim=1)
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - started
                cpu = values[0].cpu().numpy()
                local_start = overlap_start - slab_start
                local_end = overlap_end - slab_start
                source_start = overlap_start - d
                source_end = overlap_end - d
                probabilities[:, local_start:local_end, h:h + 128, w:w + 128] += cpu[:, source_start:source_end]
                counts[local_start:local_end, h:h + 128, w:w + 128] += 1
                peak_rss = max(peak_rss, process.memory_info().rss / MIB)
                del x, values, cpu, normalized
            if np.any(counts == 0):
                raise RuntimeError(f"{case_id}: uncovered voxel in slab {slab_start}:{slab_end}")
            probabilities /= counts[None]
            prediction_padded[slab_start:slab_end] = probabilities.argmax(axis=0)
            del probabilities, counts
            gc.collect()
            peak_rss = max(peak_rss, process.memory_info().rss / MIB)
    prediction = prediction_padded[crop]
    del prediction_padded
    metrics = bounded_segmentation_metrics(prediction, target)
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
    overlay_path = None
    overlay_status = "ok"
    if int(np.prod(image_shape)) <= 64_000_000:
        overlay_path = output / "overlays" / f"{case_id}.png"
        tumor_per_slice = (target == 2).sum(axis=(0, 1))
        renal_per_slice = (target > 0).sum(axis=(0, 1))
        scores = tumor_per_slice if tumor_per_slice.max() else renal_per_slice
        overlay_index = int(scores.argmax())
        overlay_image = np.asarray(image_nii.dataobj[:, :, overlay_index], dtype=np.float32)
        save_overlay(overlay_image, target, prediction, overlay_path, case_id)
        del overlay_image
    else:
        overlay_status = "skipped_memory_budget"
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
        "overlay": str(overlay_path) if overlay_path else None,
        "overlay_status": overlay_status,
    }
    del target, prediction, image_source
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_saved_prediction(cache: Path, case_id: str, output: Path) -> dict:
    """Resume a case whose prediction was durably written before interruption."""
    case_dir = cache / case_id
    image_nii = nib.load(case_dir / "imaging.nii.gz", keep_file_open=True)
    label_nii = nib.load(case_dir / "segmentation.nii.gz", keep_file_open=True)
    prediction_path = output / "predictions" / f"{case_id}.nii.gz"
    prediction_nii = nib.load(prediction_path)
    target = np.asarray(label_nii.dataobj, dtype=np.uint8)
    prediction = np.asarray(prediction_nii.dataobj, dtype=np.uint8)
    if prediction.shape != target.shape:
        raise ValueError(f"{case_id}: saved prediction shape mismatch")
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    metrics = bounded_segmentation_metrics(prediction, target)
    for class_id in (1, 2, 3):
        value, status = hd95(prediction == class_id, target == class_id, spacing)
        metrics["classes"][str(class_id)]["hd95_mm"] = value
        metrics["classes"][str(class_id)]["surface_status"] = status
    for name, labels in HEC_CLASSES.items():
        value, status = hd95(np.isin(prediction, labels), np.isin(target, labels), spacing)
        metrics["hec"][name]["hd95_mm"] = value
        metrics["hec"][name]["surface_status"] = status
    padded_shape = tuple(max(128, int(value)) for value in target.shape)
    windows = len(starts(padded_shape[0], 128, 64)) * len(starts(padded_shape[1], 128, 64)) * len(starts(padded_shape[2], 128, 64))
    result = {
        "case_id": case_id, **metrics, "shape": list(target.shape),
        "spacing": list(spacing), "windows": windows,
        "inference_seconds": None, "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0, "peak_rss_mib": 0.0,
        "prediction": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
        "overlay": None, "overlay_status": "skipped_memory_budget",
        "resumed_from_prediction": True,
    }
    del target, prediction
    gc.collect()
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
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        # The checkpoint state dict is no longer needed after loading. Keeping
        # it alongside the model and volume buffers breaks the RSS budget on
        # the largest independent cases.
        del payload
        gc.collect()
        model.eval()
        rows = []
        for case_id in cases:
            saved_prediction = output / "predictions" / f"{case_id}.nii.gz"
            if saved_prediction.exists():
                row = evaluate_saved_prediction(cache, case_id, output)
            else:
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
