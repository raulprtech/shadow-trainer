#!/usr/bin/env python3
"""Train one deterministic STU-Net campaign arm with case-boundary recovery."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path("/home/raulprtech/stream-hot-kits-mini")
SHADOW = Path("/home/raulprtech/shadow-trainer")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SHADOW / "research"))

from experiments.stage29.runtime import boot_id, configure
from experiments.stage32.full_volume import evaluate_case
from experiments.stage32.patches import extract_targeted_patch
from kits23_mini_worker import build_model, canonical_name, normalize_logits
from kits23_stage19_worker import atomic_torch_save
from stage23_bounded_nifti import BoundedWindowStager, label_counts, load_case
from stage34_resilient_staging import ProgressAwareRcloneSource
from stunet_campaign import (
    atomic_json,
    baseline_loss,
    hierarchical_loss,
    sample_targets,
    sha256_file,
)

GIB = 2**30
MIB = 2**20


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), **payload}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def resource_snapshot(process: psutil.Process | None = None) -> dict:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = process.memory_info().rss if process else 0
    return {
        "time": time.time(),
        "monotonic": time.monotonic(),
        "boot_id": boot_id(),
        "disk_free_bytes": shutil.disk_usage("/mnt/c").free,
        "available_ram_bytes": memory.available,
        "swap_used_bytes": swap.used,
        "process_rss_bytes": rss,
    }


def assert_resources(process: psutil.Process | None = None, reserve: int = 0) -> dict:
    state = resource_snapshot(process)
    failures = []
    if state["disk_free_bytes"] < 20 * GIB + reserve:
        failures.append("physical_disk_floor")
    if state["available_ram_bytes"] < 512 * MIB:
        failures.append("available_ram_floor")
    if state["swap_used_bytes"] > 256 * MIB:
        failures.append("swap_limit")
    if process and state["process_rss_bytes"] > int(4.5 * GIB):
        failures.append("process_rss_limit")
    if failures:
        raise RuntimeError("resource_guard:" + ",".join(failures))
    return state


def validation(model, device, cache: Path, cases: list[str]) -> dict:
    model.eval()
    rows = []
    for case_id in cases:
        case = evaluate_case(model, device, cache, case_id, 128, 0.5)
        rows.append(case)
    kidney = [row["per_class"]["1"]["dice_hard"] for row in rows]
    tumor = [row["per_class"]["2"]["dice_hard"] for row in rows]
    cyst = [row["per_class"]["3"]["dice_hard"] for row in rows]
    return {
        "cases": rows,
        "kidney_mean": float(np.mean([value for value in kidney if value is not None])),
        "tumor_mean": float(np.mean([value for value in tumor if value is not None])),
        "cyst_mean": float(np.mean([value for value in cyst if value is not None])),
    }


def save_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    summary: dict,
    next_epoch: int,
    next_case_index: int,
    global_step: int,
    best_tumor: float,
    best_epoch: int,
    schedule: dict,
) -> None:
    assert_resources(reserve=256 * MIB)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise FloatingPointError("refusing checkpoint with nonfinite model tensor")
    atomic_torch_save(
        {
            "schema_version": "shadowtrainer.stunet-checkpoint/v1",
            "arm": summary["arm"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng": capture_rng(),
            "next_epoch": next_epoch,
            "next_case_index": next_case_index,
            "global_step": global_step,
            "best_tumor": best_tumor,
            "best_epoch": best_epoch,
            "schedule": schedule,
            "base_sha256": summary["base_sha256"],
            "summary": summary,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cohort = json.loads((args.session / "cohort.json").read_text())
    schedule = {
        "train_cases": cohort["train_cases"],
        "development_cases": cohort["development_cases"],
        "seed": config["seed"],
        "patches_per_case": config["patches_per_case"],
        "epochs": 1 if args.pilot else args.epochs,
    }
    arm_name = f"pilot_{args.arm}" if args.pilot else f"arm_{args.arm}"
    output = args.session / arm_name
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    summary_path = output / "summary.json"
    latest_path = output / "latest.checkpoint.pt"
    best_path = output / "best.checkpoint.pt"
    base_path = Path(config["base_checkpoint"])
    base_sha256 = sha256_file(base_path)
    if base_sha256 != config["base_checkpoint_sha256"]:
        raise RuntimeError("base checkpoint identity changed")
    summary = {
        "schema_version": "shadowtrainer.stunet-arm/v1",
        "status": "running",
        "arm": args.arm,
        "loss": "stage20" if args.arm == "A" else "hierarchical_renal_tumor",
        "pilot": args.pilot,
        "epochs_requested": schedule["epochs"],
        "base_sha256": base_sha256,
        "started_at": time.time(),
        "configuration": schedule,
    }
    atomic_json(summary_path, summary)
    handles = []
    source = None
    process = psutil.Process()
    try:
        assert_resources(process)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # The historical runtime enables strict deterministic algorithms.
        # Apply the campaign's audited warn-only policy afterwards because
        # CUDA NLL/CrossEntropy has no deterministic implementation in this
        # PyTorch/CUDA build. Seeds and deterministic cuDNN remain enabled.
        configure(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device("cuda")
        raw_manifest = json.loads(Path(config["manifest"]).read_text())
        sizes = {
            row[kind]["source"]: row[kind]["size_bytes"]
            for row in raw_manifest["cases"]
            for kind in ("image", "label")
        }
        source = ProgressAwareRcloneSource(
            config["remote"], sizes, "/mnt/c", 20 * GIB,
            attempts=3, stall_seconds=180, attempt_seconds=1800, poll_seconds=2,
        )
        cache = Path(config["training_cache"])
        stager = BoundedWindowStager(
            Path(config["manifest"]), cache, int(config["training_cache_bytes"]),
            source, validate_nifti=True,
        )
        development_cases = cohort["development_cases"]
        stager.stage_window(development_cases)
        model, stage_stack, handles, stages = build_model("WORKSPACE_AWARE_HYBRID_V2", device)
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        model.load_state_dict(base["model_state"])
        del base
        heads, backbone = [], []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)
            (heads if canonical_name(name).startswith("seg_outputs.") else backbone).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": 1e-5, "name": "backbone"},
                {"params": heads, "lr": 1e-4, "name": "head"},
            ],
            weight_decay=1e-5,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        next_epoch = next_case_index = global_step = 0
        baseline = validation(model, device, cache, development_cases)
        best_tumor = baseline["tumor_mean"]
        best_epoch = 0
        summary["baseline_development"] = baseline
        if not latest_path.exists():
            save_training_checkpoint(
                best_path, model=model, optimizer=optimizer, scaler=scaler, summary=summary,
                next_epoch=0, next_case_index=0, global_step=0, best_tumor=best_tumor,
                best_epoch=best_epoch, schedule=schedule,
            )
        if latest_path.exists():
            resumed = torch.load(latest_path, map_location="cpu", weights_only=False)
            if resumed["schedule"] != schedule or resumed["base_sha256"] != base_sha256:
                raise RuntimeError("resume provenance mismatch")
            model.load_state_dict(resumed["model_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            scaler.load_state_dict(resumed["scaler_state"])
            restore_rng(resumed["rng"])
            next_epoch = resumed["next_epoch"]
            next_case_index = resumed["next_case_index"]
            global_step = resumed["global_step"]
            best_tumor = resumed["best_tumor"]
            best_epoch = resumed["best_epoch"]
            append_jsonl(metrics_path, {"kind": "run_resumed", "global_step": global_step})
        loss_function = baseline_loss if args.arm == "A" else hierarchical_loss
        peak_allocated = peak_reserved = peak_rss = 0.0
        epochs = schedule["epochs"]
        train_cases = schedule["train_cases"][:1] if args.pilot else schedule["train_cases"]
        for epoch in range(next_epoch, epochs):
            epoch_losses = []
            start_index = next_case_index if epoch == next_epoch else 0
            for case_index in range(start_index, len(train_cases)):
                assert_resources(process)
                case_id = train_cases[case_index]
                stager.stage_window([case_id], pinned=development_cases)
                image, label, metadata = load_case(cache, case_id)
                values, counts = label_counts(label)
                labels = values.tolist()
                targets = sample_targets(labels, case_id, epoch)
                for patch_index, target_class in enumerate(targets):
                    sample_seed = (
                        config["seed"] + epoch * 1000003 + case_index * 10007 + patch_index * 101
                    )
                    x_cpu, y_cpu, patch_metadata = extract_targeted_patch(
                        image, label, 128, target_class, sample_seed, True
                    )
                    x, y = x_cpu.to(device), y_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = normalize_logits(model(x))
                        loss, cross_entropy, dice, renal, tumor = loss_function(logits, y)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    value = float(loss.detach().cpu())
                    epoch_losses.append(value)
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / MIB)
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / MIB)
                    peak_rss = max(peak_rss, process.memory_info().rss / MIB)
                    append_jsonl(metrics_path, {
                        "kind": "train_step", "arm": args.arm, "epoch": epoch + 1,
                        "case_index": case_index, "patch_index": patch_index,
                        "case_id": case_id, "target_class": target_class,
                        "loss": value, "cross_entropy": float(cross_entropy.detach().cpu()),
                        "dice_term": float(dice.detach().cpu()),
                        "renal_term": float(renal.detach().cpu()),
                        "tumor_term": float(tumor.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "peak_allocated_mib": peak_allocated,
                        "peak_reserved_mib": peak_reserved,
                        "rss_mib": process.memory_info().rss / MIB,
                        "source_labels": labels,
                        "source_label_counts": {str(int(v)): int(n) for v, n in zip(values, counts)},
                        **patch_metadata,
                    })
                    del x_cpu, y_cpu, x, y, logits, loss, cross_entropy, dice, renal, tumor
                del image, label
                gc.collect()
                following_case = case_index + 1
                following_epoch = epoch
                if following_case >= len(train_cases):
                    following_case = 0
                    following_epoch = epoch + 1
                save_training_checkpoint(
                    latest_path, model=model, optimizer=optimizer, scaler=scaler,
                    summary=summary, next_epoch=following_epoch,
                    next_case_index=following_case, global_step=global_step,
                    best_tumor=best_tumor, best_epoch=best_epoch, schedule=schedule,
                )
                append_jsonl(metrics_path, {
                    "kind": "case_boundary", "epoch": epoch + 1, "case_id": case_id,
                    "next_epoch": following_epoch, "next_case_index": following_case,
                    "global_step": global_step, "resources": resource_snapshot(process),
                })
            development = validation(model, device, cache, development_cases)
            eligible = (
                development["kidney_mean"] >= baseline["kidney_mean"] - 0.02
                and development["tumor_mean"] > best_tumor
            )
            if eligible:
                best_tumor = development["tumor_mean"]
                best_epoch = epoch + 1
                save_training_checkpoint(
                    best_path, model=model, optimizer=optimizer, scaler=scaler,
                    summary=summary, next_epoch=epoch + 1, next_case_index=0,
                    global_step=global_step, best_tumor=best_tumor,
                    best_epoch=best_epoch, schedule=schedule,
                )
            epoch_row = {
                "kind": "epoch", "epoch": epoch + 1, "global_step": global_step,
                "mean_train_loss": float(np.mean(epoch_losses)),
                "development": development, "eligible": eligible,
                "best_epoch": best_epoch, "best_tumor": best_tumor,
            }
            append_jsonl(metrics_path, epoch_row)
            summary.update({
                "epochs_completed": epoch + 1, "global_step": global_step,
                "last_epoch": epoch_row, "best_epoch": best_epoch,
                "best_tumor": best_tumor, "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved, "peak_rss_mib": peak_rss,
            })
            atomic_json(summary_path, summary)
            next_case_index = 0
        last_path = output / "last.checkpoint.pt"
        save_training_checkpoint(
            last_path, model=model, optimizer=optimizer, scaler=scaler, summary=summary,
            next_epoch=epochs, next_case_index=0, global_step=global_step,
            best_tumor=best_tumor, best_epoch=best_epoch, schedule=schedule,
        )
        summary.update({
            "status": "success", "finished_at": time.time(),
            "duration_seconds": time.time() - summary["started_at"],
            "cache_final": stager.status(), "structural_stages": stages,
        })
    except Exception as exc:
        summary.update({
            "status": "error", "error": repr(exc), "traceback": traceback.format_exc(),
            "failed_at": time.time(),
        })
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        summary["transfers"] = source.transfers if source else []
        summary["transfer_attempts"] = source.attempt_log if source else []
        summary["disk_free_end_bytes"] = shutil.disk_usage("/mnt/c").free
        atomic_json(summary_path, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "baseline_development"}, indent=2))
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
