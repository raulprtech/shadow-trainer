"""Audited execution, checkpointing, and resume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import JobConfig
from .errors import AdmissionError, IntegrityError
from .events import EventLog, atomic_json
from .state_digest import state_digest
from .policy import Plan, build_plan
from .reporting import redact, render_report
from .resources import snapshot
from .staging import CaseStager, LocalSource, RcloneSource, load_manifest
from .workloads import create_workload


def _config_digest(config: JobConfig) -> str:
    canonical = json.dumps(config.public_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inspect_job(config: JobConfig) -> tuple[dict[str, Any], Plan]:
    disk_anchor = config.resources.disk_check_path
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    environment = snapshot(disk_anchor)
    return environment, build_plan(config, environment)


def _source(config: JobConfig):
    if config.data.source.type == "local":
        root = Path(config.data.source.root).expanduser()
        if not root.is_absolute():
            root = (config.config_path.parent / root).resolve()
        return LocalSource(root)
    return RcloneSource(config.data.source.root)


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _assert_disk_floor(config: JobConfig, reserve_bytes: int = 0) -> None:
    anchor = config.resources.disk_check_path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free
    required = config.resources.disk_floor_bytes + reserve_bytes
    if free < required:
        raise IntegrityError(f"runtime disk reservation violated: {free} < {required}")


def _atomic_torch_save(path: Path, payload: dict) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _windows(case_ids: list[str], size: int) -> list[list[str]]:
    return [case_ids[index:index + size] for index in range(0, len(case_ids), size)]


def run_job(
    config: JobConfig,
    *,
    resume_dir: Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    environment, plan = inspect_job(config)
    if not plan.admitted:
        raise AdmissionError("job rejected by resource admission", details=plan.as_dict())
    run_dir = (resume_dir or config.output_dir).resolve()
    if resume_dir is None and run_dir.exists() and any(run_dir.iterdir()):
        raise IntegrityError(f"output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(run_dir / "events.jsonl")
    started = time.time()
    atomic_json(run_dir / "environment.json", redact(environment))
    atomic_json(run_dir / "plan.json", plan.as_dict())
    atomic_json(run_dir / "job.json", redact(config.public_dict()))
    events.emit({"kind": "run_started", "job_id": config.job_id, "resume": bool(resume_dir)})

    records = load_manifest(config.data.manifest)
    case_ids = list(records)
    batches = _windows(case_ids, config.training.cases_per_window)
    stager = CaseStager(
        config.data.manifest,
        config.data.cache_dir,
        config.resources.cache_bytes,
        _source(config),
        validate_nifti=config.data.validate_nifti,
        event_callback=events.emit,
    )
    workload = create_workload(config.workload.type, config.seed, config.workload.options)
    checkpoint_path = run_dir / "latest.checkpoint.pt"
    next_epoch = next_window = global_step = 0
    if resume_dir is not None:
        if not checkpoint_path.is_file():
            raise IntegrityError("resume requested without latest.checkpoint.pt")
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("config_digest") != _config_digest(config):
            raise IntegrityError("checkpoint job configuration digest mismatch")
        workload.restore(checkpoint["workload"])
        next_epoch = int(checkpoint["next_epoch"])
        next_window = int(checkpoint["next_window"])
        global_step = int(checkpoint["global_step"])
        events.emit(
            {
                "kind": "run_resumed",
                "next_epoch": next_epoch,
                "next_window": next_window,
                "global_step": global_step,
            }
        )

    import torch

    events.emit({
        "kind": "workload_ready",
        "resume": bool(resume_dir),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "workload_state_sha256": state_digest(workload.checkpoint_state()),
    })

    summary: dict[str, Any] = {
        "schema_version": "shadowtrainer.summary/v1",
        "job_id": config.job_id,
        "status": "running",
        "started_at": started,
        "selected_strategy": plan.selected_strategy,
        "prefetch_status": plan.prefetch_status,
        "global_step": global_step,
    }
    atomic_json(run_dir / "summary.json", summary)

    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-prefetch")
        if plan.selected_strategy == "prefetch"
        else None
    )
    pending: tuple[int, int, list[str], Future] | None = None
    execution_started = time.perf_counter()
    try:
        stop = False
        for epoch in range(next_epoch, config.training.epochs):
            window_start = next_window if epoch == next_epoch else 0
            for window_index in range(window_start, len(batches)):
                current = batches[window_index]
                if pending and pending[0:3] == (epoch, window_index, current):
                    prefetch_wait_started = time.perf_counter()
                    staged_paths = pending[3].result()
                    prefetch_wait_seconds = time.perf_counter() - prefetch_wait_started
                    events.emit({
                        "kind": "prefetch_consumed", "epoch": epoch,
                        "window": window_index, "wait_seconds": prefetch_wait_seconds,
                    })
                    pending = None
                else:
                    staged_paths = stager.stage_batch(current)

                future_epoch, future_index = epoch, window_index + 1
                if future_index >= len(batches):
                    future_epoch, future_index = epoch + 1, 0
                if executor and future_epoch < config.training.epochs:
                    future_cases = batches[future_index]
                    protected_bytes = sum(records[item].total_bytes for item in set(current + future_cases))
                    if protected_bytes <= config.resources.cache_bytes:
                        future = executor.submit(
                            stager.stage_batch, future_cases, protected_extra=current
                        )
                        pending = (future_epoch, future_index, future_cases, future)
                        events.emit(
                            {
                                "kind": "prefetch_launched",
                                "epoch": future_epoch,
                                "window": future_index,
                                "cases": future_cases,
                            }
                        )
                    else:
                        events.emit(
                            {
                                "kind": "prefetch_skipped",
                                "reason": "current_and_next_exceed_cache",
                            }
                        )

                for case_id, case_path in zip(current, staged_paths, strict=True):
                    row = workload.train_case(case_path, case_id)
                    global_step += 1
                    events.emit(
                        {
                            "kind": "train_step",
                            "epoch": epoch,
                            "window": window_index,
                            "global_step": global_step,
                            "case_id": case_id,
                            **row,
                        }
                    )
                    if not row["finite"]:
                        raise FloatingPointError("non-finite training loss")
                    if config.training.max_steps and global_step >= config.training.max_steps:
                        stop = True
                    if stop_after_steps and global_step >= stop_after_steps:
                        stop = True
                    if stop:
                        break

                candidate_epoch, candidate_window = epoch, window_index + 1
                if candidate_window >= len(batches):
                    candidate_epoch, candidate_window = epoch + 1, 0
                checkpoint_payload = {
                    "schema_version": "shadowtrainer.checkpoint/v1",
                    "config_digest": _config_digest(config),
                    "next_epoch": candidate_epoch,
                    "next_window": candidate_window,
                    "global_step": global_step,
                    "workload": workload.checkpoint_state(),
                }
                current_artifact_bytes = (
                    _directory_bytes(run_dir) + stager.status()["occupancy_bytes"]
                )
                remaining_reservation = max(
                    0, config.resources.artifact_budget_bytes - current_artifact_bytes
                )
                _assert_disk_floor(config, remaining_reservation)
                _atomic_torch_save(checkpoint_path, checkpoint_payload)
                _assert_disk_floor(config)
                events.emit(
                    {
                        "kind": "checkpoint",
                        "next_epoch": candidate_epoch,
                        "next_window": candidate_window,
                        "global_step": global_step,
                    }
                )
                artifact_bytes = _directory_bytes(run_dir) + stager.status()["occupancy_bytes"]
                if artifact_bytes > config.resources.artifact_budget_bytes:
                    raise IntegrityError("combined cache and run artifacts exceeded budget")
                summary.update(
                    {
                        "global_step": global_step,
                        "next_epoch": candidate_epoch,
                        "next_window": candidate_window,
                        "cache": stager.status(),
                        "artifact_bytes": artifact_bytes,
                    }
                )
                atomic_json(run_dir / "summary.json", summary)
                if stop:
                    break
            next_window = 0
            if stop:
                break
        if pending:
            pending[3].cancel()
        expected_steps = len(case_ids) * config.training.epochs
        complete = global_step >= expected_steps or (
            config.training.max_steps is not None and global_step >= config.training.max_steps
        )
        summary.update(
            {
                "status": "success" if complete else "interrupted",
                "finished_at": time.time(),
                "duration_seconds": time.time() - started,
                "execution_seconds": time.perf_counter() - execution_started,
                "global_step": global_step,
                "expected_steps": expected_steps,
                "cache": stager.status(),
            }
        )
        events.emit({"kind": "run_finished", "status": summary["status"]})
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "finished_at": time.time(),
                "duration_seconds": time.time() - started,
                "execution_seconds": time.perf_counter() - execution_started,
                "global_step": global_step,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        events.emit(
            {"kind": "run_failed", "error_type": type(exc).__name__, "message": str(exc)}
        )
        atomic_json(run_dir / "summary.json", redact(summary))
        render_report(run_dir)
        raise
    finally:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
    atomic_json(run_dir / "summary.json", redact(summary))
    render_report(run_dir)
    return summary


def resume_job(run_dir: Path) -> dict[str, Any]:
    job_path = run_dir.resolve() / "job.json"
    if not job_path.is_file():
        raise IntegrityError("run directory does not contain job.json")
    return run_job(JobConfig.load(job_path), resume_dir=run_dir.resolve())
