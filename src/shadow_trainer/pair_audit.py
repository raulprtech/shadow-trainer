"""Exact semantic audit for independently executed sync/prefetch run pairs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .errors import IntegrityError
from .events import atomic_json

REQUIRED = ("job.json", "plan.json", "environment.json", "events.jsonl",
            "summary.json", "latest.checkpoint.pt")
SEMANTIC_STEP_KEYS = ("epoch", "window", "global_step", "case_id", "finite",
                      "loss", "gradient_norm", "foreground_dice", "patch_start",
                      "patch_size", "parameter_count", "resnet_depth", "input_mode")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read pair artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"pair artifact must be an object: {path}")
    return value


def _events(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("event is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"invalid event at {path}:{number}: {exc}") from exc
    return rows


def _load(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise IntegrityError(f"incomplete pair run {run_dir.name}: {', '.join(missing)}")
    import torch
    return {
        "dir": run_dir,
        "job": _json(run_dir / "job.json"),
        "plan": _json(run_dir / "plan.json"),
        "environment": _json(run_dir / "environment.json"),
        "summary": _json(run_dir / "summary.json"),
        "events": _events(run_dir / "events.jsonl"),
        "checkpoint": torch.load(run_dir / "latest.checkpoint.pt",
                                 map_location="cpu", weights_only=False),
    }


def _manifest_digest(job: dict[str, Any]) -> str:
    raw = job.get("data", {}).get("manifest")
    if not isinstance(raw, str) or not Path(raw).is_file():
        raise IntegrityError("pair job references an unavailable manifest")
    value = _json(Path(raw))
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _environment_signature(environment: dict[str, Any]) -> dict[str, Any]:
    torch = environment.get("torch", {})
    nvidia = environment.get("nvidia", {})
    devices = nvidia.get("devices", [])
    device = devices[0] if devices else {}
    return {
        "torch": torch.get("version"), "cuda": torch.get("cuda_build"),
        "cudnn": torch.get("cudnn_version"), "gpu": device.get("name"),
        "gpu_total_bytes": device.get("memory_total_bytes"),
        "driver": device.get("driver_version"),
    }


def _step_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in SEMANTIC_STEP_KEYS if key in row}
            for row in events if row.get("kind") == "train_step"]


def _state_digest(value: Any) -> str:
    import torch
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda x: repr(x)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        elif isinstance(item, float):
            if not math.isfinite(item):
                digest.update(repr(item).encode())
            else:
                digest.update(item.hex().encode())
            digest.update(b"\0")
        else:
            digest.update(type(item).__name__.encode() + b":" + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def audit_pair(sync_dir: str | Path, prefetch_dir: str | Path,
               output: str | Path | None = None) -> dict[str, Any]:
    sync = _load(Path(sync_dir))
    prefetch = _load(Path(prefetch_dir))
    checks: dict[str, bool] = {}
    checks["declared_strategies"] = (
        sync["summary"].get("selected_strategy") == "sync"
        and prefetch["summary"].get("selected_strategy") == "prefetch"
    )
    for key in ("seed", "training", "workload"):
        checks[f"job_{key}"] = sync["job"].get(key) == prefetch["job"].get(key)
    checks["manifest"] = _manifest_digest(sync["job"]) == _manifest_digest(prefetch["job"])
    checks["environment"] = (_environment_signature(sync["environment"])
                             == _environment_signature(prefetch["environment"]))
    checks["success"] = (sync["summary"].get("status") == "success"
                         and prefetch["summary"].get("status") == "success")
    checks["steps"] = _step_signature(sync["events"]) == _step_signature(prefetch["events"])
    for key in ("next_epoch", "next_window", "global_step"):
        checks[f"checkpoint_{key}"] = sync["checkpoint"].get(key) == prefetch["checkpoint"].get(key)
    sync_digest = _state_digest(sync["checkpoint"].get("workload"))
    prefetch_digest = _state_digest(prefetch["checkpoint"].get("workload"))
    checks["workload_state"] = sync_digest == prefetch_digest
    failed = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "schema_version": "shadowtrainer.pair-audit/v1",
        "status": "exact" if not failed else "diverged",
        "checks": checks,
        "failed_checks": failed,
        "sync": {"run_id": sync["job"].get("job_id"),
                 "duration_seconds": sync["summary"].get("duration_seconds"),
                 "workload_state_sha256": sync_digest},
        "prefetch": {"run_id": prefetch["job"].get("job_id"),
                     "duration_seconds": prefetch["summary"].get("duration_seconds"),
                     "workload_state_sha256": prefetch_digest},
        "performance_comparison_eligible": not failed,
    }
    if output is not None:
        atomic_json(Path(output).expanduser().resolve(), result)
    return result
