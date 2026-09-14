"""Exact semantic audit for independently executed sync/prefetch run pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import IntegrityError
from .events import atomic_json
from .state_digest import state_digest

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
    host = environment.get("host", {})
    torch = environment.get("torch", {})
    nvidia = environment.get("nvidia", {})
    devices = nvidia.get("devices", [])
    identities = [{
        "index": device.get("index"),
        "name": device.get("name"),
        "memory_total_bytes": device.get("memory_total_bytes"),
        "driver_version": device.get("driver_version"),
    } for device in devices if isinstance(device, dict)]
    return {
        "host": {"kernel": host.get("kernel"), "platform": host.get("platform"),
                 "python": host.get("python")},
        "dependencies": environment.get("dependencies", {}),
        "torch": {"version": torch.get("version"),
                  "cuda_available": torch.get("cuda_available"),
                  "cuda_build": torch.get("cuda_build"),
                  "cudnn_version": torch.get("cudnn_version"),
                  "device_count": torch.get("device_count"),
                  "device_name": torch.get("device_name")},
        "nvidia_available": nvidia.get("available"),
        "devices": identities,
    }


def _step_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in SEMANTIC_STEP_KEYS if key in row}
            for row in events if row.get("kind") == "train_step"]



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
    sync_ready = [row for row in sync["events"] if row.get("kind") == "workload_ready"]
    prefetch_ready = [row for row in prefetch["events"] if row.get("kind") == "workload_ready"]
    checks["fresh_runs"] = (len(sync_ready) == len(prefetch_ready) == 1
                            and sync_ready[0].get("resume") is False
                            and prefetch_ready[0].get("resume") is False)
    checks["initial_workload_state"] = (checks["fresh_runs"]
        and sync_ready[0].get("workload_state_sha256")
        == prefetch_ready[0].get("workload_state_sha256"))
    checks["deterministic_algorithms"] = (checks["fresh_runs"]
        and sync_ready[0].get("deterministic_algorithms") is True
        and prefetch_ready[0].get("deterministic_algorithms") is True)
    checks["steps"] = _step_signature(sync["events"]) == _step_signature(prefetch["events"])
    for key in ("next_epoch", "next_window", "global_step"):
        checks[f"checkpoint_{key}"] = sync["checkpoint"].get(key) == prefetch["checkpoint"].get(key)
    sync_digest = state_digest(sync["checkpoint"].get("workload"))
    prefetch_digest = state_digest(prefetch["checkpoint"].get("workload"))
    checks["workload_state"] = sync_digest == prefetch_digest
    failed = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "schema_version": "shadowtrainer.pair-audit/v1",
        "status": "exact" if not failed else "diverged",
        "checks": checks,
        "failed_checks": failed,
        "sync": {"run_id": sync["job"].get("job_id"),
                 "duration_seconds": sync["summary"].get("duration_seconds"),
                 "execution_seconds": sync["summary"].get("execution_seconds"),
                 "initial_workload_state_sha256": sync_ready[0].get("workload_state_sha256") if sync_ready else None,
                 "workload_state_sha256": sync_digest},
        "prefetch": {"run_id": prefetch["job"].get("job_id"),
                     "duration_seconds": prefetch["summary"].get("duration_seconds"),
                     "execution_seconds": prefetch["summary"].get("execution_seconds"),
                     "initial_workload_state_sha256": prefetch_ready[0].get("workload_state_sha256") if prefetch_ready else None,
                     "workload_state_sha256": prefetch_digest},
        "performance_comparison_eligible": not failed,
    }
    if output is not None:
        atomic_json(Path(output).expanduser().resolve(), result)
    return result
