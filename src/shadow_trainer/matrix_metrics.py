"""Derive per-arm resource metrics for paired matrix orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import IntegrityError


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def collect_arm_metrics(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    try:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot derive arm metrics: {exc}") from exc
    train = [row for row in events if row.get("kind") == "train_step"]
    staging = [row for row in events if row.get("kind") == "staging"]
    allocated = _finite(train, "peak_gpu_allocated_bytes")
    reserved = _finite(train, "peak_gpu_reserved_bytes")
    stage_times = _finite(staging, "seconds")
    foreground_stage_times = _finite(
        [row for row in staging if not str(row.get("worker", "")).startswith("shadow-prefetch")],
        "seconds",
    )
    background_stage_times = _finite(
        [row for row in staging if str(row.get("worker", "")).startswith("shadow-prefetch")],
        "seconds",
    )
    prefetch_wait_times = _finite(
        [row for row in events if row.get("kind") == "prefetch_consumed"], "wait_seconds"
    )
    cache_occupancies = _finite(staging, "occupancy_bytes")
    cache = summary.get("cache") if isinstance(summary.get("cache"), dict) else {}
    return {
        "duration_seconds": summary.get("duration_seconds"),
        "execution_seconds": summary.get("execution_seconds"),
        "steps": summary.get("global_step"),
        "peak_gpu_allocated_bytes": int(max(allocated)) if allocated else None,
        "peak_gpu_reserved_bytes": int(max(reserved)) if reserved else None,
        "staging_seconds": sum(stage_times),
        "foreground_staging_seconds": sum(foreground_stage_times),
        "background_staging_seconds": sum(background_stage_times),
        "prefetch_wait_seconds": sum(prefetch_wait_times),
        "io_wait_seconds": sum(foreground_stage_times) + sum(prefetch_wait_times),
        "bytes_transferred": int(sum(row.get("bytes", 0) for row in staging
                                     if row.get("event") == "cache_miss")),
        "cache_hits": sum(row.get("event") == "cache_hit" for row in staging),
        "cache_misses": sum(row.get("event") == "cache_miss" for row in staging),
        "cache_occupancy_bytes": cache.get("occupancy_bytes"),
        "peak_cache_occupancy_bytes": (int(max(cache_occupancies))
                                         if cache_occupancies else cache.get("occupancy_bytes")),
        "cache_budget_bytes": cache.get("budget_bytes"),
        "artifact_bytes": summary.get("artifact_bytes"),
    }
