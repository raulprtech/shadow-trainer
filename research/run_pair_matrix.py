#!/usr/bin/env python3
"""Run cold/warm sync-prefetch pairs in independent Python processes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from shadow_trainer.config import JobConfig
from shadow_trainer.events import atomic_json
from shadow_trainer.matrix_metrics import collect_arm_metrics
from shadow_trainer.pair_audit import audit_pair
from shadow_trainer.source_verification import verify_local_manifest
from shadow_trainer.staging import CaseStager, LocalSource, load_manifest

SCHEMA = "shadowtrainer.pair-matrix/v1"
PHYSICAL_DISK = Path("/mnt/c")
DISK_FLOOR = 20 * 2**30
SESSION_CEILING = 3 * 2**30
REPETITIONS = 3


def _bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _physical_free() -> int:
    return shutil.disk_usage(PHYSICAL_DISK).free


def _guard(session: Path | None = None) -> None:
    free = _physical_free()
    if free < DISK_FLOOR:
        raise RuntimeError(f"physical_disk_floor: {free} < {DISK_FLOOR}")
    if session is not None and _bytes(session) > SESSION_CEILING:
        raise RuntimeError(f"session_artifact_ceiling: {_bytes(session)} > {SESSION_CEILING}")


def _absolute_template(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    for key in ("manifest",):
        value = Path(raw["data"][key]).expanduser()
        raw["data"][key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    if raw["data"]["source"]["type"] != "local":
        raise RuntimeError("paired matrix accepts local sources only")
    source = Path(raw["data"]["source"]["root"]).expanduser()
    raw["data"]["source"]["root"] = str(source.resolve() if source.is_absolute() else (root / source).resolve())
    raw["resources"]["disk_check_path"] = str(PHYSICAL_DISK)
    raw["resources"]["disk_floor_bytes"] = DISK_FLOOR
    return raw


def _prepare_job(template: dict, session: Path, workload: str, condition: str,
                 repetition: int, strategy: str) -> Path:
    _guard(session)
    raw = json.loads(json.dumps(template))
    identity = f"{workload}-{condition}-r{repetition}-{strategy}"
    raw["job_id"] = identity
    raw["strategy"] = strategy
    arm = session / workload / condition / f"r{repetition}" / strategy
    raw["output_dir"] = str((arm / "run").resolve())
    raw["data"]["cache_dir"] = str((arm / "cache").resolve())
    if condition == "warm":
        total = sum(record.total_bytes for record in load_manifest(Path(raw["data"]["manifest"])).values())
        raw["resources"]["cache_bytes"] = max(raw["resources"]["cache_bytes"], total)
    arm.mkdir(parents=True, exist_ok=False)
    job = arm / "job.json"
    job.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    if condition == "warm":
        config = JobConfig.load(job)
        records = load_manifest(config.data.manifest)
        CaseStager(config.data.manifest, config.data.cache_dir,
                   config.resources.cache_bytes,
                   LocalSource(Path(config.data.source.root))).stage_batch(list(records))
    _guard(session)
    return job


def _run(job: Path) -> dict:
    _guard(job.parents[4])
    completed = subprocess.run([sys.executable, "-m", "shadow_trainer", "run", str(job)],
                               text=True, capture_output=True, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError(f"run_failed:{job.parent.name}:{completed.stderr[-1000:]}")
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    templates = {
        "tiny3d": root / "pair_jobs" / "tiny3d.json",
        "nifti3d": root / "pair_jobs" / "nifti3d.json",
        "resnet18": root / "pair_jobs" / "resnet18.json",
        "resnet50": root / "pair_jobs" / "resnet50.json",
    }
    preflight = {"physical_free_bytes": _physical_free(), "required_floor_bytes": DISK_FLOOR,
                 "session_ceiling_bytes": SESSION_CEILING, "workloads": list(templates)}
    try:
        _guard()
        verified = {}
        for workload, template_path in templates.items():
            template = _absolute_template(template_path)
            verified[workload] = verify_local_manifest(
                template["data"]["manifest"], template["data"]["source"]["root"]
            )
        preflight["manifests"] = verified
    except Exception as exc:
        print(json.dumps(preflight, indent=2))
        print(f"preflight_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(preflight, indent=2))
    if args.preflight_only:
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is used")
    session = args.output_dir.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": SCHEMA, "started_at": time.time(),
              "preflight": preflight, "protocol": {"workloads": list(templates),
              "conditions": ["cold", "warm"], "repetitions": REPETITIONS,
              "timing_field": "execution_seconds"}, "pairs": [], "status": "running"}
    atomic_json(session / "matrix.json", result)
    try:
        for workload, template_path in templates.items():
            template = _absolute_template(template_path)
            for condition in ("cold", "warm"):
                for repetition in range(1, REPETITIONS + 1):
                    _guard(session)
                    jobs = {strategy: _prepare_job(template, session, workload, condition,
                                                   repetition, strategy)
                            for strategy in ("sync", "prefetch")}
                    summaries = {strategy: _run(job) for strategy, job in jobs.items()}
                    pair_root = jobs["sync"].parent.parent
                    audit_path = pair_root / "pair-audit.json"
                    audit = audit_pair(jobs["sync"].parent / "run",
                                       jobs["prefetch"].parent / "run", audit_path)
                    entry = {"workload": workload, "condition": condition,
                             "repetition": repetition, "audit": audit,
                             "sync_seconds": summaries["sync"]["execution_seconds"],
                             "prefetch_seconds": summaries["prefetch"]["execution_seconds"]}
                    entry["sync_metrics"] = collect_arm_metrics(jobs["sync"].parent / "run")
                    entry["prefetch_metrics"] = collect_arm_metrics(jobs["prefetch"].parent / "run")
                    if audit["status"] == "exact":
                        entry["observed_ratio_sync_over_prefetch"] = (
                            entry["sync_seconds"] / entry["prefetch_seconds"])
                    result["pairs"].append(entry)
                    atomic_json(session / "matrix.json", result)
                    if audit["status"] != "exact":
                        raise RuntimeError(f"pair_diverged:{workload}:{condition}:r{repetition}")
        result["status"] = "success"
    except Exception as exc:
        result["status"] = ("guard_stopped" if any(
            marker in str(exc) for marker in ("floor", "ceiling", "reservation")
        ) else "failed")
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(session / "matrix.json", result)
        raise
    finally:
        result["finished_at"] = time.time()
        result["artifact_bytes"] = _bytes(session)
        atomic_json(session / "matrix.json", result)
    print(session / "matrix.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
