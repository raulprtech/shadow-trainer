"""Portable local fixture and short 3D-CNN demonstration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import JobConfig
from .runtime import run_job


def prepare_demo(output_dir: Path, *, use_cuda: bool = True) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"demo output directory is not empty: {output_dir}")
    fixture = output_dir / "fixture"
    source = fixture / "source"
    source.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in range(6):
        case_id = f"sample_{index:02d}"
        payload = hashlib.sha256(f"shadow-trainer-demo-{index}".encode()).digest() * 32
        file_path = source / f"{case_id}.bin"
        file_path.write_bytes(payload)
        cases.append(
            {
                "case_id": case_id,
                "files": [
                    {
                        "name": "input.bin",
                        "source": file_path.name,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    manifest = fixture / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "shadowtrainer.manifest/v1", "cases": cases}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": "shadowtrainer.job/v1",
        "job_id": "portable-3d-cnn-demo",
        "seed": 20260910,
        "output_dir": str(output_dir / "run"),
        "strategy": "auto",
        "data": {
            "manifest": str(manifest),
            "source": {"type": "local", "root": str(source)},
            "cache_dir": str(output_dir / "cache"),
            "validate_nifti": False,
        },
        "resources": {
            "cache_bytes": 4096,
            "disk_floor_bytes": 0,
            "min_available_ram_bytes": 64 * 2**20,
            "max_swap_bytes": 2**30,
            "min_gpu_free_bytes": 64 * 2**20 if use_cuda else 0,
            "require_cuda": use_cuda,
            "artifact_budget_bytes": 128 * 2**20,
            "disk_check_path": "/mnt/c" if Path("/mnt/c").exists() else str(output_dir),
        },
        "training": {"epochs": 2, "cases_per_window": 2},
        "workload": {
            "type": "tiny3d",
            "options": {
                "device": "cuda" if use_cuda else "cpu",
                "input_shape": [1, 8, 8, 8],
                "classes": 2,
            },
        },
    }
    config_path = output_dir / "demo-job.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def run_demo(output_dir: Path, *, use_cuda: bool = True) -> dict:
    config_path = prepare_demo(output_dir, use_cuda=use_cuda)
    return run_job(JobConfig.load(config_path))
