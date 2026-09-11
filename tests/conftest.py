import json
from pathlib import Path


def write_case_source(root: Path, count: int = 3, size: int = 16):
    import hashlib

    source = root / "source"
    source.mkdir()
    cases = []
    for index in range(count):
        case_id = f"case_{index}"
        payload = bytes([index + 1]) * size
        name = f"{case_id}.bin"
        (source / name).write_bytes(payload)
        cases.append(
            {
                "case_id": case_id,
                "files": [
                    {
                        "name": "input.bin",
                        "source": name,
                        "size_bytes": size,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return source, manifest


def write_job(root: Path, *, require_cuda: bool = False, disk_floor: int = 0):
    source, manifest = write_case_source(root)
    payload = {
        "schema_version": "shadowtrainer.job/v1",
        "job_id": "test-job",
        "seed": 42,
        "output_dir": str(root / "run"),
        "strategy": "auto",
        "data": {
            "manifest": str(manifest),
            "source": {"type": "local", "root": str(source)},
            "cache_dir": str(root / "cache"),
        },
        "resources": {
            "cache_bytes": 64,
            "disk_floor_bytes": disk_floor,
            "min_available_ram_bytes": 0,
            "max_swap_bytes": 2**63 - 1,
            "min_gpu_free_bytes": 0,
            "require_cuda": require_cuda,
            "artifact_budget_bytes": 16 * 2**20,
        },
        "training": {"epochs": 1, "cases_per_window": 1},
        "workload": {
            "type": "tiny3d",
            "options": {"device": "cuda" if require_cuda else "cpu"},
        },
    }
    path = root / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
