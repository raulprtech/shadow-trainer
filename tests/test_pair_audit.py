import json

import torch

from shadow_trainer.config import JobConfig
from shadow_trainer.pair_audit import audit_pair
from shadow_trainer.runtime import run_job

from conftest import write_case_source


def _config(root, source, manifest, strategy):
    payload = {
        "schema_version": "shadowtrainer.job/v1", "job_id": f"pair-{strategy}",
        "seed": 91, "output_dir": str(root / f"run-{strategy}"), "strategy": strategy,
        "data": {"manifest": str(manifest), "source": {"type": "local", "root": str(source)},
                 "cache_dir": str(root / f"cache-{strategy}")},
        "resources": {"cache_bytes": 64, "disk_floor_bytes": 0,
            "min_available_ram_bytes": 0, "max_swap_bytes": 2**63 - 1,
            "min_gpu_free_bytes": 0, "require_cuda": False,
            "artifact_budget_bytes": 16 * 2**20},
        "training": {"epochs": 2, "cases_per_window": 1},
        "workload": {"type": "tiny3d", "options": {"device": "cpu"}},
    }
    path = root / f"{strategy}.json"
    path.write_text(json.dumps(payload))
    return JobConfig.load(path)


def test_exact_pair_and_mutated_checkpoint(tmp_path):
    source, manifest = write_case_source(tmp_path)
    sync = _config(tmp_path, source, manifest, "sync")
    prefetch = _config(tmp_path, source, manifest, "prefetch")
    run_job(sync)
    run_job(prefetch)
    output = tmp_path / "audit.json"
    exact = audit_pair(sync.output_dir, prefetch.output_dir, output)
    assert exact["status"] == "exact"
    assert exact["performance_comparison_eligible"] is True
    checkpoint_path = prefetch.output_dir / "latest.checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first = next(iter(checkpoint["workload"]["model"].values()))
    first.view(-1)[0] += 1
    torch.save(checkpoint, checkpoint_path)
    diverged = audit_pair(sync.output_dir, prefetch.output_dir)
    assert diverged["status"] == "diverged"
    assert diverged["failed_checks"] == ["workload_state"]
