import json

import torch

from shadow_trainer.config import JobConfig
from shadow_trainer.runtime import resume_job, run_job

from conftest import write_case_source


def _job(root, source, manifest, name):
    payload = {
        "schema_version": "shadowtrainer.job/v1", "job_id": name, "seed": 73,
        "output_dir": str(root / f"run-{name}"), "strategy": "sync",
        "data": {"manifest": str(manifest), "source": {"type": "local", "root": str(source)},
                 "cache_dir": str(root / f"cache-{name}")},
        "resources": {"cache_bytes": 64, "disk_floor_bytes": 0,
            "min_available_ram_bytes": 0, "max_swap_bytes": 2**63 - 1,
            "min_gpu_free_bytes": 0, "require_cuda": False,
            "artifact_budget_bytes": 16 * 2**20},
        "training": {"epochs": 1, "cases_per_window": 1},
        "workload": {"type": "tiny3d", "options": {"device": "cpu"}},
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload))
    return JobConfig.load(path)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def test_resume_matches_uninterrupted_model_optimizer_and_rng(tmp_path):
    source, manifest = write_case_source(tmp_path)
    baseline = _job(tmp_path, source, manifest, "baseline")
    resumed = _job(tmp_path, source, manifest, "resumed")
    assert run_job(baseline)["status"] == "success"
    assert run_job(resumed, stop_after_steps=1)["status"] == "interrupted"
    assert resume_job(resumed.output_dir)["status"] == "success"
    base_state = torch.load(baseline.output_dir / "latest.checkpoint.pt",
                            map_location="cpu", weights_only=False)["workload"]
    resumed_state = torch.load(resumed.output_dir / "latest.checkpoint.pt",
                               map_location="cpu", weights_only=False)["workload"]
    _assert_equal(base_state, resumed_state)
