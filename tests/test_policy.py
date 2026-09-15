from shadow_trainer.config import JobConfig
from shadow_trainer.policy import PREFETCH_STABLE, build_plan

from conftest import write_job


def environment(*, disk_free=10**12, ram=10**12, swap=0, cuda=True, gpu=10**12):
    return {
        "disk": {"free_bytes": disk_free},
        "memory": {"available_bytes": ram, "swap_used_bytes": swap},
        "torch": {"cuda_available": cuda},
        "nvidia": {"devices": [{"memory_free_bytes": gpu}] if gpu else []},
        "dependencies": {"rclone": True},
    }


def test_auto_selects_prefetch_after_equivalence_gate(tmp_path):
    config = JobConfig.load(write_job(tmp_path))
    plan = build_plan(config, environment())
    assert PREFETCH_STABLE
    assert plan.admitted
    assert plan.selected_strategy == "prefetch"
    assert plan.prefetch_status == "stable"
    assert plan.warnings == ()


def test_disk_rejection_is_deterministic(tmp_path):
    config = JobConfig.load(write_job(tmp_path, disk_floor=100))
    plan = build_plan(config, environment(disk_free=120))
    assert not plan.admitted
    assert plan.reasons == ("disk_headroom",)
    required = config.resources.disk_floor_bytes + config.resources.artifact_budget_bytes
    assert not build_plan(config, environment(disk_free=required - 1)).admitted
    assert build_plan(config, environment(disk_free=required)).admitted


def test_cuda_rejection(tmp_path):
    config = JobConfig.load(write_job(tmp_path, require_cuda=True))
    plan = build_plan(config, environment(cuda=False, gpu=0))
    assert not plan.admitted
    assert "cuda_unavailable" in plan.reasons
