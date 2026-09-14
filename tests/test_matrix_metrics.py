import json

import pytest

from shadow_trainer.matrix_metrics import collect_arm_metrics


def test_collect_arm_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"duration_seconds": 3.0, "execution_seconds": 2.5,
        "global_step": 2, "artifact_bytes": 90,
        "cache": {"occupancy_bytes": 40, "budget_bytes": 64}}))
    rows = [
        {"kind": "staging", "event": "cache_miss", "bytes": 20, "seconds": 0.4,
         "worker": "MainThread", "occupancy_bytes": 20},
        {"kind": "staging", "event": "cache_hit", "seconds": 0.2,
         "worker": "shadow-prefetch_0", "occupancy_bytes": 40},
        {"kind": "prefetch_consumed", "wait_seconds": 0.1},
        {"kind": "train_step", "peak_gpu_allocated_bytes": 10,
         "peak_gpu_reserved_bytes": 16},
        {"kind": "train_step", "peak_gpu_allocated_bytes": 12,
         "peak_gpu_reserved_bytes": 16},
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    metrics = collect_arm_metrics(run)
    assert metrics["peak_gpu_allocated_bytes"] == 12
    assert metrics["peak_gpu_reserved_bytes"] == 16
    assert metrics["bytes_transferred"] == 20
    assert metrics["cache_hits"] == metrics["cache_misses"] == 1
    assert metrics["execution_seconds"] == 2.5
    assert metrics["staging_seconds"] == pytest.approx(0.6)
    assert metrics["foreground_staging_seconds"] == pytest.approx(0.4)
    assert metrics["background_staging_seconds"] == pytest.approx(0.2)
    assert metrics["prefetch_wait_seconds"] == pytest.approx(0.1)
    assert metrics["io_wait_seconds"] == pytest.approx(0.5)
    assert metrics["peak_cache_occupancy_bytes"] == 40
    assert metrics["artifact_bytes"] == 90
