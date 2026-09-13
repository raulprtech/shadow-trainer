import json

import pytest

from shadow_trainer.errors import IntegrityError
from shadow_trainer.scientific import build_evidence_bundle


def _write_run(root):
    run = root / "run"
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"workload": {"type": "tiny3d"}}))
    (run / "plan.json").write_text(json.dumps({"selected_strategy": "sync"}))
    (run / "environment.json").write_text(json.dumps({"nvidia": {"devices": [{"name": "Test GPU"}]}}))
    (run / "summary.json").write_text(json.dumps({"status": "success", "global_step": 1,
        "expected_steps": 1, "duration_seconds": 2.0, "selected_strategy": "sync",
        "cache": {"budget_bytes": 20, "occupancy_bytes": 10}}))
    (run / "events.jsonl").write_text("\n".join([
        json.dumps({"kind": "staging", "event": "cache_miss", "bytes": 10, "seconds": 0.5}),
        json.dumps({"kind": "train_step", "loss": 0.25, "finite": True,
                    "peak_gpu_allocated_bytes": 5, "peak_gpu_reserved_bytes": 8})]) + "\n")
    return run


def test_build_bundle_is_traceable(tmp_path):
    run = _write_run(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": "shadowtrainer.evidence-spec/v1",
        "records": [{"id": "r1", "label": "Run 1", "run_dir": str(run),
                     "classification": "valid", "permitted_use": "test only"}]}))
    payload = build_evidence_bundle(spec, tmp_path / "out")
    row = payload["records"][0]
    assert len(row["bundle_sha256"]) == 64
    assert row["bytes_transferred"] == 10
    assert row["all_steps_finite"] is True
    assert "Run 1" in (tmp_path / "out" / "figures" / "peak-vram.svg").read_text()


def test_incomplete_run_is_rejected(tmp_path):
    run = _write_run(tmp_path)
    (run / "events.jsonl").unlink()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": "shadowtrainer.evidence-spec/v1",
        "records": [{"id": "broken", "run_dir": str(run), "classification": "invalid"}]}))
    with pytest.raises(IntegrityError, match="incomplete run bundle"):
        build_evidence_bundle(spec, tmp_path / "out")
