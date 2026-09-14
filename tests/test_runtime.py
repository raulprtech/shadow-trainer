import json
from types import SimpleNamespace

import pytest
import shadow_trainer.runtime as runtime

from shadow_trainer.config import JobConfig
from shadow_trainer.errors import IntegrityError
from shadow_trainer.demo import run_demo
from shadow_trainer.reporting import redact
from shadow_trainer.runtime import resume_job, run_job

from conftest import write_job


def test_cpu_demo_generates_complete_evidence(tmp_path):
    summary = run_demo(tmp_path / "demo", use_cuda=False)
    run = tmp_path / "demo" / "run"
    assert summary["status"] == "success"
    assert summary["global_step"] == 12
    for name in (
        "job.json",
        "plan.json",
        "environment.json",
        "events.jsonl",
        "summary.json",
        "latest.checkpoint.pt",
        "report-summary.json",
        "report.html",
    ):
        assert (run / name).is_file()


def test_interrupted_run_resumes_without_repeating_steps(tmp_path, monkeypatch):
    config = JobConfig.load(write_job(tmp_path))
    first = run_job(config, stop_after_steps=1)
    assert first["status"] == "interrupted"
    assert first["global_step"] == 1
    final = resume_job(tmp_path / "run")
    assert final["status"] == "success"
    assert final["global_step"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    ]
    steps = [row["global_step"] for row in rows if row.get("kind") == "train_step"]
    assert steps == [1, 2, 3]
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _path: SimpleNamespace(free=9))
    with pytest.raises(IntegrityError, match="reservation"):
        runtime._assert_disk_floor(config, reserve_bytes=10)


def test_report_redacts_secret_fields_and_bearer_values():
    value = redact(
        {"runtime_token": "visible", "message": "Authorization: Bearer abc.def"}
    )
    assert value["runtime_token"] == "[REDACTED]"
    assert "abc.def" not in value["message"]
