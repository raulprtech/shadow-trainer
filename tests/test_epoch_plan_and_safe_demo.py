import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shadow_trainer import demo
from shadow_trainer import cli
from shadow_trainer.config import JobConfig

spec = importlib.util.spec_from_file_location(
    "campaign_epoch_plan", Path(__file__).parents[1] / "research/campaign_epoch_plan.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_new_plan_then_resume_never_reduces_epochs(tmp_path):
    state = {}
    assert module.frozen_epochs(state, tmp_path, 4) == 4
    assert module.frozen_epochs(state, tmp_path, 2) == 4
    assert state == {"epochs_selected": 4}


def test_resume_never_increases_epochs(tmp_path):
    assert module.frozen_epochs({"epochs_selected": 2}, tmp_path, 4) == 2


def test_conflicting_legacy_arms_fail_without_mutation(tmp_path):
    arm = tmp_path / "arm_A"
    arm.mkdir()
    (arm / "summary.json").write_text(json.dumps({"epochs_requested": 4}))
    state = {"epochs_selected": 2}
    with pytest.raises(ValueError, match="conflicts"):
        module.frozen_epochs(state, tmp_path, 4)
    assert state == {"epochs_selected": 2}
    with pytest.raises(ValueError, match="missing"):
        module.frozen_epochs({}, tmp_path, 4)


@pytest.mark.parametrize("value", [True, 0, 3, "4"])
def test_invalid_frozen_epochs_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        module.frozen_epochs({"epochs_selected": value}, tmp_path, 4)


def test_safe_demo_prepare_no_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * 2**30))
    run = Mock(side_effect=AssertionError("training must not start"))
    monkeypatch.setattr(cli, "run_demo", run)
    output = tmp_path / "demo"
    assert cli.main(["demo", "--cpu", "--prepare-only", "--output-dir", str(output)]) == 0
    config = JobConfig.load(output / "demo-job.json")
    assert config.resources.disk_floor_bytes == 20 * 2**30
    assert config.resources.min_available_ram_bytes == 1536 * 2**20
    assert config.resources.max_swap_bytes == 256 * 2**20
    assert not config.resources.require_cuda
    assert not config.output_dir.exists()
    assert not config.data.cache_dir.exists()
    run.assert_not_called()


def test_low_disk_rejected_before_creating_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 2**30))
    output = tmp_path / "demo"
    with pytest.raises(RuntimeError, match="headroom"):
        demo.prepare_demo(output, use_cuda=False)
    assert not output.exists()


def test_existing_directory_or_symlink_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="already exists"):
        demo.prepare_demo(tmp_path, use_cuda=False)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="symlink"):
        demo.prepare_demo(link, use_cuda=False)
