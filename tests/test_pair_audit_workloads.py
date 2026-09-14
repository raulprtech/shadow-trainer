import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from shadow_trainer.config import JobConfig
from shadow_trainer.pair_audit import audit_pair
from shadow_trainer.runtime import run_job

from conftest import write_case_source


def _nifti_source(root):
    source = root / "nifti-source"
    source.mkdir()
    cases = []
    for index in range(2):
        case_id = f"nifti_{index}"
        case = source / case_id
        case.mkdir()
        image = np.arange(20**3, dtype=np.float32).reshape(20, 20, 20) + index
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[8:12, 8:12, 8:12] = 1
        nib.save(nib.Nifti1Image(image, np.eye(4)), case / "imaging.nii.gz")
        nib.save(nib.Nifti1Image(label, np.eye(4)), case / "segmentation.nii.gz")
        files = []
        for name in ("imaging.nii.gz", "segmentation.nii.gz"):
            path = case / name
            files.append({"name": name, "source": f"{case_id}/{name}",
                          "size_bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        cases.append({"case_id": case_id, "files": files})
    manifest = root / "nifti-manifest.json"
    manifest.write_text(json.dumps({"cases": cases}))
    return source, manifest


def _config(root, source, manifest, strategy, kind, options):
    identity = f"{kind}-{strategy}"
    payload = {
        "schema_version": "shadowtrainer.job/v1", "job_id": identity, "seed": 101,
        "output_dir": str(root / f"run-{identity}"), "strategy": strategy,
        "data": {"manifest": str(manifest), "source": {"type": "local", "root": str(source)},
                 "cache_dir": str(root / f"cache-{identity}"),
                 "validate_nifti": kind == "nifti_patch3d"},
        "resources": {"cache_bytes": 2**20, "disk_floor_bytes": 0,
            "min_available_ram_bytes": 0, "max_swap_bytes": 2**63 - 1,
            "min_gpu_free_bytes": 0, "require_cuda": False,
            "artifact_budget_bytes": 256 * 2**20},
        "training": {"epochs": 1, "cases_per_window": 1, "max_steps": 2},
        "workload": {"type": kind, "options": options},
    }
    path = root / f"{identity}.json"
    path.write_text(json.dumps(payload))
    return JobConfig.load(path)


@pytest.mark.parametrize("kind,options", [
    ("resnet2p5d", {"device": "cpu", "depth": 18, "input_size": 32,
                     "base_width": 8, "classes": 3}),
    ("resnet2p5d", {"device": "cpu", "depth": 50, "input_size": 32,
                     "base_width": 8, "classes": 3}),
    ("nifti_patch3d", {"device": "cpu", "patch_size": 16,
                       "base_channels": 2, "classes": 4, "amp": False}),
])
def test_sync_prefetch_exact_for_each_adapter(tmp_path, kind, options):
    if kind == "nifti_patch3d":
        source, manifest = _nifti_source(tmp_path)
    else:
        source, manifest = write_case_source(tmp_path, count=2, size=32)
    sync = _config(tmp_path, source, manifest, "sync", kind, options)
    prefetch = _config(tmp_path, source, manifest, "prefetch", kind, options)
    assert run_job(sync)["status"] == "success"
    assert run_job(prefetch)["status"] == "success"
    result = audit_pair(sync.output_dir, prefetch.output_dir)
    assert result["status"] == "exact", result["failed_checks"]
