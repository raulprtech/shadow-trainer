"""Offline process-isolated bridge smoke; synthetic CPU data only.

No imports from Clinical-Nigma or Shadow Trainer enter this harness process.
Use a fresh output directory; existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

GIB = 2**30
FLOOR = 20 * GIB
SESSION_LIMIT = 64 * 2**20


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load(path: Path):
    return json.loads(path.read_text())


def phase_log_path(session: Path, name: str) -> Path:
    return session / ("phase-" + name + ".json")


def guard() -> dict:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    free = shutil.disk_usage("/mnt/c").free
    if free < FLOOR + SESSION_LIMIT:
        raise RuntimeError("physical C: disk floor plus session reservation not available")
    if memory["MemAvailable"] < 1536 * 2**20:
        raise RuntimeError("available RAM is below 1.5 GiB")
    if memory["SwapTotal"] - memory["SwapFree"] > 256 * 2**20:
        raise RuntimeError("swap usage exceeds 256 MiB")
    return {"disk_free_bytes": free, "ram_available_bytes": memory["MemAvailable"],
            "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"]}


def child_env(root: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in
           ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR")
           if key in os.environ}
    env.update(PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-root", type=Path, required=True)
    parser.add_argument("--clinical-python", type=Path, required=True)
    parser.add_argument("--shadow-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    resources = guard()
    if args.preflight_only:
        print(json.dumps(resources, indent=2))
        return 0
    session = args.output_dir.absolute()
    if session.exists() or session.is_symlink():
        raise RuntimeError("output directory already exists; use a fresh directory")
    for interpreter in (args.clinical_python, args.shadow_python):
        if not interpreter.is_file():
            raise RuntimeError("a required existing interpreter is unavailable")
    shadow_root = Path(__file__).resolve().parents[1]
    clinical_root = args.clinical_root.resolve()
    session.mkdir(parents=True)
    source = session / "source"
    source.mkdir()
    phases = []
    checks = []
    summary = {"schema_version": "shadowtrainer.bridge-smoke/v1",
               "classification": "synthetic_cpu_integration_only",
               "status": "running", "preflight": resources,
               "training_steps_limit": 2, "cuda_visible_devices": "",
               "phases": phases, "checks": checks}

    def run(name, command, root, expected=0):
        guard()
        started = time.monotonic()
        result = subprocess.run([str(x) for x in command], cwd=session,
                                env=child_env(root), capture_output=True,
                                text=True, timeout=120, check=False)
        save(phase_log_path(session, name), {
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "duration_seconds": time.monotonic() - started,
        })
        phases.append({"name": name, "returncode": result.returncode})
        if result.returncode != expected:
            raise RuntimeError(f"{name} exit {result.returncode}, expected {expected}; see phase log")
        used = sum(p.stat().st_size for p in session.rglob("*") if p.is_file())
        if used > SESSION_LIMIT:
            raise RuntimeError("session artifact budget exceeded")
        return result

    clinical = [args.clinical_python, "-m", "nigma.cli", "shadow-trainer"]
    shadow = [args.shadow_python, "-m", "shadow_trainer"]
    try:
        cases = []
        for i in range(2):
            file = source / f"synthetic_{i}.bin"
            file.write_bytes(bytes([i + 1]) * 16)
            cases.append({"case_id": f"synthetic_{i}", "files": [
                {"name": "input.bin", "source": file.name,
                 "size_bytes": 16, "sha256": digest(file)}]})
        manifest = session / "manifest.json"
        save(manifest, {"schema_version": "shadowtrainer.manifest/v1", "cases": cases})
        experiment = {
            "schema_version": "clinical-nigma.experiment/v1",
            "experiment_id": "synthetic-bridge-round2",
            "objective": "Verify offline process boundaries on fabricated data",
            "disease": "not_applicable_synthetic", "endpoint": "interface_conformance",
            "prediction_time": "not_applicable_synthetic", "modalities": ["synthetic"],
            "execution_split": "development",
            "splits": {"train_ref": "split://synthetic/train/v1",
                       "development_ref": "split://synthetic/development/v1",
                       "locked_test_ref": "split://synthetic/locked-test/v1"},
            "seed": 20260919, "variant_id": "tiny3d-synthetic-v1",
            "constraints": {
                "cache_bytes": 64, "disk_floor_bytes": FLOOR,
                "min_available_ram_bytes": 1536 * 2**20,
                "max_swap_bytes": 256 * 2**20, "min_gpu_free_bytes": 0,
                "artifact_budget_bytes": 16 * 2**20, "require_cuda": False,
                "strategy": "sync",
            },
        }
        binding = {
            "schema_version": "clinical-nigma.shadow-runtime/v1",
            "split_ref": "split://synthetic/development/v1",
            "manifest": "manifest.json", "manifest_sha256": digest(manifest),
            "source_type": "local", "source_root": "source", "cache_dir": "cache",
            "output_dir": "run", "disk_check_path": "/mnt/c",
            "workload_type": "tiny3d",
            "workload_options": {"device": "cpu", "input_shape": [1, 4, 4, 4]},
            "epochs": 1, "cases_per_window": 1, "max_steps": 2,
        }
        save(session / "experiment.json", experiment)
        save(session / "binding.json", binding)
        run("build", clinical + ["job-build", "--experiment", "experiment.json",
                                "--runtime", "binding.json", "--output", "job.json", "--json"],
            clinical_root)
        run("plan", clinical + ["plan", "--job", "job.json", "--shadow-root", shadow_root,
                               "--python", args.shadow_python, "--json"], clinical_root)
        plan = json.loads(load(phase_log_path(session, "plan"))["stdout"])["result"]
        assert plan["plan"]["admitted"] and plan["plan"]["selected_strategy"] == "sync"
        checks.append("real_plan_admitted_sync")
        run("execute", shadow + ["run", "job.json"], shadow_root)
        actual = load(session / "run" / "summary.json")
        assert actual["status"] == "success" and actual["global_step"] == 2
        assert actual["selected_strategy"] == "sync"
        events = [json.loads(line) for line in
                  (session / "run" / "events.jsonl").read_text().splitlines()]
        steps = [e for e in events if e.get("kind") == "train_step"]
        assert [e["global_step"] for e in steps] == [1, 2]
        for step in steps:
            assert math.isfinite(step["loss"])
            assert step["peak_gpu_allocated_bytes"] == 0
            assert step["peak_gpu_reserved_bytes"] == 0
        checks.append("two_confirmed_finite_cpu_steps")
        run("report", shadow + ["report", "run"], shadow_root)
        assert (session / "run" / "report.html").stat().st_size > 0
        run("receipt", clinical + ["receipt-build", "--experiment", "experiment.json",
                                  "--run-dir", "run", "--output", "receipt.json", "--json"],
            clinical_root)
        receipt = load(session / "receipt.json")
        assert receipt["run_status"] == "success"
        assert receipt["decision"] == "pending_human_review"
        assert receipt["manifest_sha256"] == digest(manifest)
        for artifact in receipt["artifacts"]:
            path = session / "run" / artifact["path"]
            assert path.resolve().is_relative_to((session / "run").resolve())
            assert digest(path) == artifact["sha256"]
            assert path.stat().st_size == artifact["size_bytes"]
        checks.append("receipt_artifact_hashes_and_pending_human_review")
        before = {p.relative_to(session / "run").as_posix(): digest(p)
                  for p in (session / "run").rglob("*") if p.is_file()}
        run("duplicate", shadow + ["run", "job.json"], shadow_root, expected=1)
        after = {p.relative_to(session / "run").as_posix(): digest(p)
                 for p in (session / "run").rglob("*") if p.is_file()}
        assert before == after
        checks.append("duplicate_run_rejected_without_evidence_changes")
        bad = copy.deepcopy(experiment)
        bad["execution_split"] = "locked_test"
        save(session / "locked-experiment.json", bad)
        run("locked_test", clinical + ["job-build", "--experiment", "locked-experiment.json",
                                      "--runtime", "binding.json", "--output", "forbidden.json",
                                      "--json"], clinical_root, expected=1)
        assert not (session / "forbidden.json").exists()
        checks.append("locked_test_rejected")
        denied = load(session / "job.json")
        denied["resources"]["disk_floor_bytes"] = 2**63 - 1
        save(session / "denied-job.json", denied)
        run("disk_rejection", shadow + ["plan", "denied-job.json", "--json"], shadow_root, expected=2)
        checks.append("disk_rejected_before_execution")
        shutil.copytree(session / "run", session / "tampered-run")
        tampered = load(session / "tampered-run" / "summary.json")
        tampered["job_id"] = "different-synthetic-job"
        save(session / "tampered-run" / "summary.json", tampered)
        run("tamper_rejection", clinical + ["receipt-build", "--experiment", "experiment.json",
                                          "--run-dir", "tampered-run", "--output", "forbidden-receipt.json",
                                          "--json"], clinical_root, expected=1)
        assert not (session / "forbidden-receipt.json").exists()
        checks.append("tampered_summary_receipt_rejected")
        summary.update(status="success", run_digest=receipt["run_digest"],
                       manifest_sha256=receipt["manifest_sha256"],
                       experiment_digest=receipt["experiment_digest"])
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        summary["session_bytes_before_summary"] = sum(
            p.stat().st_size for p in session.rglob("*") if p.is_file())
        save(session / "bridge-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
