"""Bounded physical CUDA demo acceptance for the installed MVP (explicit use)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
from shadow_trainer.events import atomic_json

GIB = 2**30
MIB = 2**20


def resources(pid=None):
    rss = 0
    if pid:
        try:
            parent = psutil.Process(pid)
            for item in [parent, *parent.children(recursive=True)]:
                try:
                    rss += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
    return {"time": time.time(), "disk_free_bytes": shutil.disk_usage("/mnt/c").free,
            "ram_available_bytes": psutil.virtual_memory().available,
            "swap_used_bytes": psutil.swap_memory().used, "tree_rss_bytes": rss}


def guard(sample, initial=False):
    if sample["disk_free_bytes"] < 20 * GIB + (128 * MIB if initial else 0):
        raise RuntimeError("physical_disk_floor")
    if sample["ram_available_bytes"] < (1536 * MIB if initial else 512 * MIB):
        raise RuntimeError("available_ram_floor")
    if sample["swap_used_bytes"] > 256 * MIB or sample["tree_rss_bytes"] > 4.5 * GIB:
        raise RuntimeError("swap_or_rss_limit")


def main(output):
    guard(resources(), initial=True)
    compute = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=15)
    if compute.stdout.strip():
        raise RuntimeError("GPU compute process already present; do not interfere")
    cli = str(Path(sys.executable).parent / "shadow-trainer")
    doctor = subprocess.run([cli, "doctor", "--json", "--disk-path", "/mnt/c"],
                            capture_output=True, text=True, check=True, timeout=45)
    environment = json.loads(doctor.stdout)
    if not environment["torch"]["cuda_available"]:
        raise RuntimeError("physical CUDA unavailable")
    if environment["nvidia"]["devices"][0]["memory_free_bytes"] < 512 * MIB:
        raise RuntimeError("insufficient GPU headroom")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "doctor.json", environment)
    started = time.monotonic()
    process = None
    outcome = {"status": "running", "scope": "synthetic_physical_cuda_demo"}
    try:
        with (output / "console.log").open("x") as log, (output / "resources.jsonl").open("x") as samples:
            process = subprocess.Popen([cli, "demo", "--output-dir", str(output / "demo")],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                sample = resources(process.pid)
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                guard(sample)
                if time.monotonic() - started > 180:
                    raise RuntimeError("demo_time_budget")
                if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 128 * MIB:
                    raise RuntimeError("demo_artifact_budget")
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        if process.returncode:
            raise RuntimeError(f"demo_exit_{process.returncode}")
        run = output / "demo/run"
        summary = json.loads((run / "summary.json").read_text())
        rows = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        steps = [r for r in rows if r.get("kind") == "train_step"]
        assert summary["status"] == "success" and summary["selected_strategy"] == "sync"
        assert [r["global_step"] for r in steps] == list(range(1, 13))
        assert all(r["finite"] and math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in steps)
        assert max(r["peak_gpu_allocated_bytes"] for r in steps) > 0
        assert all(r["occupancy_bytes"] <= r["budget_bytes"] == 4096
                   for r in rows if r.get("kind") == "staging")
        files = ["job.json", "plan.json", "environment.json", "events.jsonl", "summary.json",
                 "latest.checkpoint.pt", "report-summary.json", "report.html"]
        artifacts = {}
        for name in files:
            data = (run / name).read_bytes()
            artifacts[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        outcome.update(status="success", steps=12, strategy="sync",
                       peak_gpu_allocated_bytes=max(r["peak_gpu_allocated_bytes"] for r in steps),
                       peak_gpu_reserved_bytes=max(r["peak_gpu_reserved_bytes"] for r in steps),
                       artifacts=artifacts)
    except Exception as error:
        outcome.update(status="failed", error=type(error).__name__ + ":" + str(error))
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        outcome["wall_seconds"] = time.monotonic() - started
        outcome["final_resources"] = resources()
        atomic_json(output / "acceptance.json", outcome)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
