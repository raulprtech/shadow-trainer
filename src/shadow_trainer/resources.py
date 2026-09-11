"""Host and accelerator telemetry without changing host state."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


def _nvidia_smi() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "reason": "nvidia-smi not found"}
    fields = "name,driver_version,memory.total,memory.free,memory.used,utilization.gpu"
    try:
        result = subprocess.run(
            [binary, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        devices = []
        for index, line in enumerate(result.stdout.strip().splitlines()):
            values = [part.strip() for part in line.split(",")]
            if len(values) != 6:
                continue
            devices.append(
                {
                    "index": index,
                    "name": values[0],
                    "driver_version": values[1],
                    "memory_total_bytes": int(values[2]) * 2**20,
                    "memory_free_bytes": int(values[3]) * 2**20,
                    "memory_used_bytes": int(values[4]) * 2**20,
                    "utilization_percent": int(values[5]),
                }
            )
        return {"available": bool(devices), "devices": devices}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def _torch_environment() -> dict[str, Any]:
    try:
        import torch

        result: dict[str, Any] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            result["device_count"] = torch.cuda.device_count()
            result["device_name"] = torch.cuda.get_device_name(0)
            result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        return result
    except ImportError:
        return {"installed": False, "cuda_available": False}


def snapshot(disk_path: str | Path = "/") -> dict[str, Any]:
    disk = shutil.disk_usage(Path(disk_path))
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process()
    return {
        "schema_version": "shadowtrainer.environment/v1",
        "captured_at": time.time(),
        "host": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "process_rss_bytes": process.memory_info().rss,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
        },
        "disk": {
            "path": str(Path(disk_path).resolve()),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
            "used_bytes": disk.used,
        },
        "nvidia": _nvidia_smi(),
        "torch": _torch_environment(),
        "dependencies": {"rclone": bool(shutil.which("rclone"))},
    }


def gpu_free_bytes(environment: dict[str, Any]) -> int:
    devices = environment.get("nvidia", {}).get("devices", [])
    return int(devices[0]["memory_free_bytes"]) if devices else 0


def print_snapshot(environment: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(environment, indent=2, sort_keys=True))
        return
    mem = environment["memory"]
    disk = environment["disk"]
    gpu = environment["nvidia"].get("devices", [])
    print(f"Platform: {environment['host']['platform']}")
    print(f"RAM available: {mem['available_bytes'] / 2**30:.2f} GiB")
    print(f"Swap used: {mem['swap_used_bytes'] / 2**20:.2f} MiB")
    print(f"Disk free ({disk['path']}): {disk['free_bytes'] / 2**30:.2f} GiB")
    if gpu:
        print(
            f"GPU: {gpu[0]['name']} — {gpu[0]['memory_free_bytes'] / 2**20:.0f} "
            f"of {gpu[0]['memory_total_bytes'] / 2**20:.0f} MiB free"
        )
    else:
        print("GPU: unavailable")
    torch = environment["torch"]
    print(
        "PyTorch: "
        + (f"{torch['version']} (CUDA {torch['cuda_build']})" if torch["installed"] else "not installed")
    )
