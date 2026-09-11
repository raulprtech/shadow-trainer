"""Conservative resource admission and backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import JobConfig
from .resources import gpu_free_bytes

# Async staging remains disabled until the frozen equivalence gate passes.
PREFETCH_STABLE = False


@dataclass(frozen=True)
class Plan:
    admitted: bool
    requested_strategy: str
    selected_strategy: Literal["sync", "prefetch"]
    prefetch_status: Literal["stable", "experimental", "disabled"]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": "shadowtrainer.plan/v1",
            "admitted": self.admitted,
            "requested_strategy": self.requested_strategy,
            "selected_strategy": self.selected_strategy,
            "prefetch_status": self.prefetch_status,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def build_plan(config: JobConfig, environment: dict) -> Plan:
    reasons: list[str] = []
    warnings: list[str] = []
    memory = environment["memory"]
    disk = environment["disk"]
    torch = environment["torch"]
    free_gpu = gpu_free_bytes(environment)

    if disk["free_bytes"] < config.resources.disk_floor_bytes + config.resources.cache_bytes:
        reasons.append("disk_headroom")
    if memory["available_bytes"] < config.resources.min_available_ram_bytes:
        reasons.append("available_ram")
    if memory["swap_used_bytes"] > config.resources.max_swap_bytes:
        reasons.append("swap_headroom")
    if config.resources.require_cuda and not torch.get("cuda_available", False):
        reasons.append("cuda_unavailable")
    if config.resources.min_gpu_free_bytes and free_gpu < config.resources.min_gpu_free_bytes:
        reasons.append("gpu_headroom")
    if config.resources.cache_bytes > config.resources.artifact_budget_bytes:
        reasons.append("cache_exceeds_artifact_budget")
    if config.data.source.type == "rclone" and not environment["dependencies"]["rclone"]:
        reasons.append("rclone_unavailable")

    if config.strategy == "sync":
        selected = "sync"
    elif config.strategy == "prefetch":
        selected = "prefetch"
        if not PREFETCH_STABLE:
            warnings.append("prefetch_is_experimental_until_equivalence_gate_passes")
    elif PREFETCH_STABLE:
        selected = "prefetch"
    else:
        selected = "sync"
        warnings.append("auto_selected_sync_pending_prefetch_equivalence_gate")

    return Plan(
        admitted=not reasons,
        requested_strategy=config.strategy,
        selected_strategy=selected,
        prefetch_status="stable" if PREFETCH_STABLE else (
            "experimental" if selected == "prefetch" else "disabled"
        ),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
