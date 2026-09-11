"""Versioned job configuration and strict validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

SCHEMA_VERSION = "shadowtrainer.job/v1"
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ConfigurationError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


@dataclass(frozen=True)
class SourceConfig:
    type: Literal["local", "rclone"]
    root: str


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    source: SourceConfig
    cache_dir: Path
    validate_nifti: bool


@dataclass(frozen=True)
class ResourceLimits:
    cache_bytes: int
    disk_floor_bytes: int
    min_available_ram_bytes: int
    max_swap_bytes: int
    min_gpu_free_bytes: int
    require_cuda: bool
    artifact_budget_bytes: int
    disk_check_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    cases_per_window: int
    max_steps: int | None


@dataclass(frozen=True)
class WorkloadConfig:
    type: str
    options: dict[str, Any]


@dataclass(frozen=True)
class JobConfig:
    schema_version: str
    job_id: str
    seed: int
    output_dir: Path
    strategy: Literal["sync", "prefetch", "auto"]
    data: DataConfig
    resources: ResourceLimits
    training: TrainingConfig
    workload: WorkloadConfig
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "JobConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read job config: {exc}") from exc
        root = config_path.parent
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ConfigurationError(f"schema_version must be {SCHEMA_VERSION!r}")
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ConfigurationError("job_id contains unsupported characters")
        strategy = raw.get("strategy", "auto")
        if strategy not in {"sync", "prefetch", "auto"}:
            raise ConfigurationError("strategy must be sync, prefetch, or auto")

        data = _mapping(raw.get("data"), "data")
        source = _mapping(data.get("source"), "data.source")
        source_type = source.get("type")
        if source_type not in {"local", "rclone"}:
            raise ConfigurationError("data.source.type must be local or rclone")
        source_root = source.get("root")
        if not isinstance(source_root, str) or not source_root.strip():
            raise ConfigurationError("data.source.root must be a non-empty string")

        resources = _mapping(raw.get("resources"), "resources")
        training = _mapping(raw.get("training"), "training")
        workload = _mapping(raw.get("workload"), "workload")
        workload_type = workload.get("type")
        if not isinstance(workload_type, str) or not workload_type:
            raise ConfigurationError("workload.type must be a non-empty string")

        def relative(value: Any, name: str) -> Path:
            if not isinstance(value, str) or not value:
                raise ConfigurationError(f"{name} must be a path string")
            candidate = Path(value).expanduser()
            return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        max_steps = training.get("max_steps")
        if max_steps is not None:
            max_steps = _positive(max_steps, "training.max_steps")
        output_dir = relative(raw.get("output_dir"), "output_dir")
        manifest_path = relative(data.get("manifest"), "data.manifest")
        cache_dir = relative(data.get("cache_dir"), "data.cache_dir")
        disk_check_raw = resources.get("disk_check_path", str(cache_dir))
        return cls(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            seed=_positive(raw.get("seed", 1), "seed", allow_zero=True),
            output_dir=output_dir,
            strategy=strategy,
            data=DataConfig(
                manifest=manifest_path,
                source=SourceConfig(type=source_type, root=source_root),
                cache_dir=cache_dir,
                validate_nifti=bool(data.get("validate_nifti", False)),
            ),
            resources=ResourceLimits(
                cache_bytes=_positive(resources.get("cache_bytes"), "resources.cache_bytes"),
                disk_floor_bytes=_positive(
                    resources.get("disk_floor_bytes", 0),
                    "resources.disk_floor_bytes",
                    allow_zero=True,
                ),
                min_available_ram_bytes=_positive(
                    resources.get("min_available_ram_bytes", 0),
                    "resources.min_available_ram_bytes",
                    allow_zero=True,
                ),
                max_swap_bytes=_positive(
                    resources.get("max_swap_bytes", 2**63 - 1),
                    "resources.max_swap_bytes",
                    allow_zero=True,
                ),
                min_gpu_free_bytes=_positive(
                    resources.get("min_gpu_free_bytes", 0),
                    "resources.min_gpu_free_bytes",
                    allow_zero=True,
                ),
                require_cuda=bool(resources.get("require_cuda", True)),
                artifact_budget_bytes=_positive(
                    resources.get("artifact_budget_bytes", 5 * 2**30),
                    "resources.artifact_budget_bytes",
                ),
                disk_check_path=relative(disk_check_raw, "resources.disk_check_path"),
            ),
            training=TrainingConfig(
                epochs=_positive(training.get("epochs", 1), "training.epochs"),
                cases_per_window=_positive(
                    training.get("cases_per_window", 1), "training.cases_per_window"
                ),
                max_steps=max_steps,
            ),
            workload=WorkloadConfig(
                type=workload_type,
                options=_mapping(workload.get("options", {}), "workload.options"),
            ),
            config_path=config_path,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "strategy": self.strategy,
            "data": {
                "manifest": str(self.data.manifest),
                "source": {"type": self.data.source.type, "root": self.data.source.root},
                "cache_dir": str(self.data.cache_dir),
                "validate_nifti": self.data.validate_nifti,
            },
            "resources": {**vars(self.resources), "disk_check_path": str(self.resources.disk_check_path)},
            "training": vars(self.training),
            "workload": {"type": self.workload.type, "options": self.workload.options},
        }
