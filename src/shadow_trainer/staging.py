"""Budgeted, atomic case-level staging."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import ConfigurationError, DependencyError, IntegrityError


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    name: str
    source: str
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    files: tuple[FileRecord, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


def _safe_relative(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{label} must stay inside its configured root")
    return path.as_posix()


def load_manifest(path: Path) -> dict[str, CaseRecord]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read manifest: {exc}") from exc
    if not isinstance(raw.get("cases"), list) or not raw["cases"]:
        raise ConfigurationError("manifest.cases must be a non-empty list")
    records: dict[str, CaseRecord] = {}
    for item in raw["cases"]:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            raise ConfigurationError("each case requires a string case_id")
        case_id = item["case_id"]
        _safe_relative(case_id, "case_id")
        if case_id in records:
            raise ConfigurationError(f"duplicate case_id: {case_id}")
        files_raw = item.get("files")
        if files_raw is None and "image" in item and "label" in item:
            files_raw = [
                {"name": "imaging.nii.gz", **item["image"]},
                {"name": "segmentation.nii.gz", **item["label"]},
            ]
        if not isinstance(files_raw, list) or not files_raw:
            raise ConfigurationError(f"{case_id}: files must be a non-empty list")
        files: list[FileRecord] = []
        names: set[str] = set()
        for raw_file in files_raw:
            if not isinstance(raw_file, dict):
                raise ConfigurationError(f"{case_id}: invalid file record")
            name = _safe_relative(str(raw_file.get("name", "")), "file name")
            source = _safe_relative(str(raw_file.get("source", "")), "file source")
            size = raw_file.get("size_bytes")
            if not name or not source or not isinstance(size, int) or size < 0:
                raise ConfigurationError(f"{case_id}: invalid file name/source/size")
            if name in names:
                raise ConfigurationError(f"{case_id}: duplicate destination {name}")
            digest = raw_file.get("sha256")
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)
            ):
                raise ConfigurationError(f"{case_id}/{name}: invalid SHA-256")
            names.add(name)
            files.append(FileRecord(name, source, size, digest.lower() if digest else None))
        records[case_id] = CaseRecord(case_id, tuple(files))
    return records


class Source:
    def materialize(self, source: str, destination: Path) -> None:
        raise NotImplementedError


class LocalSource(Source):
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def materialize(self, source: str, destination: Path) -> None:
        candidate = (self.root / _safe_relative(source, "source")).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise IntegrityError("source escaped local root") from exc
        if not candidate.is_file():
            raise IntegrityError(f"source file missing: {source}")
        shutil.copyfile(candidate, destination)


class RcloneSource(Source):
    def __init__(self, remote: str):
        binary = shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")
        if not Path(binary).is_file() or not os.access(binary, os.X_OK):
            raise DependencyError("rclone is not installed or executable")
        self.binary = binary
        self.remote = remote.rstrip("/")

    def materialize(self, source: str, destination: Path) -> None:
        safe = _safe_relative(source, "source")
        try:
            subprocess.run(
                [
                    self.binary,
                    "copyto",
                    f"{self.remote}/{safe}",
                    str(destination),
                    "--contimeout",
                    "15s",
                    "--timeout",
                    "60s",
                    "--retries",
                    "2",
                    "--low-level-retries",
                    "2",
                    "--buffer-size",
                    "4M",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise IntegrityError(f"rclone failed for {safe}: {type(exc).__name__}") from exc


class CaseStager:
    def __init__(
        self,
        manifest: Path,
        cache_dir: Path,
        budget_bytes: int,
        source: Source,
        *,
        validate_nifti: bool = False,
        event_callback=None,
    ):
        self.records = load_manifest(manifest)
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = budget_bytes
        self.source = source
        self.validate_nifti = validate_nifti
        self.event_callback = event_callback
        self.state_path = self.cache_dir / "cache_state.json"
        self.events_path = self.cache_dir / "staging_events.jsonl"
        self.lock_path = self.cache_dir / ".stager.lock"

    @contextmanager
    def locked(self):
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield

    def _state(self) -> dict:
        if not self.state_path.exists():
            return {"schema_version": "shadowtrainer.cache/v1", "cases": {}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _event(self, payload: dict) -> None:
        row = {"timestamp": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if self.event_callback:
            self.event_callback({"kind": "staging", **row})

    @staticmethod
    def _occupancy(state: dict) -> int:
        return sum(int(item["bytes"]) for item in state["cases"].values())

    def _validate_file(self, path: Path, record: FileRecord) -> None:
        if path.stat().st_size != record.size_bytes:
            raise IntegrityError(
                f"{record.name}: expected {record.size_bytes} bytes, got {path.stat().st_size}"
            )
        if record.sha256 and file_sha256(path) != record.sha256:
            raise IntegrityError(f"{record.name}: SHA-256 mismatch")

    def _validate_nifti(self, case_dir: Path) -> dict:
        if not self.validate_nifti:
            return {}
        try:
            import nibabel as nib
            import numpy as np
        except ImportError as exc:
            raise DependencyError("NIfTI validation requires nibabel and numpy") from exc
        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"
        if not image_path.exists() or not label_path.exists():
            raise IntegrityError("NIfTI validation requires imaging and segmentation files")
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)
        image = np.asanyarray(image_nii.dataobj)
        label = np.asanyarray(label_nii.dataobj)
        if image.shape != label.shape:
            raise IntegrityError("image and label shapes differ")
        labels = {int(value) for value in np.unique(label)}
        if not labels.issubset({0, 1, 2, 3}):
            raise IntegrityError(f"invalid label values: {sorted(labels)}")
        return {
            "shape": list(image.shape),
            "spacing": [float(value) for value in image_nii.header.get_zooms()[:3]],
            "labels": sorted(labels),
        }

    def _evict(self, required: int, state: dict, protected: set[str]) -> list[str]:
        evicted: list[str] = []
        while self._occupancy(state) + required > self.budget_bytes:
            candidates = [
                (case_id, value)
                for case_id, value in state["cases"].items()
                if case_id not in protected
            ]
            if not candidates:
                raise IntegrityError("cache budget cannot fit the protected cases")
            victim, _ = min(candidates, key=lambda pair: float(pair[1]["last_access"]))
            shutil.rmtree(self.cache_dir / victim)
            del state["cases"][victim]
            evicted.append(victim)
        return evicted

    def stage_batch(
        self, case_ids: Iterable[str], *, protected_extra: Iterable[str] = ()
    ) -> list[Path]:
        ordered = list(dict.fromkeys(case_ids))
        protected = set(ordered) | set(protected_extra)
        unknown = set(ordered).difference(self.records)
        if unknown:
            raise ConfigurationError(f"unknown cases: {sorted(unknown)}")
        requested_bytes = sum(self.records[item].total_bytes for item in protected)
        if requested_bytes > self.budget_bytes:
            raise IntegrityError("protected cases exceed cache budget")

        with self.locked():
            state = self._state()
            paths: list[Path] = []
            for case_id in ordered:
                record = self.records[case_id]
                case_dir = self.cache_dir / case_id
                started = time.perf_counter()
                if case_id in state["cases"] and case_dir.is_dir():
                    state["cases"][case_id]["last_access"] = time.time()
                    self._event({"event": "cache_hit", "case_id": case_id})
                    paths.append(case_dir)
                    continue
                evicted = self._evict(record.total_bytes, state, protected)
                partial = self.cache_dir / f".{case_id}.partial"
                shutil.rmtree(partial, ignore_errors=True)
                partial.mkdir()
                try:
                    for file_record in record.files:
                        destination = partial / file_record.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        self.source.materialize(file_record.source, destination)
                        self._validate_file(destination, file_record)
                    metadata = self._validate_nifti(partial)
                    os.replace(partial, case_dir)
                except Exception:
                    shutil.rmtree(partial, ignore_errors=True)
                    raise
                elapsed = time.perf_counter() - started
                state["cases"][case_id] = {
                    "bytes": record.total_bytes,
                    "last_access": time.time(),
                    "metadata": metadata,
                }
                self._event(
                    {
                        "event": "cache_miss",
                        "case_id": case_id,
                        "bytes": record.total_bytes,
                        "seconds": elapsed,
                        "evicted": evicted,
                    }
                )
                paths.append(case_dir)
            self._save_state(state)
            return paths

    def status(self) -> dict:
        state = self._state()
        occupancy = self._occupancy(state)
        return {
            "budget_bytes": self.budget_bytes,
            "occupancy_bytes": occupancy,
            "utilization": occupancy / self.budget_bytes,
            "cached_cases": sorted(state["cases"]),
        }
