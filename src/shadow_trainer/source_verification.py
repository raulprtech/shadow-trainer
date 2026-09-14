"""Read-only verification of local experiment manifests and source objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import IntegrityError
from .staging import file_sha256, load_manifest


def verify_local_manifest(manifest: str | Path, source_root: str | Path) -> dict[str, Any]:
    """Verify every declared local object without populating a cache."""
    manifest_path = Path(manifest).expanduser().resolve()
    root = Path(source_root).expanduser().resolve()
    records = load_manifest(manifest_path)
    file_count = 0
    total_bytes = 0
    for case in records.values():
        for record in case.files:
            if record.sha256 is None:
                raise IntegrityError(f"{case.case_id}/{record.name}: SHA-256 is required")
            candidate = (root / record.source).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise IntegrityError(f"{case.case_id}/{record.name}: source escaped root") from exc
            if not candidate.is_file():
                raise IntegrityError(f"{case.case_id}/{record.name}: source file missing")
            actual_size = candidate.stat().st_size
            if actual_size != record.size_bytes:
                raise IntegrityError(
                    f"{case.case_id}/{record.name}: expected {record.size_bytes} bytes, "
                    f"got {actual_size}"
                )
            if file_sha256(candidate) != record.sha256:
                raise IntegrityError(f"{case.case_id}/{record.name}: SHA-256 mismatch")
            file_count += 1
            total_bytes += actual_size
    return {
        "schema_version": "shadowtrainer.source-verification/v1",
        "manifest_sha256": file_sha256(manifest_path),
        "case_count": len(records),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "status": "verified",
    }
