#!/usr/bin/env python3
"""Create a tiny deterministic local source for ResNet systems calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "workspace"
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in range(6):
        case_id = f"cal_{index:02d}"
        case = source / case_id
        case.mkdir(exist_ok=True)
        path = case / "input.bin"
        payload = hashlib.sha256(f"shadow-trainer:{case_id}".encode()).digest() * 64
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace non-canonical fixture: {path}")
        path.write_bytes(payload)
        cases.append({"case_id": case_id, "files": [{"name": "input.bin", "source": case_id + "/input.bin",
            "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]})
    manifest = {"schema_version": "shadowtrainer.manifest/v1", "cases": cases}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
