#!/usr/bin/env python3
"""Fail when checked-in scientific outputs differ from current run artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from shadow_trainer.scientific import build_evidence_bundle


def main() -> None:
    root = Path(__file__).resolve().parent
    expected = root / "generated"
    names = ("evidence.json", "evidence.csv", "evidence.md",
             "figures/peak-vram.svg", "figures/runtime.svg")
    with tempfile.TemporaryDirectory(prefix="shadow-evidence-") as temporary:
        actual = Path(temporary)
        build_evidence_bundle(root / "evidence-spec.json", actual)
        stale = [name for name in names
                 if not (expected / name).is_file()
                 or (expected / name).read_bytes() != (actual / name).read_bytes()]
    if stale:
        raise SystemExit("stale or missing generated evidence: " + ", ".join(stale))
    print("evidence_package_current")


if __name__ == "__main__":
    main()
