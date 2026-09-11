"""Atomic JSON artifacts and append-only event recording."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EventLog:
    def __init__(self, path: Path):
        self.path = path

    def emit(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = {"timestamp": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row
