"""Canonical structured hashing for PyTorch workload state."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def state_digest(value: Any) -> str:
    """Hash nested checkpoint state without depending on pickle byte layout."""
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        elif isinstance(item, float):
            digest.update((item.hex() if math.isfinite(item) else repr(item)).encode())
            digest.update(b"\0")
        else:
            digest.update(type(item).__name__.encode() + b":" + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()
