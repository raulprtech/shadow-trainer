"""Neutral workload interface and the first PyTorch 3D adapter."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Protocol

from .errors import ConfigurationError, DependencyError


class Workload(Protocol):
    def train_case(self, case_dir: Path, case_id: str) -> dict[str, Any]: ...

    def checkpoint_state(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any]) -> None: ...


class Tiny3DWorkload:
    """Small deterministic CNN used by the portable MVP demonstration."""

    def __init__(self, seed: int, options: dict[str, Any]):
        try:
            import torch
        except ImportError as exc:
            raise DependencyError("the tiny3d workload requires PyTorch") from exc
        self.torch = torch
        self.seed = seed
        self.shape = tuple(options.get("input_shape", [1, 8, 8, 8]))
        if (
            len(self.shape) != 4
            or not isinstance(self.shape[0], int)
            or self.shape[0] < 1
            or any(not isinstance(v, int) or v < 4 for v in self.shape[1:])
        ):
            raise ConfigurationError(
                "tiny3d input_shape requires channels >= 1 and three dimensions >= 4"
            )
        self.classes = int(options.get("classes", 2))
        self.learning_rate = float(options.get("learning_rate", 0.01))
        requested_device = options.get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise DependencyError("tiny3d requested CUDA but CUDA is unavailable")
        self.device = torch.device(requested_device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(self.shape[0], 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, self.classes),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _sample(self, case_dir: Path, case_id: str):
        digest = hashlib.sha256(case_id.encode("utf-8"))
        for path in sorted(case_dir.rglob("*")):
            if path.is_file():
                digest.update(path.read_bytes())
        seed = int.from_bytes(digest.digest()[:8], "big") ^ self.seed
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        x = self.torch.randn((1, *self.shape), generator=generator)
        target = self.torch.tensor([seed % self.classes], dtype=self.torch.long)
        return x.to(self.device), target.to(self.device)

    def train_case(self, case_dir: Path, case_id: str) -> dict[str, Any]:
        torch = self.torch
        self.model.train()
        x, target = self._sample(case_dir, case_id)
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        logits = self.model(x)
        loss = self.loss_fn(logits, target)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        else:
            peak_allocated = peak_reserved = 0
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(grad.detach().cpu()),
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "finite": bool(torch.isfinite(loss).item()),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

    def restore(self, state: dict[str, Any]) -> None:
        torch = self.torch
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])


def create_workload(kind: str, seed: int, options: dict[str, Any]) -> Workload:
    if kind == "tiny3d":
        return Tiny3DWorkload(seed, options)
    if kind == "nifti_patch3d":
        from .nifti_workload import NiftiPatch3DWorkload
        return NiftiPatch3DWorkload(seed, options)
    raise ConfigurationError(f"unsupported workload: {kind}")
