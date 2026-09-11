"""Memory-bounded NIfTI patch segmentation workload."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, DependencyError, IntegrityError


class NiftiPatch3DWorkload:
    """Train a compact 3D CNN from bounded proxy slices, never a full volume."""

    def __init__(self, seed: int, options: dict[str, Any]):
        try:
            import nibabel as nib
            import numpy as np
            import torch
        except ImportError as exc:
            raise DependencyError(
                "nifti_patch3d requires torch, numpy, and nibabel"
            ) from exc
        self.nib = nib
        self.np = np
        self.torch = torch
        self.seed = seed
        self.step = 0
        self.patch_size = int(options.get("patch_size", 64))
        self.classes = int(options.get("classes", 4))
        self.base_channels = int(options.get("base_channels", 4))
        self.learning_rate = float(options.get("learning_rate", 1e-3))
        self.amp = bool(options.get("amp", True))
        if self.patch_size < 16 or self.classes < 2 or self.base_channels < 2:
            raise ConfigurationError("invalid nifti_patch3d model dimensions")
        device_name = str(options.get("device", "cuda"))
        if device_name == "cuda" and not torch.cuda.is_available():
            raise DependencyError("nifti_patch3d requested CUDA but CUDA is unavailable")
        self.device = torch.device(device_name)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        channels = self.base_channels
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(1, channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(1, channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(1, channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(channels, self.classes, kernel_size=1),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        initial = options.get("initial_checkpoint")
        if initial:
            payload = torch.load(Path(initial).expanduser(), map_location="cpu", weights_only=False)
            state = payload.get("workload", payload)
            self.restore(state)

    def _bounded_patch(self, case_dir: Path, case_id: str):
        np = self.np
        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"
        if not image_path.is_file() or not label_path.is_file():
            raise IntegrityError(f"{case_id}: missing imaging or segmentation NIfTI")
        image_nii = self.nib.load(image_path)
        label_nii = self.nib.load(label_path)
        if tuple(image_nii.shape) != tuple(label_nii.shape) or len(image_nii.shape) != 3:
            raise IntegrityError(f"{case_id}: incompatible 3D image and label shapes")
        if any(size < self.patch_size for size in image_nii.shape):
            raise IntegrityError(
                f"{case_id}: volume {image_nii.shape} is smaller than patch {self.patch_size}"
            )
        digest = hashlib.sha256(f"{self.seed}:{self.step}:{case_id}".encode()).digest()
        starts = []
        for axis, size in enumerate(image_nii.shape):
            maximum = size - self.patch_size
            value = int.from_bytes(digest[axis * 4:axis * 4 + 4], "big")
            starts.append(value % (maximum + 1))
        slices = tuple(slice(start, start + self.patch_size) for start in starts)
        image = np.asarray(image_nii.dataobj[slices], dtype=np.float32)
        label = np.asarray(label_nii.dataobj[slices], dtype=np.int64)
        if not np.isfinite(image).all():
            raise IntegrityError(f"{case_id}: non-finite image patch")
        if label.min() < 0 or label.max() >= self.classes:
            raise IntegrityError(f"{case_id}: label outside configured classes")
        image = np.clip(image, -1024.0, 1024.0)
        image = (image + 1024.0) / 2048.0
        return (
            self.torch.from_numpy(image[None, None]).to(self.device),
            self.torch.from_numpy(label[None]).to(self.device),
            starts,
        )

    def train_case(self, case_dir: Path, case_id: str) -> dict[str, Any]:
        torch = self.torch
        self.model.train()
        x, target, starts = self._bounded_patch(case_dir, case_id)
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp and self.device.type == "cuda",
        ):
            logits = self.model(x)
            log_probabilities = torch.log_softmax(logits, dim=1)
            selected = torch.gather(log_probabilities, 1, target.unsqueeze(1))
            loss = -selected.mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1
        with torch.no_grad():
            predicted = logits.argmax(1)
            foreground = target > 0
            intersection = ((predicted == target) & foreground).sum()
            denominator = foreground.sum() + ((predicted > 0) & foreground).sum()
            dice = (2 * intersection.float() / denominator.clamp_min(1)).item()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
        else:
            allocated = reserved = 0
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "foreground_dice": float(dice),
            "patch_start": starts,
            "patch_size": self.patch_size,
            "peak_gpu_allocated_bytes": allocated,
            "peak_gpu_reserved_bytes": reserved,
            "finite": bool(torch.isfinite(loss).item()),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
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
        self.step = int(state.get("step", 0))
        if "rng" in state:
            random.setstate(state["rng"]["python"])
            torch.set_rng_state(state["rng"]["torch"])
            if torch.cuda.is_available() and state["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
