"""Deterministic ResNet-18/50 topology adapter for 2.5D systems calibration."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, DependencyError


class _BasicBlock:
    expansion = 1

    @staticmethod
    def build(torch, in_channels, channels, stride):
        nn = torch.nn
        projection = None
        if stride != 1 or in_channels != channels:
            projection = nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                                       nn.GroupNorm(1, channels))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
                self.norm1 = nn.GroupNorm(1, channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                self.norm2 = nn.GroupNorm(1, channels)
                self.projection = projection

            def forward(self, x):
                identity = x if self.projection is None else self.projection(x)
                out = torch.relu(self.norm1(self.conv1(x)))
                out = self.norm2(self.conv2(out))
                return torch.relu(out + identity)

        return Block()


class _Bottleneck:
    expansion = 4

    @staticmethod
    def build(torch, in_channels, channels, stride):
        nn = torch.nn
        out_channels = channels * 4
        projection = None
        if stride != 1 or in_channels != out_channels:
            projection = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                       nn.GroupNorm(1, out_channels))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
                self.norm1 = nn.GroupNorm(1, channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
                self.norm2 = nn.GroupNorm(1, channels)
                self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
                self.norm3 = nn.GroupNorm(1, out_channels)
                self.projection = projection

            def forward(self, x):
                identity = x if self.projection is None else self.projection(x)
                out = torch.relu(self.norm1(self.conv1(x)))
                out = torch.relu(self.norm2(self.conv2(out)))
                out = self.norm3(self.conv3(out))
                return torch.relu(out + identity)

        return Block()


def _make_resnet(torch, depth: int, in_channels: int, classes: int, base_width: int):
    if depth == 18:
        block, counts = _BasicBlock, [2, 2, 2, 2]
    elif depth == 50:
        block, counts = _Bottleneck, [3, 4, 6, 3]
    else:
        raise ConfigurationError("resnet2p5d depth must be 18 or 50")
    nn = torch.nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(in_channels, base_width, 7, 2, 3, bias=False),
                                      nn.GroupNorm(1, base_width), nn.ReLU(),
                                      nn.MaxPool2d(3, 2, 1))
            current = base_width
            layers = []
            for stage, count in enumerate(counts):
                channels = base_width * (2**stage)
                blocks = []
                for index in range(count):
                    stride = 2 if stage > 0 and index == 0 else 1
                    blocks.append(block.build(torch, current, channels, stride))
                    current = channels * block.expansion
                layers.append(nn.Sequential(*blocks))
            self.layers = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(current, classes))

        def forward(self, x):
            return self.head(self.layers(self.stem(x)))

    return Model()


class ResNet2p5DWorkload:
    """Train ResNet topology on deterministic adjacent-slice proxy inputs.

    The adapter is a systems calibration workload. It does not decode medical
    images and cannot support a clinical or accuracy claim.
    """

    def __init__(self, seed: int, options: dict[str, Any]):
        try:
            import torch
        except ImportError as exc:
            raise DependencyError("resnet2p5d requires PyTorch") from exc
        self.torch = torch
        self.seed = seed
        self.depth = int(options.get("depth", 18))
        self.input_size = int(options.get("input_size", 64))
        self.in_channels = int(options.get("in_channels", 3))
        self.classes = int(options.get("classes", 4))
        self.base_width = int(options.get("base_width", 64))
        if self.input_size < 32 or self.in_channels < 1 or self.classes < 2 or self.base_width < 8:
            raise ConfigurationError("invalid resnet2p5d dimensions")
        requested = str(options.get("device", "cuda"))
        if requested == "cuda" and not torch.cuda.is_available():
            raise DependencyError("resnet2p5d requested CUDA but CUDA is unavailable")
        self.device = torch.device(requested)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        self.model = _make_resnet(torch, self.depth, self.in_channels,
                                  self.classes, self.base_width).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(options.get("learning_rate", 1e-3)))
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def _sample(self, case_dir: Path, case_id: str):
        digest = hashlib.sha256(f"{self.seed}:{case_id}".encode())
        for path in sorted(case_dir.rglob("*")):
            if path.is_file():
                digest.update(path.read_bytes())
        sample_seed = int.from_bytes(digest.digest()[:8], "big")
        generator = self.torch.Generator(device="cpu").manual_seed(sample_seed)
        image = self.torch.randn((1, self.in_channels, self.input_size, self.input_size), generator=generator)
        target = self.torch.tensor([sample_seed % self.classes], dtype=self.torch.long)
        return image.to(self.device), target.to(self.device)

    def train_case(self, case_dir: Path, case_id: str) -> dict[str, Any]:
        torch = self.torch
        image, target = self._sample(case_dir, case_id)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        logits = self.model(image)
        selected = torch.gather(torch.log_softmax(logits, dim=1), 1, target.unsqueeze(1))
        loss = -selected.mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
        else:
            allocated = reserved = 0
        return {"loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "peak_gpu_allocated_bytes": allocated,
                "peak_gpu_reserved_bytes": reserved,
                "parameter_count": self.parameter_count,
                "resnet_depth": self.depth,
                "input_mode": "2.5d-proxy",
                "finite": bool(torch.isfinite(loss).item())}

    def checkpoint_state(self) -> dict[str, Any]:
        torch = self.torch
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}

    def restore(self, state: dict[str, Any]) -> None:
        torch = self.torch
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
