# MVP physical validation — 2026-09-10

## Product demo

- Host: WSL2, Linux 5.15.153.1.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB.
- Driver: 610.62.
- PyTorch: 2.12.0+cu130; CUDA 13.0; cuDNN 92000.
- Result: success, 12/12 steps in 2.316 seconds.
- Peak reserved VRAM: 23,068,672 bytes.
- Cache: 4,096/4,096 bytes, no budget violation.
- Physical disk measured through /mnt/c: 28,557,885,440 bytes free at start.
- Strategy: auto selected sync because Stage38 remains open.

Canonical report:
demo-output-physical-v2/run/report.html

## Real NIfTI workload

The first attempt was admitted but failed before a completed step because the
CUDA NLL backward selected by cross-entropy has no deterministic
implementation in the installed PyTorch path. Its failed summary and HTML were
preserved.

The loss was replaced by the equivalent log-softmax plus gather formulation.
The corrected run used a new identifier and output directory.

- Result: success, 2/2 cases in 4.816 seconds.
- Input: two cached KiTS23 NIfTI image/segmentation pairs.
- Patch: 64 cubed; compact four-class 3D CNN; AMP fp16.
- Losses: 0.8374724388 and 0.7953504324, both finite.
- Peak reserved VRAM: 67,108,864 bytes.
- Cache: 83,549,098/134,217,728 bytes.
- Physical disk floor: 20 GiB, observed through /mnt/c.

Canonical report:
evidence/kits23-local-2case-run-v2/report.html

This is systems evidence only. Foreground Dice from two random training patches
is not a clinical metric or product claim.

## Automated verification

- Shadow Trainer suite: 16 passed.
- Stage38 plus Stage34/32 regression selection: 11 passed.

## Stage38 result

Revision 20260910r2 prepared two independently verified caches. The sync arm
was guard-stopped after 108/144 steps because physical C: fell below the
20-GiB floor. It had completed and resumed from the epoch-1 boundary without
OOM, numeric failure, or remote error.

The prefetch arm was not launched. Stage38 does not pass and auto must remain
sync. Deleting reconstructible Stage37 caches recovered ext4 space but the
Windows VHDX did not compact while WSL was running. A new attempt requires WSL
shutdown/compaction and fresh run names.
