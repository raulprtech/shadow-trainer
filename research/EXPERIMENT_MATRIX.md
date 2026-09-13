# Experiment matrix and decision gates

## Research questions

**RQ1 — Admission safety.** Can the runtime reject an unsafe workload before
dataset transfer or GPU allocation when VRAM, RAM, swap, or disk headroom is
insufficient?

**RQ2 — Bounded locality.** Can a dataset larger than the configured local
cache be processed while cache occupancy and combined artifacts remain within
hard byte limits?

**RQ3 — Recovery semantics.** Does a resumed job restore model, optimizer, RNG,
and dataset position without repeating confirmed training steps?

**RQ4 — Overlap correctness.** When prefetch overlaps data movement with
training, is the complete observable result equivalent to synchronous staging
under a paired protocol?

**RQ5 — Generality.** Do the same runtime contracts hold for a small synthetic
3D CNN, real NIfTI patch segmentation, ResNet18/50, and a 2.5D workload?

## Hypotheses

| ID | Hypothesis | Required evidence | Current status |
| --- | --- | --- | --- |
| H1 | Admission deterministically prevents resource-floor violations. | Unit rejection tests plus physical low-disk rejection before staging. | Partially supported; physical rejection observed, consolidated run pending. |
| H2 | Peak cache occupancy never exceeds its configured budget. | Dataset larger than cache; occupancy trace; zero partial promotions. | Supported for MVP fixture; larger remote validation pending. |
| H3 | Resume reproduces uninterrupted state for the same job. | Paired checkpoint hashes, step ids, losses and final state. | Bitwise model, optimizer and RNG equality supported for Tiny3D; other adapters pending. |
| H4 | Prefetch is exactly equivalent to sync and reduces exposed I/O time. | Valid paired Stage38 run with independent caches and identical seeds. | Open; no claim permitted. |
| H5 | Runtime contracts transfer across workload families. | At least three adapters using the public API. | Tiny3D and NIfTI validated; ResNet/2.5D pending. |

## Baselines

1. **Framework-local baseline:** ordinary PyTorch/MONAI data loading with all
   required cases already local. This measures runtime overhead, not capacity.
2. **Shadow Trainer sync:** atomic case staging with bounded LRU and no overlap.
3. **Shadow Trainer prefetch:** same manifest, order, seed and checkpoint, with
   only data movement overlap changed. Experimental until H4 passes.
4. **Recovery baseline:** restart from the last epoch-level checkpoint versus
   Shadow Trainer's durable case-window boundary.

MONAI `CacheDataset`/`SmartCacheDataset`, CheckFreq, ZeRO-Offload, and STEER are
related systems, not interchangeable baselines. The paper must compare their
contracts and evaluate any executable baseline whose environment is compatible.

## Required run matrix

| Family | Data | Mode | Repetitions | Metrics | Gate |
| --- | --- | --- | ---: | --- | --- |
| Tiny3D | Local fixture > cache | sync | 5 cold + 5 warm | limits, time, eviction, hashes | MVP regression |
| NIfTI 3D | KiTS23 staged cases | sync | 3 cold + 3 warm | VRAM, RSS, cache, I/O, finite loss | Physical feasibility |
| NIfTI 3D | Same paired manifest | prefetch | 3 cold + 3 warm | same plus exact final hashes | Stage38 |
| ResNet18 | Fixed 2D/2.5D fixture | sync | 3 | VRAM, throughput, checkpoint | Generality |
| ResNet50 | Fixed 2D/2.5D fixture | sync | 3 | VRAM, throughput, checkpoint | Generality |
| Failure injection | Synthetic rclone | sync | one per failure | rejection point, partial files, recovery | Safety |

Each arm runs in a fresh process. Paired arms use identical manifests, order,
seeds, initial checkpoints and environment metadata. Cold and warm runs are
reported separately; they are never averaged together.

## Binary decisions

- Enable `auto -> prefetch` only after all paired hashes and step metrics match.
- If equivalence fails, retain `auto -> sync` and publish the negative result.
- Do not report speedup from any pair with different input cache warmth.
- Do not report clinical Dice from training patches.
- Stop before preparation when physical C: has less than 30 GiB free; retain a
  20-GiB runtime floor and a 5-GiB combined cache/artifact ceiling.

## Exact next execution

After compacting WSL and confirming at least 30 GiB free on `/mnt/c`:

    cd /home/raulprtech/stream-hot-kits-mini
    /home/raulprtech/clinical_core/.venv/bin/python experiments/stage38_r3/prepare.py
    /home/raulprtech/clinical_core/.venv/bin/python experiments/stage38_r3/run_arm.py sync

Inspect the sync summary and guard reasons. Only if it is complete:

    /home/raulprtech/clinical_core/.venv/bin/python experiments/stage38_r3/run_arm.py prefetch
    /home/raulprtech/clinical_core/.venv/bin/python experiments/stage38_r3/audit.py

Then rebuild the product evidence bundle and update the manuscript. Never
reuse a partial r2 run or reduce the disk floor to force admission.
