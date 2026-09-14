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
| H3 | Resume reproduces uninterrupted state for the same job. | Paired checkpoint hashes, step ids, losses and final state. | Exact small CPU pairs cover all four adapters; repeated physical CUDA equivalence remains pending. |
| H4 | Prefetch is exactly equivalent to sync and reduces exposed I/O time. | Valid paired Stage38 run with independent caches and identical seeds. | Open; no claim permitted. |
| H5 | Runtime contracts transfer across workload families. | At least three adapters using the public API. | Supported at physical feasibility level by P1--P4; representative training remains pending. |

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

| Family | Data | Conditions | Strategies | Repetitions per cell | Gate |
| --- | --- | --- | --- | ---: | --- |
| Tiny3D | Local fixture greater than cache | cold + fully warm | sync + prefetch | 3 paired | Regression and equivalence |
| NIfTI 3D | Fixed verified two-case manifest | cold + fully warm | sync + prefetch | 3 paired | Physical CUDA equivalence |
| ResNet18 | Fixed verified six-case fixture | cold + fully warm | sync + prefetch | 3 paired | Generality |
| ResNet50 | Fixed verified six-case fixture | cold + fully warm | sync + prefetch | 3 paired | Generality |
| Failure injection | Synthetic rclone | n/a | sync | one per failure | Safety |

The paired matrix therefore launches 48 new workload processes. Failure injection
is a separate integration suite and is not mixed into timing analysis.

Each arm runs in a fresh process. Paired arms use identical manifests, order,
seeds, initial checkpoints and environment metadata. Cold and warm runs are
reported separately; they are never averaged together.

## Binary decisions

- Enable `auto -> prefetch` only after all paired hashes and step metrics match.
- If equivalence fails, retain `auto -> sync` and publish the negative result.
- Do not report speedup from any pair with different input cache warmth.
- Do not report clinical Dice from training patches.
- Stop before session creation, warm-cache preparation, or an arm when physical
  C: has less than the immutable 20-GiB floor; retain a 3-GiB matrix-session ceiling.

## Exact next execution

After compacting WSL and confirming at least 20 GiB free on `/mnt/c`:

    cd /home/raulprtech/shadow-trainer
    PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python \
      research/run_pair_matrix.py --preflight-only

Only after a successful preflight, create a fresh matrix directory:

    PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python \
      research/run_pair_matrix.py \
      --output-dir research/workspace/pair-matrix-physical-r1

If and only if `matrix.json` ends in `success` with every pair exact:

    shadow-trainer summarize-pairs \
      research/workspace/pair-matrix-physical-r1/matrix.json \
      --output-dir research/workspace/pair-matrix-physical-r1/analysis

The summary compiler suppresses timings for any missing, divergent, or invalid
pair. Stage38 remains a separate large-case confirmation after this matrix. Never
reuse a partial output directory or reduce the disk floor to force admission.
