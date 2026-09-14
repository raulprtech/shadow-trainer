# Paired-matrix block — 2026-09-13

## Physical preflight

The 48-process physical matrix was not launched. Its preflight observed
5,355,749,376 free bytes on C: against the unchanged 21,474,836,480-byte
runtime floor and returned exit code 2 before creating a session directory.
After CPU integration verification, C: had 5,110,845,440 free bytes. A final
preflight observed 6,466,846,720 free bytes and remained blocked. The final
Windows inventory reported 3,910,651,904 free bytes; the fluctuation does not
change the decision. No GPU
pair, timing comparison, or Stage38 arm is valid from this block. The non-deleting recovery procedure is recorded in
`research/PHYSICAL_DISK_RECOVERY.md`.

The matrix runner is configured for:

- Tiny3D, NIfTI 3D, ResNet18, and ResNet50;
- cold and fully warm cache conditions kept separate;
- three independently spawned sync/prefetch pairs per condition;
- 48 new Python processes in total;
- exact audit after every pair;
- immediate stop on divergence;
- 20-GiB physical C: floor and 3-GiB session ceiling.

## Safe semantic verification

CPU fixtures completed exact sync/prefetch pairs for all four configurations.
The auditor compared:

- manifest canonical digest;
- seed, training and workload configuration;
- Torch, CUDA/cuDNN, GPU and driver signature where present;
- ordered epoch, window, step, case, loss and workload metrics;
- checkpoint epoch, window and global step;
- model, optimizer and RNG state digest.

A separate negative test modified one checkpoint parameter and the auditor
returned `diverged` with `workload_state` as the failed check. The complete
suite now contains 29 passing tests.

All four source manifests were also verified read-only before launch: the NIfTI
manifest contains two cases totaling 83,549,098 bytes, while each Tiny3D and
ResNet manifest contains six cases totaling 12,288 bytes; every declared size
and SHA-256 digest matched.

## Claim boundary

The CPU fixtures establish that the auditor and current adapter semantics can
detect equality and deliberate divergence. They do not establish physical CUDA
equivalence, prefetch speedup, representative I/O behavior, or large-dataset
capacity. `auto` must remain synchronous until the physical matrix and Stage38
requirements are satisfied.

## Resume command after disk recovery

First verify the preflight without creating artifacts:

    cd /home/raulprtech/shadow-trainer
    PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python \
      research/run_pair_matrix.py --preflight-only

When it returns success and C: has at least 20 GiB free, use a new output name:

    PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python \
      research/run_pair_matrix.py \
      --output-dir research/workspace/pair-matrix-physical-r1

Do not reuse a partial output directory. The runner writes `matrix.json` after
each pair, so a guard stop or divergence remains auditable.
