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
- per-arm admission reserves the full artifact budget above the floor;
- runtime rechecks the remaining reservation before and the floor after every
  checkpoint.
- telemetry separates foreground staging, background staging, exposed prefetch
  wait, peak cache occupancy, peak VRAM, transferred bytes and artifact bytes.

## Safe semantic verification

CPU fixtures completed exact sync/prefetch pairs for all four configurations.
The auditor compared:

- manifest canonical digest;
- seed, training and workload configuration;
- fresh-process initialized workload hash and deterministic-algorithm status;
- Torch, CUDA/cuDNN, GPU and driver signature where present;
- ordered epoch, window, step, case, loss and workload metrics;
- checkpoint epoch, window and global step;
- model, optimizer and RNG state digest.
- monotonic `execution_seconds` from ready state through final checkpoint,
  separate from whole-process duration.

The Tiny3D audit test now launches sync and prefetch in independent Python
interpreters. Controlled mutations of the initialized-state event and one final
checkpoint parameter separately return `diverged` at the corresponding gate.
The complete suite now contains 30 passing tests.

The read-only verifier exposed that the historical NIfTI manifest declared sizes
but no SHA-256 values. That file remains unchanged. The matrix now references a
new canonical manifest with four digests. Verification passes for its two cases
(totaling 83,549,098 bytes) and for each six-case Tiny3D/ResNet profile
(totaling 12,288 bytes). The canonical NIfTI manifest digest is
`1b37437c33aebb608f4994927ef1b3955ee3f61a09aceffd57f31c66df472254`.

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

## Physical execution after disk recovery — 2026-09-15

WSL virtual-disk compaction increased free space on the backing Windows volume
to 54,855,868,416 bytes. The unchanged 21,474,836,480-byte floor therefore
passed. Preflight reverified all manifests before session creation, including
the 83,549,098-byte NIfTI source and its canonical manifest digest.

The new `pair-matrix-physical-r1` session completed all 48 independent Python
processes: four workloads, two cache conditions, three repetitions, and two
strategies. All 24 sync/prefetch audits returned `exact`; all 24 were eligible
for comparison; no pair was excluded and no disk, memory, OOM, or divergence
guard stopped the session. The session occupied approximately 1.3 GiB and the
physical volume retained approximately 50 GiB free after execution.

The strongest measured effect was the cold-cache NIfTI3D cell. Its median
audited execution interval was 4.9572 s for sync and 3.4889 s for prefetch,
giving a median sync/prefetch ratio of 1.4209 and a deterministic 10,000-sample
bootstrap 95% interval of [1.2842, 1.4739]. Cold ResNet18 also produced an
interval above one ([1.0290, 1.2190]), but the absolute intervals were short and
the sample count remains three. Every warm-cache interval, along with Tiny3D
and ResNet50 cold, crossed one. The admissible conclusion is therefore a
workload- and cache-condition-specific overlap benefit, not universal speedup.

The canonical lightweight evidence is frozen under
`research/evidence/pair-matrix-physical-r1/`. The full 1.3-GiB run tree remains
under the ignored `research/workspace/` directory.
