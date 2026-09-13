# Autonomous research block — 2026-09-13

## Host state

- Physical C: free space at admission check: 7,696,302,080 bytes.
- KiTS23 job: rejected before staging with reason `disk_headroom`.
- WSL ext4 free space before calibration: 947,474,763,776 bytes.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB total.

No disk floor was reduced and no heavy run was attempted on C:.

## Product changes

- Added `shadow-trainer benchmark` and the versioned
  `shadowtrainer.evidence-spec/v1` contract.
- Added strict run-bundle validation, derived CSV/JSON/Markdown metrics,
  provenance SHA-256, and SVG generation restricted to valid records.
- Added a ResNet-18/50 topology adapter for deterministic 2.5D proxy systems
  calibration.
- Added a paired uninterrupted-versus-resume state equivalence test.

## Physical calibrations

### P3 — ResNet18

- Four configured steps completed in 6.813 seconds.
- 11,178,564 trainable parameters.
- Peak allocated/reserved GPU memory: 235.15/258 MiB.
- Cache occupancy/budget: 8/8 KiB.
- Combined cache and run artifacts: 134,244,588 bytes under 512 MiB.
- All recorded losses finite.

### P4 — ResNet50

- Two configured steps completed in 5.278 seconds.
- 23,516,228 trainable parameters.
- Peak allocated/reserved GPU memory: 471/488 MiB.
- Cache occupancy/budget: 4/8 KiB.
- Combined cache and run artifacts: 282,414,116 bytes under 1 GiB.
- All recorded losses finite.

These independent, short runs show adapter feasibility only. Runtime and loss
values are not comparative-performance or model-quality results.

## Preserved diagnostic

The first ResNet18 attempt was rejected before model construction because the
fixture generator emitted non-contract manifest keys. Its partial run metadata
was moved to `research/workspace/run-resnet18-manifest-failure`. The generator
was corrected to emit `case_id`, `name`, and `source`, and the successful run
used the intended output name. This directory is ignored because it is locally
reconstructible.

## Verification

- 21 automated tests pass.
- Python bytecode compilation passes.
- Generated SVG files parse as XML.
- Generated evidence contains no run/cache paths.
- D1 diagnostic evidence is absent from numeric SVG figures.
- LaTeX brace balance passes; no TeX engine is installed locally, so PDF
  compilation remains unverified.

## Open gates

- WSL/Windows disk recovery and Stage38 r3 paired experiment.
- Repeated cold/warm runs and confidence intervals.
- Framework-local PyTorch/MONAI baseline.
- Recovery state equivalence for NIfTI and ResNet adapters.
- Real adjacent-slice 2.5D data instead of deterministic proxy inputs.
- Human review of authorship, IP boundary, venue format, and bibliography.
