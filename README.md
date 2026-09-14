# Shadow Trainer

Shadow Trainer is an auditable local runtime for training AI workloads when
VRAM, RAM, disk, and dataset locality are hard constraints. It stages complete
cases through a bounded cache, rejects unsafe jobs before GPU allocation,
records resource and integrity evidence, and creates resumable checkpoints.

This repository is the clean MVP product extraction. The historical
Stream-HOT experiment archive remains unchanged at
/home/raulprtech/stream-hot-kits-mini.

## Current safety status

- sync is the stable default.
- auto selects sync while the Stage38 equivalence gate remains open.
- Explicit prefetch is available for controlled experiments and is reported
  as experimental.
- No clinical performance claim is made.

## Environment

The validated workstation environment is currently:

- WSL/Linux
- NVIDIA RTX 3050 Ti Laptop GPU, 4096 MiB
- Python 3.12
- PyTorch 2.12 + CUDA 13

For the local machine, use the existing interpreter:

    /home/raulprtech/clinical_core/.venv/bin/python

Install the package without duplicating PyTorch:

    python -m pip install -e /home/raulprtech/shadow-trainer --no-deps

## Commands

    shadow-trainer doctor --json
    shadow-trainer plan JOB.json --json
    shadow-trainer run JOB.json
    shadow-trainer resume RUN_DIR
    shadow-trainer report RUN_DIR
    shadow-trainer benchmark research/evidence-spec.json --output-dir research/generated
    shadow-trainer audit-pair SYNC_RUN PREFETCH_RUN --output pair-audit.json
    shadow-trainer summarize-pairs MATRIX.json --output-dir pair-summary
    shadow-trainer demo --output-dir demo-output

Use --cpu only for portable smoke tests. The Circuito 14 evidence run must use
the physical NVIDIA GPU.

The public workload adapters are `tiny3d`, `nifti_patch3d`, and `resnet2p5d`
(ResNet-18/50 topology). The latter is a systems calibration adapter, not a
medical-quality benchmark.

## Job contract

Jobs use shadowtrainer.job/v1. Paths may be absolute or relative to the job
file. Byte budgets are integers so admission never depends on ambiguous unit
parsing. See docs/example-job.json.

The generic manifest format is shadowtrainer.manifest/v1, with cases and named
files. The loader also accepts the existing KiTS manifest shape with image and
label objects.

## Run artifacts

Every admitted run produces job.json, plan.json, environment.json,
events.jsonl, summary.json, latest.checkpoint.pt, report-summary.json, and a
self-contained report.html. Failures also produce a summary and report.

## Product boundary

The September MVP does not include ARIA, active Nigma integration, autonomous
adaptation, distributed training, a cloud service, or model promotion. Those
remain subsequent milestones after the Circuito 14 application gate.
