# Shadow Trainer

Shadow Trainer is an auditable local runtime for training AI workloads when
VRAM, RAM, disk, and dataset locality are hard constraints. It stages complete
cases through a bounded cache, checks configured admission limits,
records resource and integrity evidence, and creates checkpoints for resumption.
Resource enforcement and recovery have adapter-specific limits; this is a
research prototype whose remaining robustness gates are tracked below.

This repository is the clean MVP product extraction. The historical
Stream-HOT experiment archive remains unchanged at
/home/raulprtech/stream-hot-kits-mini.

## Current safety status

- sync is the stable default.
- auto selects sync while the Stage38 equivalence gate remains open.
- Explicit prefetch is available for controlled experiments and is reported
  as experimental.
- No clinical performance claim is made.
- Historical passing tests do not establish universal recovery or resource
  guarantees. Later audits identified guard, checkpoint, NIfTI-validation and
  prefetch-cleanup edge cases that still require fixes and regression checks.
- STU-Net research supervisors and the installed generic runtime are separate
  execution paths. Their measured capabilities must not be conflated.

## Repository and local artifacts

The private development repository is
https://github.com/raulprtech/shadow-trainer. `src/shadow_trainer/` contains
the product runtime, `research/` the experimental supervisors, `tests/` their
checks, and `docs/` the supporting documentation.

Public presentation demo:
https://ia-local-circuito14-demo.raulavenger21.chatgpt.site. It replays saved
synthetic GPU events and presents separate aggregate STU-Net evidence; it
does not run training in the browser. See docs/PUBLIC_DEMO_20260930.md for
the presenter guide and offline copy.

Datasets, checkpoints, virtual environments, caches, detailed case-level
receipts and application documents remain local. Some historical experiment
configs contain workstation-specific paths; they document those experiments
and must be adapted before execution on another machine. See
docs/GITHUB_UPLOAD_20260930.md for the upload scope and validation record.

## Environment

The validated workstation environment is currently:

- WSL/Linux
- NVIDIA RTX 3050 Ti Laptop GPU, 4096 MiB
- Python 3.12
- PyTorch 2.12 + CUDA 13

Use the dedicated repository environment, not the clinical environment:

    .venv/bin/python
    .venv/bin/shadow-trainer

The offline Linux/Python 3.12 dependency lock is
research/requirements-mvp-linux-py312.lock. See docs/REPRODUCIBLE_MVP.md
for reconstruction, resource budgets, tests and physical-demo acceptance.

For a new development checkout with network access, the basic setup is:

    git clone https://github.com/raulprtech/shadow-trainer.git
    cd shadow-trainer
    python3 -m venv .venv
    .venv/bin/python -m pip install -e '.[torch,nifti,dev]'
    .venv/bin/shadow-trainer doctor --json

This downloads dependencies; the host Python/CUDA stack and available disk
must be checked first. The hash-locked offline reconstruction above is the
recipe for the captured workstation environment. Some research evaluator
tests additionally require SciPy and Matplotlib in a scientific environment.
STU-Net training also depends on the separate historical laboratory and
authorized data/model assets.

Prepare a small offline fixture without training:

    .venv/bin/shadow-trainer demo --cpu --prepare-only --output-dir demo-output-new
    .venv/bin/shadow-trainer plan demo-output-new/demo-job.json --json

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
    shadow-trainer demo --cpu --prepare-only --output-dir NEW_DIRECTORY

Use --cpu only for portable smoke tests. The Circuito 14 evidence run must use
the physical NVIDIA GPU.

The public workload adapters are `tiny3d`, `nifti_patch3d`, and `resnet2p5d`
(ResNet-18/50 topology). The latter is a systems calibration adapter, not a
medical-quality benchmark.

## Job contract

Generic jobs use shadowtrainer.job/v1; the sealed Clinical-Nigma bridge uses
shadowtrainer.job/v2. Paths may be absolute or relative to the job
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
