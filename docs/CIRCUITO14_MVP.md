# Circuito 14 MVP definition

## Industrial problem

AI and edge-engineering teams frequently have workloads whose model, temporary
workspace, and datasets exceed the local GPU or disk available for iteration.
The usual choices are manual trial-and-error, reduced experiments, new
workstations, or cloud infrastructure.

## MVP claim

Shadow Trainer can inspect a constrained workstation, reject unsafe jobs,
stage a dataset through a bounded local window, run a 3D AI workload, resume
from durable state, and produce an auditable report.

## Acceptance

1. The portable GPU demo completes in under ten minutes.
2. The run respects configured cache and artifact budgets.
3. Missing disk, RAM, swap, GPU, CUDA, rclone, or integrity is rejected with a
   stable reason.
4. A controlled interruption resumes without re-running completed windows.
5. The physical validation uses the RTX 3050 Ti and records real telemetry.
6. No invalid experimental pair supports a product claim.

## Explicit exclusions

The MVP is not a medical product, SaaS, educational application, distributed
trainer, autonomous model updater, or replacement for Nigma.
