# Claims ledger

This ledger is the authority for external technical claims. A claim remains
blocked unless the listed evidence exists and matches its permitted use.

| Claim | Evidence | Status | Allowed wording |
| --- | --- | --- | --- |
| Shadow Trainer executes on a physical 4-GiB RTX 3050 Ti. | P1–P4 environment and event bundles. | Supported. | “Validated on an RTX 3050 Ti Laptop GPU with 4 GiB VRAM.” |
| The cache respects an exact byte budget in the observed runs. | P1 4096/4096 B; P3 8192/8192 B; P2 and P4 below budget. | Supported for observed runs and covered by unit tests. | “No cache-budget violation was observed; LRU enforcement is tested.” |
| Real NIfTI image-label pairs can be trained through bounded patches. | P2. | Supported as systems feasibility. | “Completed two real NIfTI cases using deterministic bounded patches.” |
| The runtime supports multiple workload families. | P1 Tiny3D, P2 NIfTI 3D, P3/P4 ResNet 2.5D proxy. | Supported at adapter-feasibility level. | “Three adapters and four physical configurations completed.” |
| Resume does not repeat confirmed step identifiers. | Integration test and Stage38 sync boundary. | Supported narrowly. | “Observed/tested step-position idempotence at durable boundaries.” |
| Tiny3D resume is bitwise identical to uninterrupted execution. | Paired model, optimizer and RNG tensor comparison. | Supported for Tiny3D only. | “Tiny3D paired test reproduces model, optimizer and RNG state exactly.” |
| Resume equivalence generalizes to every adapter. | Paired NIfTI and ResNet state comparisons. | Not yet supported. | Do not claim. |
| Small CPU sync/prefetch fixtures are semantically exact. | Pair-audit tests for Tiny3D, NIfTI, ResNet18 and ResNet50; Tiny3D arms use independent processes and compare initialized state. | Supported only for small CPU fixtures. | “The audit harness passes controlled CPU fixtures.” |
| Prefetch is equivalent to sync on the tested physical CUDA workloads. | `pair-matrix-physical-r1`: 24/24 exact pairs across Tiny3D, NIfTI3D, ResNet18 and ResNet50, cold/warm, three repetitions. | Supported for the tested matrix. | “Prefetch and sync produced identical audited training state in all 24 tested CUDA pairs.” |
| Prefetch accelerates cold NIfTI3D staging in the tested protocol. | Three exact cold-cache NIfTI3D pairs: median sync/prefetch ratio 1.4209, bootstrap 95% CI [1.2842, 1.4739]. | Supported narrowly. | “In the tested two-case cold-cache NIfTI3D protocol, prefetch reduced audited execution time by a median factor of 1.42× (three pairs; bootstrap 95% CI 1.28–1.47×).” |
| Prefetch universally accelerates training. | Warm-cache and small-workload confidence intervals cross 1; the matrix has only three pairs per cell. | Not supported. | Do not claim universal acceleration; report neutral and negative cells alongside the cold NIfTI3D result. |
| Paired timing statistics are eligible for publication. | Successful complete matrix, three exact repetitions in every workload/condition cell, no exclusions, and `execution_seconds` protocol. | Supported for this matrix. | Report the complete eight-cell table, protocol, hardware, low sample count, and bootstrap intervals together. |
| Shadow Trainer trains a dataset larger than all local storage. | Complete remote streamed experiment with occupancy trace. | Pending. | Current claim is limited to bounded staging mechanics. |
| Shadow Trainer improves clinical segmentation. | Held-out patient-level evaluation. | Not supported or in MVP scope. | Do not claim. |

## Rules

1. Product and application materials may use only the “Allowed wording” column.
2. The generated numeric figures include only records classified `valid`, but
   validity does not imply that independent workloads are comparable.
3. Diagnostic and invalid runs may explain engineering decisions, never support
   performance numbers.
4. Any changed source artifact changes the evidence-bundle SHA-256 and requires
   review before figures or manuscript tables are regenerated.
