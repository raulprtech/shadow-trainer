# Shadow Trainer research package v0.1

This directory turns run artifacts into a traceable evidence bundle. It is not
a claim registry: every record is explicitly classified, and the figure
generator includes numeric values only from records marked `valid`.

Build the bundle with the installed CLI:

    shadow-trainer benchmark research/evidence-spec.json \
      --output-dir research/generated

Or without installation:

    PYTHONPATH=src python -m shadow_trainer benchmark \
      research/evidence-spec.json --output-dir research/generated

The command validates the five required run artifacts, parses every JSONL
event, derives metrics, computes a SHA-256 digest over the bundle, and writes:

- `evidence.json`: machine-readable canonical record;
- `evidence.csv`: analysis input;
- `evidence.md`: human-review table and digests;
- `figures/peak-vram.svg` and `figures/runtime.svg`.

## Evidence policy

- `valid`: may appear in numeric figures within its stated permitted use.
- `engineering_only`: useful for product validation but excluded from figures.
- `diagnostic`: failure analysis only.
- `invalid`: retained for audit, never used to support a result.
- `pending`: named requirement with no completed evidence.

Changing a run artifact changes its bundle digest. Generated outputs never
contain source or cache paths; only the stable evidence id is published.

## Current scope

P1 and P2 establish feasibility, not comparative performance. D1 establishes
that failures remain visible and auditable. Stage38 is described in the
experiment matrix but is not imported because its prefetch arm was never run.
No speedup, clinical quality, or universal equivalence statement is currently
permitted.

## Paired physical matrix

If the physical preflight fails on disk headroom, follow the non-deleting
`research/PHYSICAL_DISK_RECOVERY.md` procedure before retrying.

The guarded matrix runner launches four workloads, two cache conditions, three
repetitions, and two strategies as 48 independent Python processes. Run the
read-only resource gate first:

    PYTHONPATH=src python research/run_pair_matrix.py --preflight-only

After a complete successful matrix, compile timing results with:

    shadow-trainer summarize-pairs MATRIX.json --output-dir pair-summary

The compiler emits JSON, CSV, and Markdown only after checking every exact pair.
Any missing cell, divergence, invalid duration, or non-success matrix suppresses
all performance-claim eligibility.
