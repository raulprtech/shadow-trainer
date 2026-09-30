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
The current claims authority is [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md).
The architecture-by-architecture product expansion and its separate evidence
gates are in [ADAPTER_ROADMAP_ES.md](../docs/ADAPTER_ROADMAP_ES.md); these
future adapters are not claimed as implemented by the STU-Net campaigns.
The subsequent `pair-matrix-physical-r1` supports exact audited state in 24
small physical fixture pairs and a narrow timing result for cold-cache NIfTI3D
(three two-case pairs). Any timing claim must include the complete eight-cell
table, protocol and uncertainty in that matrix's `pair-summary.json`; it is not
universal acceleration or equivalence for the long STU-Net protocol. Clinical
quality claims remain unsupported. `auto` remains synchronous until the
separate long STU-Net gate passes.

## Paired physical matrix

If the physical preflight fails on disk headroom, follow the non-deleting
`research/PHYSICAL_DISK_RECOVERY.md` procedure before retrying.

The guarded matrix runner launches four workloads, two cache conditions, three
repetitions, and two strategies as 48 independent Python processes. Run the
read-only resource gate first:
After the disk gate passes, the same preflight verifies the size and SHA-256 of
every local source object before creating a session.


    PYTHONPATH=src python research/run_pair_matrix.py --preflight-only

After a complete successful matrix, compile timing results with:

    shadow-trainer summarize-pairs MATRIX.json --output-dir pair-summary

The compiler emits JSON, CSV, and Markdown only after checking every exact pair. It accepts only matrices that explicitly declare
`execution_seconds` as their timing field.
Any missing cell, divergence, invalid duration, or non-success matrix suppresses
all performance-claim eligibility.


## STU-Net campaign for thesis review

The bounded campaign is configured in `research/stunet_campaign_config.json`.
Run its read-only gate first:

    PYTHONPATH=src python research/run_stunet_campaign.py \
      --config research/stunet_campaign_config.json --preflight-only

After OAuth and a passing gate, launch the autonomous session with:

    PYTHONPATH=src python research/run_stunet_campaign.py \
      --config research/stunet_campaign_config.json --max-hours 8

Resume only with the durable session directory:

    PYTHONPATH=src python research/run_stunet_campaign.py \
      --resume research/workspace/stunet-campaign-20260915-r1 --max-hours 8

The protocol compares frozen Stage20, the Stage20 loss, and a hierarchical
renal/tumor loss. It freezes a hash-selected evaluation cohort, stages cases
through bounded caches, checkpoints every training case, exports NIfTI masks
and overlays, and closes with JSON, CSV, HTML and presentation artifacts.
Stage38 remains the separate long STU-Net sync/prefetch gate.

## Deferred nnU-Net baseline

After the current STU-Net replication closes, consider nnU-Net v2 trained from
scratch as a segmentation-quality baseline. Start with the same 64 training
patients and four development patients; expand to the frozen 96-case cohort
only if the 4-GiB GPU and disk preflight pass. Keep the locked test unopened.
Compare full-volume patient-level metrics and resource cost, not case count
alone. STU-Net starts from a pretrained Stage20 checkpoint, so this comparison
is between practical pipelines and cannot isolate architecture, pretraining,
or Shadow Trainer. A later STU-Net-from-scratch ablation would address part of
that confounding. Record nnU-Net's own configuration and preprocessing, plus
time, updates, VRAM, RAM and disk; do not relabel a reduced configuration as
the standard `3d_fullres` baseline. Do not start this benchmark concurrently
with the active STU-Net campaign.

## Offline clinical-bridge conformance smoke

`run_bridge_smoke.py` exercises Clinical-Nigma job-build/plan, Shadow Trainer
CPU run/report, and Clinical-Nigma receipt-build in separate processes. It uses
two fabricated 16-byte cases and at most two Tiny3D steps per attempt, with CUDA
disabled. It checks artifact hashes, duplicate-run refusal, locked-test refusal,
disk rejection and tampered-summary rejection. It is not a clinical experiment.

See [round-2 handoff](../docs/coordination/integration-round2.md) for the command,
resource limits, valid result and retained diagnostic attempt. Always use a new
output directory and run `--preflight-only` first. This smoke does not close the
long STU-Net gate or establish ARIA/Nigma/EGO compatibility.
