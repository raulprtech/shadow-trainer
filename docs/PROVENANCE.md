# Provenance and experimental boundary

Shadow Trainer 0.1 extracts general mechanisms from the Stream-HOT laboratory
without modifying or renaming the historical artifacts.

| Product component | Experimental origin |
| --- | --- |
| Atomic case staging and integrity | case_stager.py |
| Cache traffic policy | stage26_policy.py |
| RAM/swap admission | stage26_memory.py |
| Backend admission | stage26_backend.py |
| RNG and optimizer resume | stage26_restart.py |
| Paired validation protocol | experiments/stage36/PROTOCOL.md |

The product uses new neutral contracts and contains no fixed KiTS paths,
experiment stage numbers, checkpoint identities, or dated run names.

## Evidence classification at extraction

- Stage36 R1b: valid exact pair; observed prefetch duration was 10.44% lower.
- Stage36 R2: invalid for claims because paired exactness failed at step 88.
- Stage37 controlled replays: exact, but not substitutes for a new full pair.
- Stage38: required before auto may enable prefetch.

Observed performance is not a universal speed claim. KiTS23 is a systems
validation workload and does not support a clinical claim.
