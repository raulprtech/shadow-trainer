# Canonical evidence status

| Evidence | Status | Permitted use |
| --- | --- | --- |
| Physical RTX 3050 Ti execution | Valid | Hardware feasibility |
| Bounded Drive case staging | Valid | Dataset larger than local cache |
| Stage36 R1b exact pair | Valid | Single-run observed result |
| Stage36 R2 timing | Invalid for performance | Failure analysis only |
| Stage37 F1-F3 replays | Diagnostic | Intermittency evidence only |
| Stage38 full pair | Pending | Required for stable prefetch |

Until Stage38 passes, product documentation and generated plans must describe
prefetch as experimental and must select synchronous staging for auto.
