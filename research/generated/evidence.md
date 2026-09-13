# Canonical Shadow Trainer evidence

Generated from immutable run artifacts. Numeric publication figures use only `valid` records.

| ID | Class | Workload | Strategy | Status | Steps | Time (s) | Peak VRAM (MiB) | Transfer (MiB) | Permitted use |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| P1 | valid | tiny3d | sync | success | 12 | 2.3162 | 22.0000 | 0.0117 | Physical GPU feasibility, bounded cache, and artifact completeness |
| P2 | valid | nifti_patch3d | sync | success | 2 | 4.8164 | 64.0000 | 36.6769 | Systems feasibility for bounded NIfTI patch training; not clinical efficacy |
| P3 | valid | resnet2p5d | sync | success | 4 | 6.8130 | 258.0000 | 0.0078 | Physical workload-adapter feasibility only; not comparative performance |
| P4 | valid | resnet2p5d | sync | success | 2 | 5.2780 | 488.0000 | 0.0039 | Physical workload-adapter feasibility only; not comparative performance |
| D1 | diagnostic | nifti_patch3d | sync | failed | 0 | 4.1826 | — | 43.0018 | Failure analysis only |

## Integrity digests

- `P1`: `2bc8ce7005aff002fb1555d1a4563b3c7fa4879446f2879d924303f4ec7305f2`
- `P2`: `20cbd4c6417300d90b8dae0e932f4eeb25b07a5a6c64cfcc3587de293a6f6b36`
- `P3`: `9a41f532298fbede91f90baf72a019a15a0632df99ed449b55d5d5567d0c1f39`
- `P4`: `6901be3824be903c0ea3f708a3a4461ba97d39d235263db8064f373e7febd174`
- `D1`: `dc88bc60d23e0a2ad31ed632bf70e2fb74fae49118ae119d0986cd4b4efc04eb`
