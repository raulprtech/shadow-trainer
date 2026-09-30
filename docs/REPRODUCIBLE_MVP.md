# Reproducible local MVP

Scope: Linux x86_64 / CPython 3.12, dedicated venv, hash-locked offline
dependencies, installed package, tests and short physical CUDA demo.
This does not certify clinical quality or the long STU-Net prefetch gate.

## Reconstruct without touching clinical environments

Run from the repository root. Require at least 20 GiB free on C: **after**
installation, plus up to 5 GiB installation reserve. Do not reuse an existing
venv or wheelhouse for a fresh reconstruction.

The cache recovery tool links wheel archives only; it does not link installed
packages. It requires cache and wheelhouse on the same filesystem. Wheels are
local inputs, not authenticated upstream by this procedure. SHA-256 fixes their
identity. Cache deletion later does not remove the linked wheelhouse.

    python3 research/prepare_offline_environment.py collect --cache /path/to/pip-cache --output NEW_WHEELHOUSE
    python3 -m venv NEW_VENV
    NEW_VENV/bin/python -m pip install --no-index --no-compile --require-hashes --find-links NEW_WHEELHOUSE -r research/requirements-mvp-linux-py312.lock
    NEW_VENV/bin/python -m pip check
    NEW_VENV/bin/python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir NEW_PACKAGE_DIR .
    NEW_VENV/bin/python -m pip install --no-index --no-deps NEW_PACKAGE_DIR/shadow_trainer-0.1.0-py3-none-any.whl

The captured lock contains 38 exact versions with archive hashes, including
torch===2.12.0 (CUDA; not 2.12.0+cpu). If any required wheel is absent, fail;
do not silently enable network or substitute a CPU build. Linux/CUDA driver
and Python itself are host prerequisites, not installed by the lock.

## Tests

With the dedicated environment installed, do not set PYTHONPATH or activate
another repository. Before tests check C: ≥20 GiB, RAM available ≥1.5 GiB and
swap ≤256 MiB. Acquire the shared nonblocking lock and use one process:

    flock -n /tmp/lab-round3-tests.lock timeout 300s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NEW_VENV/bin/python -m pytest tests

Historical STU-Net staging imports are lazy: only launching that separate
laboratory campaign requires its old repository. Unit tests must not launch it.

## Demo

Prepare and inspect without training:

    NEW_VENV/bin/shadow-trainer demo --cpu --prepare-only --output-dir NEW_FIXTURE
    NEW_VENV/bin/shadow-trainer plan NEW_FIXTURE/demo-job.json --json

For a physical demo, first verify NVIDIA compute-process availability, free
VRAM, C: ≥20 GiB plus artifact reserve, RAM ≥1.5 GiB and swap ≤256 MiB.
Use a fresh output and the shared lock. GPU workloads require explicit
authorization; never start them from a monitoring heartbeat.

    flock -n /tmp/lab-round3-tests.lock timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NEW_VENV/bin/shadow-trainer demo --output-dir NEW_GPU_DEMO

Acceptance: 12 finite steps, actual CUDA recorded, sync selected, cache ≤4096 B,
durable checkpoint, all standard JSON/JSONL files and offline HTML. Preserve
failures; never overwrite outputs. Standard reports are for local review and
may contain paths: sanitize before external distribution.

## Verified acceptance — 20 September 2026

A fresh isolated .venv was installed offline from the captured lock. pip check
passed; all 38 locked versions match; 21 installed Python modules match current
source hashes. No clinical environment or borrowed PYTHONPATH is needed.

The full suite passed: 65 tests, zero failures/errors/skips (27.289 s).
The installed CLI completed 12 finite Tiny3D steps on the physical RTX 3050 Ti
with CUDA 13, sync and 22 MiB peak reserved GPU memory. Its monitored command
took 9.342 s; this is one demonstration, not a speedup benchmark. Observed
minimum free disk was 35.786 GiB, minimum available RAM 3.937 GiB and maximum
swap 3.773 MiB. No guard tripped.

Evidence: docs/coordination/reproducible-mvp-acceptance-2026-09-20.json.
Physical report: research/workspace/mvp-physical-r1/demo/run/report.html.
Test results: research/workspace/mvp-tests-r1.xml.
Frozen built wheel: research/workspace/mvp-package-r1/shadow_trainer-0.1.0-py3-none-any.whl.
Wheel SHA-256: 7e17113f27ade40b3edf0c2269c07bcab9b7f2ddcb814fa40f31045714849f2f.

The reconstruction environment is documented for this platform, not certified
on arbitrary hardware. Rebuilding a wheel need not produce identical ZIP bytes;
the captured wheel and its hash identify the tested package. CPU recovery is
covered by the suite; this new physical demo was uninterrupted. Clinical r2
evaluation and long STU-Net Stage38 remain separate pending gates.
