#!/usr/bin/env python3
"""Explicit bounded physical validation: A/B, first development case, new output.

No training, staging, final split or alteration of historical predictions.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
from prediction_provenance import sha256, verify
from stunet_campaign import atomic_json

GIB, MIB = 2**30, 2**20


def resources(pid=None):
    rss = 0
    if pid:
        try:
            parent = psutil.Process(pid)
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
    return {'time': time.time(), 'free_disk_bytes': shutil.disk_usage('/mnt/c').free,
            'ram_available_bytes': psutil.virtual_memory().available,
            'swap_used_bytes': psutil.swap_memory().used, 'tree_rss_bytes': rss}


def guard(sample, initial=False):
    if sample['free_disk_bytes'] < 20 * GIB + (256 * MIB if initial else 0):
        raise RuntimeError('physical_disk_floor')
    if sample['ram_available_bytes'] < 1536 * MIB:
        raise RuntimeError('evaluation_ram_floor')
    if sample['swap_used_bytes'] > 192 * MIB or sample['tree_rss_bytes'] > 4.5 * GIB:
        raise RuntimeError('evaluation_swap_or_rss_guard')


def nvidia(query):
    return subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def execute(source, output, candidate):
    if candidate not in ("A", "B"):
        raise ValueError("candidate must be A or B")
    guard(resources(), initial=True)
    if output.exists():
        raise ValueError('output must be new')
    if nvidia('--query-compute-apps=pid'):
        raise RuntimeError('GPU compute process already active')
    gpu = nvidia('--query-gpu=name,driver_version,memory.total,memory.free')
    if int(gpu.split(',')[-1].strip()) < 2048:
        raise RuntimeError('GPU headroom below 2 GiB')
    config = json.loads((source / 'resolved-config.json').read_text())
    arm = json.loads((source / 'arm_A/summary.json').read_text())
    cases = sorted(arm['configuration']['development_cases'])
    if len(cases) != 4 or len(set(cases)) != 4 or set(cases) & set(arm['configuration']['train_cases']):
        raise ValueError('development cohort invalid')
    case = cases[0]  # selection rule never consults performance
    historical_path = source / f'evaluation/development/{candidate}/summary.json'
    historical = json.loads(historical_path.read_text())
    old = next(row for row in historical['cases'] if row['case_id'] == case)
    checkpoint = source / f'arm_{candidate}/best.checkpoint.pt'
    if sha256(checkpoint) != historical['checkpoint_sha256']:
        raise ValueError('candidate checkpoint mismatch')
    cache = Path(config['training_cache'])
    input_paths = {name: cache / case / filename for name, filename in
                   [('image', 'imaging.nii.gz'), ('label', 'segmentation.nii.gz')]}
    old_prediction = source / f'evaluation/development/{candidate}/predictions/{case}.nii.gz'
    input_paths.update(checkpoint=checkpoint, historical_summary=historical_path, historical_prediction=old_prediction)
    before = {name: sha256(path) for name, path in input_paths.items()}
    if before['historical_prediction'] != old['prediction_sha256']:
        raise ValueError('historical prediction mismatch')
    output.mkdir(parents=True)
    arm_output = output / f'arm_{candidate}'
    arm_output.mkdir()
    (arm_output / 'best.checkpoint.pt').symlink_to(checkpoint)
    atomic_json(output / 'cohort.json', {'development_cases': [case]})
    atomic_json(output / 'config.json', {'seed': config['seed'], 'base_checkpoint': str(checkpoint),
                                        'training_cache': str(cache)})
    started = time.monotonic()
    result = {'schema': 'shadowtrainer.stunet-development-physical/v1', 'status': 'running',
              'case_alias': 'D1', 'selection': 'first sorted development ID', 'model': candidate, 'source_session': source.name,
              'gpu': gpu, 'input_hashes': before, 'phases': [],
              'scope': 'one real development case, inference and saved-prediction reuse; no training'}
    samples = []
    process = None
    try:
        command = [sys.executable, str(Path(__file__).with_name('stunet_campaign_evaluate.py')),
                   '--config', str(output / 'config.json'), '--session', str(output), '--model', candidate, '--development']
        with (output / 'resources.jsonl').open('x') as telemetry:
            for phase in ('initial', 'reuse'):
                phase_started = time.monotonic()
                with (output / f'{phase}.log').open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    while process.poll() is None:
                        sample = resources(process.pid)
                        samples.append(sample)
                        telemetry.write(json.dumps(sample) + '\n')
                        telemetry.flush()
                        guard(sample)
                        if time.monotonic() - started > 600:
                            raise RuntimeError('ten_minute_time_budget')
                        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 256 * MIB:
                            raise RuntimeError('artifact_budget')
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                if process.returncode:
                    raise RuntimeError(f'{phase}_exit_{process.returncode}')
                summary = json.loads((output / f'evaluation/development/{candidate}/summary.json').read_text())
                atomic_json(output / f'{phase}-summary.json', summary)
                if summary['status'] != 'success' or len(summary['cases']) != 1:
                    raise ValueError('incomplete physical evaluation')
                row = summary['cases'][0]
                prediction = output / f'evaluation/development/{candidate}/predictions/{case}.nii.gz'
                expected = {'checkpoint_sha256': before['checkpoint'], 'image_sha256': before['image'],
                            'label_sha256': before['label'], 'protocol': summary['protocol']}
                receipt = verify(prediction, expected)
                if phase == 'initial':
                    first = summary
                    prediction_digest = receipt['prediction_sha256']
                    if row['peak_allocated_mib'] <= 0 or row.get('resumed_from_prediction'):
                        raise ValueError('expected actual CUDA inference')
                elif (summary['aggregate'] != first['aggregate'] or not row.get('resumed_from_prediction')
                      or receipt['prediction_sha256'] != prediction_digest):
                    raise ValueError('reuse mismatch')
                result['phases'].append({'phase': phase, 'elapsed_seconds': time.monotonic() - phase_started,
                                         'prediction_sha256': receipt['prediction_sha256']})
        # Independent slab comparison and geometry, CPU only after the child exits.
        import nibabel as nib
        import numpy as np
        from audit_development_masks import geometry, overlap_counts
        image, target, current = [nib.load(p, keep_file_open=True) for p in
                                   (input_paths['image'], input_paths['label'], prediction)]
        geometry(image, target, current)
        counts = overlap_counts(current, target, check=lambda: guard(resources()))
        for group, names in counts.items():
            for name, values in names.items():
                if any(first['cases'][0][group][name][k] != v for k, v in values.items()):
                    raise ValueError('new prediction count mismatch')
        historical_nii = nib.load(old_prediction, keep_file_open=True)
        geometry(image, target, historical_nii)
        equal = True
        for start in range(0, current.shape[2], 4):
            guard(resources())
            equal &= np.array_equal(current.dataobj[:, :, start:start + 4], historical_nii.dataobj[:, :, start:start + 4])
        if before != {name: sha256(path) for name, path in input_paths.items()}:
            raise ValueError('historical input changed')
        result['arrays_equal_to_historical'] = bool(equal)
        if not equal:
            raise ValueError('candidate did not reproduce historical prediction')
        result.update(status='passed', arrays_equal_to_historical=bool(equal), geometry_verified=True,
                      mask_counts_verified=True, aggregate=first['aggregate'],
                      peak_allocated_mib=first['peak_allocated_mib'], peak_reserved_mib=first['peak_reserved_mib'],
                      historical_files_unchanged=True)
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result['elapsed_seconds'] = time.monotonic() - started
        result['resource_samples'] = len(samples)
        if samples:
            result['min_free_disk_bytes'] = min(s['free_disk_bytes'] for s in samples)
            result['min_ram_available_bytes'] = min(s['ram_available_bytes'] for s in samples)
            result['max_swap_bytes'] = max(s['swap_used_bytes'] for s in samples)
            result['max_tree_rss_bytes'] = max(s['tree_rss_bytes'] for s in samples)
        result['code_hashes'] = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('stunet_campaign_evaluate.py'))}
        atomic_json(output / 'acceptance.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=('A', 'B'), required=True)
    args = parser.parse_args()
    with Path('/home/raulprtech/stream-hot-kits-mini/.stage23_supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(execute(args.source.resolve(), args.output.resolve(), args.candidate))
