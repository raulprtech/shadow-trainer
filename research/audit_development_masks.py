#!/usr/bin/env python3
"""Offline development-only mask audit. No torch, inference, or final split."""
import argparse
import json
import math
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import psutil

from prediction_provenance import sha256

GROUPS = {'classes': {'1': [1], '2': [2], '3': [3]},
          'hec': {'kidney_and_masses': [1, 2, 3], 'kidney_mass': [2, 3], 'tumor': [2]}}


def guard():
    if shutil.disk_usage('/mnt/c').free < 20 * 2**30 + 2**20:
        raise RuntimeError('physical_disk_floor')
    if psutil.virtual_memory().available < 512 * 2**20 or psutil.swap_memory().used > 192 * 2**20:
        raise RuntimeError('memory_guard')
    if psutil.Process().memory_info().rss > 1536 * 2**20:
        raise RuntimeError('audit_rss_guard')


def geometry(image, target, prediction):
    for other in (target, prediction):
        if len(image.shape) != 3 or image.shape != other.shape:
            raise ValueError('shape mismatch')
        if not np.isfinite(image.affine).all() or not np.isfinite(other.affine).all() or not np.allclose(image.affine, other.affine, rtol=0, atol=1e-5):
            raise ValueError('affine mismatch')
        if nib.aff2axcodes(image.affine) != nib.aff2axcodes(other.affine):
            raise ValueError('orientation mismatch')
        zooms = image.header.get_zooms()[:3]
        if not all(math.isfinite(x) and x > 0 for x in zooms) or not np.allclose(zooms, other.header.get_zooms()[:3], rtol=0, atol=1e-5):
            raise ValueError('spacing mismatch')


def overlap_counts(prediction, target, check=guard):
    result = {group: {name: dict(prediction_voxels=0, target_voxels=0, intersection_voxels=0)
                      for name in names} for group, names in GROUPS.items()}
    # Last-axis slabs follow NIfTI storage order, bounded to approximately 8 MiB
    # of decoded values even for large in-plane dimensions.
    depth = max(1, min(8, (8 * 2**20) // (max(1, target.shape[0] * target.shape[1]) * 16)))
    for start in range(0, target.shape[2], depth):
        check()
        p = np.asanyarray(prediction.dataobj[:, :, start:start + depth])
        t = np.asanyarray(target.dataobj[:, :, start:start + depth])
        if not np.isin(p, [0, 1, 2, 3]).all() or not np.isin(t, [0, 1, 2, 3]).all():
            raise ValueError('noninteger, nonfinite or unknown mask label')
        for group, names in GROUPS.items():
            for name, labels in names.items():
                pm, tm = np.isin(p, labels), np.isin(t, labels)
                item = result[group][name]
                item['prediction_voxels'] += int(pm.sum())
                item['target_voxels'] += int(tm.sum())
                item['intersection_voxels'] += int((pm & tm).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    guard()
    rows = []
    for session in args.session:
        arm = json.loads((session / 'arm_A/summary.json').read_text())
        ids = sorted(arm['configuration']['development_cases'])
        if len(ids) != 4 or len(set(ids)) != 4 or set(ids) & set(arm['configuration']['train_cases']):
            raise ValueError('development cohort invalid')
        cache = Path(json.loads((session / 'resolved-config.json').read_text())['training_cache'])
        for model in ('B0', 'A', 'B'):
            source = session / f'evaluation/development/{model}/summary.json'
            summary = json.loads(source.read_text())
            if summary['status'] != 'success' or summary['cohort'] != 'development' or sorted(row['case_id'] for row in summary['cases']) != ids:
                raise ValueError('summary split/completeness mismatch')
            for row in sorted(summary['cases'], key=lambda row: row['case_id']):
                case = row['case_id']
                image_path, label_path = (cache / case / name for name in ('imaging.nii.gz', 'segmentation.nii.gz'))
                pred_path = session / f'evaluation/development/{model}/predictions/{case}.nii.gz'
                paths = {'image': image_path, 'label': label_path, 'prediction': pred_path, 'summary': source}
                before = {name: sha256(path) for name, path in paths.items()}
                if before['prediction'] != row['prediction_sha256']:
                    raise ValueError('prediction hash mismatch')
                image, target, prediction = (nib.load(p, keep_file_open=True) for p in (image_path, label_path, pred_path))
                geometry(image, target, prediction)
                if list(target.shape) != row['shape'] or not np.allclose(target.header.get_zooms()[:3], row['spacing'], rtol=0, atol=1e-5):
                    raise ValueError('summary geometry mismatch')
                measured = overlap_counts(prediction, target)
                for group, names in measured.items():
                    for name, counts in names.items():
                        for key, value in counts.items():
                            if row[group][name][key] != value:
                                raise ValueError('mask/count mismatch')
                if before != {name: sha256(path) for name, path in paths.items()}:
                    raise ValueError('source changed during audit')
                rows.append({'session': session.name, 'model': model, 'case': f'D{ids.index(case) + 1}',
                             'hashes': before, 'counts_match': True, 'geometry_match': True,
                             'counts': measured})
                del image, target, prediction
                print(f'{session.name} {model} D{ids.index(case) + 1}: verified', flush=True)
    guard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'schema': 'shadowtrainer.development-mask-audit/v1', 'status': 'passed',
                   'auditor_sha256': sha256(Path(__file__)), 'rows': rows,
                   'limits': ['Development only; no clinical validation.', 'No HD95 recomputation.',
                              'Current-source consistency does not retroactively prove checkpoint-to-prediction provenance.']}, stream, indent=2)


if __name__ == '__main__':
    main()
