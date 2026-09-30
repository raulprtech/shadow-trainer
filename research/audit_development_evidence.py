#!/usr/bin/env python3
"""Read-only STU-Net development audit. Never opens final-cohort results.

No torch, GPU, model loading, source images, or labels. Hashes bind existing
predictions/checkpoints; overlap scores are recomputed from stored counts,
NOT independently from masks. Outputs use development aliases, not case IDs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equal(actual, expected):
    return actual is expected if actual is None or expected is None else math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def counts(item):
    p, t, i = (item[k] for k in ('prediction_voxels', 'target_voxels', 'intersection_voxels'))
    require(all(isinstance(x, int) and x >= 0 for x in (p, t, i)) and i <= min(p, t), 'invalid counts')
    values = {'dice': 2 * i / (p + t) if p + t else None,
              'precision': i / p if p else (1.0 if not t else 0.0),
              'recall': i / t if t else (1.0 if not p else None)}
    for key, value in values.items():
        require(equal(item[key], value), 'count/metric mismatch: ' + key)
    return values


def audit(sessions):
    hashes, summaries, per_case, reference = {}, [], [], None
    for session in sessions:
        arms = {}
        for arm in ('A', 'B'):
            path = session / f'arm_{arm}/summary.json'
            arms[arm] = json.loads(path.read_text())
            hashes[f'{session.name}/arm_{arm}/summary.json'] = digest(path)
            require(arms[arm]['status'] == 'success', 'incomplete training')
        require(arms['A']['configuration'] == arms['B']['configuration'], 'unpaired training schedules')
        require(arms['A']['base_sha256'] == arms['B']['base_sha256'], 'different initialization')
        config = arms['A']['configuration']
        ids = sorted(config['development_cases'])
        require(len(ids) == len(set(ids)) == 4, 'expected four unique development cases')
        require(not set(ids) & set(config['train_cases']), 'train/development overlap')
        aliases = {case: f'D{index + 1}' for index, case in enumerate(ids)}
        expected = len(config['train_cases']) * config['epochs'] * config['patches_per_case']
        for arm in arms.values():
            require(arm['global_step'] == expected, 'update count mismatch')
        models = {}
        for model in ('B0', 'A', 'B'):
            path = session / f'evaluation/development/{model}/summary.json'
            data = json.loads(path.read_text())
            hashes[f'{session.name}/development/{model}/summary.json'] = digest(path)
            require(data['status'] == 'success' and data['cohort'] == 'development', 'incomplete or wrong split')
            rows = sorted(data['cases'], key=lambda row: row['case_id'])
            require(sorted(data['cases_requested']) == ids == [row['case_id'] for row in rows], 'case mismatch')
            cp = Path(data['checkpoint'])
            require(cp.is_file() and digest(cp) == data['checkpoint_sha256'], 'checkpoint digest mismatch')
            if model == 'B0':
                require(data['checkpoint_sha256'] == arms['A']['base_sha256'], 'baseline digest mismatch')
            models[model] = {'checkpoint_sha256': data['checkpoint_sha256'], 'metrics': {}}
            signature = [(row['case_id'], row['shape'], row['spacing'],
                          {group: {key: item['target_voxels'] for key, item in row[group].items()} for group in ('classes', 'hec')}) for row in rows]
            if reference is None:
                reference = signature
            require(signature == reference, 'target or geometry mismatch between models/runs')
            for row in rows:
                pred = session / f"evaluation/development/{model}/predictions/{row['case_id']}.nii.gz"
                require(pred.is_file() and digest(pred) == row['prediction_sha256'], 'prediction digest mismatch')
                hashes[f'{session.name}/development/{model}/{aliases[row["case_id"]]}.nii.gz'] = row['prediction_sha256']
                for group in ('classes', 'hec'):
                    for name, item in row[group].items():
                        values = counts(item)
                        per_case.append({'session': session.name, 'model': model, 'case': aliases[row['case_id']],
                                         'metric': f'{group}/{name}', **values, 'surface_status': item.get('surface_status')})
            for group in ('classes', 'hec'):
                for name in rows[0][group]:
                    items = [row[group][name] for row in rows]
                    ds = [x['dice'] for x in items if x['dice'] is not None]
                    den = sum(x['prediction_voxels'] + x['target_voxels'] for x in items)
                    stats = {'mean_dice': statistics.mean(ds) if ds else None,
                             'median_dice': statistics.median(ds) if ds else None,
                             'micro_dice': 2 * sum(x['intersection_voxels'] for x in items) / den if den else None}
                    for key, value in stats.items():
                        require(equal(data['aggregate'][group][name][key], value), 'aggregate mismatch')
                    models[model]['metrics'][f'{group}/{name}'] = {**stats, 'total_cases': len(items), 'defined_dice_cases': len(ds)}
            if model != 'B0':
                require(equal(models[model]['metrics']['classes/2']['mean_dice'], arms[model]['best_tumor']), 'selection mismatch')
        comparisons = {}
        for candidate, baseline in (('A', 'B0'), ('B', 'B0'), ('B', 'A')):
            values = {}
            for metric in ('classes/1', 'classes/2', 'hec/kidney_and_masses'):
                left = {x['case']: x['dice'] for x in per_case if x['session'] == session.name and x['model'] == candidate and x['metric'] == metric}
                right = {x['case']: x['dice'] for x in per_case if x['session'] == session.name and x['model'] == baseline and x['metric'] == metric}
                ds = [left[k] - right[k] for k in left if left[k] is not None and right[k] is not None]
                values[metric] = {'mean_delta': statistics.mean(ds), 'median_delta': statistics.median(ds), 'improved': sum(x > 0 for x in ds), 'n': len(ds)}
            comparisons[f'{candidate}-{baseline}'] = values
        summaries.append({'session': session.name, 'training_seed': config['seed'], 'train_cases': len(config['train_cases']),
                          'development_cases': len(ids), 'epochs': config['epochs'], 'updates_per_arm': expected,
                          'best_epochs': {a: arms[a]['best_epoch'] for a in arms}, 'models': models, 'comparisons': comparisons})
    return {'schema': 'shadowtrainer.development-audit/v1', 'scope': 'development_only_counts_and_artifact_integrity',
            'source_hashes': hashes, 'sessions': summaries, 'per_case': per_case,
            'limitations': ['Same four model-selection patients across runs; not eight independent patients.',
                            'No final/test results opened; no independent mask-to-count recomputation.',
                            'Current artifact hashes do not prove execution-time code identity or absence of previous leakage.',
                            'No clinical efficacy, statistical significance, or generalization claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output must be new')
    require(shutil.disk_usage('/mnt/c').free >= 20 * 2**30 + 2**20, 'physical disk floor')
    result = audit(args.session)
    result['auditor_sha256'] = digest(Path(__file__))
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'development-per-case.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_case'][0]))
        writer.writeheader()
        writer.writerows(result['per_case'])
    print(json.dumps({'status': 'passed', 'sessions': len(result['sessions']), 'rows': len(result['per_case']), 'hashes': len(result['source_hashes'])}))


if __name__ == '__main__':
    main()
