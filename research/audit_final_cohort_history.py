#!/usr/bin/env python3
"""Inventory textual references to a frozen cohort without opening result content."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXCLUDED = ['*.nii', '*.nii.gz', '*.pt', '*.png', '*.pdf', '*.html', '*.log', 'resources.jsonl']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(path, shadow, lab):
    text = str(path)
    if '/stunet-campaign-20260915-r1/' in text:
        return 'evaluated_in_r1_same_final_cohort'
    if '/stunet-campaign-20260915-r2/' in text:
        return 'r2_campaign_or_partial_evaluation'
    if path == shadow / 'research/jobs/stunet-r2-final-frozen-plan.json':
        return 'current_frozen_plan'
    if path.is_relative_to(lab / 'manifests') or path.is_relative_to(lab / 'source_manifest'):
        return 'dataset_inventory_or_split_source'
    return 'unclassified_reference_requires_review'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--shadow-root', type=Path, required=True)
    parser.add_argument('--laboratory-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    cohort = json.loads(args.cohort.read_text())
    cases = cohort['evaluation_cases']
    if len(cases) != len(set(cases)) or not cases:
        raise ValueError('invalid frozen cohort')
    command = ['rg', '-l', '--no-ignore', '--hidden']
    for pattern in EXCLUDED:
        command.extend(['--glob', '!' + pattern])
    command.extend(['|'.join(cases), str(args.laboratory_root), str(args.shadow_root)])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    paths = sorted(Path(line) for line in result.stdout.splitlines() if line)
    entries = [{'path': str(path), 'classification': classify(path, args.shadow_root, args.laboratory_root)}
               for path in paths]
    classes = {}
    for row in entries:
        classes[row['classification']] = classes.get(row['classification'], 0) + 1
    conclusion = ('known_reused_final_cohort_not_new_independent_evaluation'
                  if classes.get('evaluated_in_r1_same_final_cohort') else
                  'no_prior_evaluation_reference_found_but_independence_not_proven')
    payload = {'schema': 'shadowtrainer.final-cohort-history-audit/v1',
               'status': 'passed' if not classes.get('unclassified_reference_requires_review') else 'review_required',
               'cohort_sha256': sha256(args.cohort), 'case_count': len(cases),
               'search_roots': [str(args.laboratory_root), str(args.shadow_root)],
               'excluded_binary_or_result_heavy_patterns': EXCLUDED,
               'entries': entries, 'classification_counts': classes,
               'conclusion': conclusion,
               'limits': ['Textual reference inventory only; absence does not prove never used.',
                          'Result files are named/classified but their metrics are not parsed.',
                          'External notebooks, deleted files, other machines and manual inspection are outside scope.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'status': payload['status'], 'references': len(entries),
                      'classes': classes, 'conclusion': conclusion}))


if __name__ == '__main__':
    main()
