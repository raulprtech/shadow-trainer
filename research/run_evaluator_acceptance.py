#!/usr/bin/env python3
"""Bounded synthetic CPU acceptance; a skipped scenario is NOT acceptance."""
import argparse
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import psutil
from prediction_provenance import sha256


def snapshot():
    return {'c_free_bytes': shutil.disk_usage('/mnt/c').free,
            'ram_available_bytes': psutil.virtual_memory().available,
            'swap_used_bytes': psutil.swap_memory().used}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    before = snapshot()
    if before['c_free_bytes'] < 20 * 2**30 + 2**20 or before['ram_available_bytes'] < 1536 * 2**20 or before['swap_used_bytes'] > 192 * 2**20:
        raise RuntimeError('acceptance resource preflight rejected')
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    for pattern in ('test_prediction_provenance.py', 'test_stunet_evaluator_flow.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(root / 'tests'), pattern=pattern))
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=2).run(suite)
    import torch
    ok = result.wasSuccessful() and not result.skipped and result.testsRun == 17 and not torch.cuda.is_initialized()
    sources = ['research/run_evaluator_acceptance.py', 'research/stunet_campaign_evaluate.py',
               'research/prediction_provenance.py', 'research/audit_development_masks.py',
               'tests/test_prediction_provenance.py', 'tests/test_stunet_evaluator_flow.py']
    report = {'schema': 'shadowtrainer.evaluator-cpu-acceptance/v1', 'status': 'passed' if ok else 'failed',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'failed_test_ids': [t.id() for t, _ in result.failures + result.errors],
              'elapsed_seconds': time.monotonic() - started, 'cuda_initialized': torch.cuda.is_initialized(),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'resources_before': before, 'resources_after': snapshot(),
              'python': sys.version.split()[0], 'torch': str(torch.__version__),
              'hashes': {p: sha256(root / p) for p in sources},
              'scope': 'real evaluator control flow and volume metrics, injected synthetic Conv3d on CPU',
              'limits': ['No STU-Net model acceptance or GPU validation.', 'No clinical data or final cohort.',
                         'Overlay rendering mocked; CUDA operations mocked.', 'Existing scientific environment; not standalone MVP packaging.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
