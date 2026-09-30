#!/usr/bin/env python3
"""Fail-closed supervisor for a frozen final cohort.

Default is metadata-only preflight. Real data reads/evaluation require all three:
--execute-final, --acknowledge-cohort-reuse and the exact plan SHA-256.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil


GIB, MIB = 2**30, 2**20


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_plan(path):
    plan = json.loads(path.read_text())
    required = {'schema', 'status', 'source_session', 'cohort', 'inputs', 'checkpoints',
                'source_hashes', 'work', 'metrics', 'resources', 'gates_before_execution'}
    if plan.get('schema') != 'shadowtrainer.frozen-evaluation-protocol/v1' or not required <= set(plan):
        raise ValueError('invalid frozen plan schema')
    if plan['status'] != 'prepared_not_authorized_for_execution':
        raise ValueError('unexpected frozen plan status')
    cases = plan['cohort']['case_ids_in_frozen_order']
    if len(cases) != 6 or len(set(cases)) != 6:
        raise ValueError('expected six unique frozen cases')
    if plan['work']['model_order'] != ['B0', 'A', 'B'] or not plan['work']['new_output_required']:
        raise ValueError('unexpected model order/output policy')
    return plan


def metadata_preflight(plan, source, output):
    issues = []
    if output.exists():
        issues.append('output_exists')
    if source.name != plan['source_session']:
        issues.append('source_session_mismatch')
    if plan['cohort']['same_final_cohort_as_r1'] is not True:
        issues.append('reuse_not_declared')
    if plan['cohort']['historical_independence'] != 'not_established_by_schedule_only_audit':
        issues.append('independence_status_changed')
    for model, row in plan['checkpoints'].items():
        path = Path(row['path'])
        if not path.is_file() or path.stat().st_size != row['size_bytes']:
            issues.append(f'checkpoint_metadata:{model}')
    for row in plan['inputs']:
        path = Path(row['path'])
        if not path.is_file() or path.stat().st_size != row['frozen_size_bytes']:
            issues.append(f"input_metadata:{row['case_id']}:{row['kind']}")
    r = plan['resources']
    total = (r['frozen_input_bytes'] + r['referenced_checkpoint_bytes'] +
             r['reserved_overhead_bytes'] + r['new_artifacts_and_temporaries_cap_bytes'])
    if total != r['combined_budget_bytes'] or r['physical_disk_floor_bytes'] != 20 * GIB:
        issues.append('resource_budget_arithmetic')
    free = shutil.disk_usage('/mnt/c').free
    if free < r['physical_disk_floor_bytes'] + r['new_artifacts_and_temporaries_cap_bytes']:
        issues.append('physical_disk_admission')
    return {'status': 'pass' if not issues else 'fail', 'issues': issues,
            'physical_free_bytes': free, 'content_hashes_read': False,
            'final_results_read': False, 'output_created': False}


def authorize(args, plan_sha):
    if not args.execute_final:
        return
    if not args.acknowledge_cohort_reuse:
        raise PermissionError('execution requires --acknowledge-cohort-reuse')
    if args.confirm_plan_sha != plan_sha:
        raise PermissionError('execution requires exact --confirm-plan-sha')


def resource_snapshot(pid=None):
    rss = 0
    if pid:
        try:
            parent = psutil.Process(pid)
            for process in [parent, *parent.children(recursive=True)]:
                try: rss += process.memory_info().rss
                except psutil.NoSuchProcess: pass
        except psutil.NoSuchProcess:
            pass
    return {'time': time.time(), 'disk_free_bytes': shutil.disk_usage('/mnt/c').free,
            'ram_available_bytes': psutil.virtual_memory().available,
            'swap_used_bytes': psutil.swap_memory().used, 'tree_rss_bytes': rss}


def enforce_resources(sample, limits, output):
    if sample['disk_free_bytes'] < limits['physical_disk_floor_bytes']:
        raise RuntimeError('physical_disk_floor')
    if sample['ram_available_bytes'] < limits['min_available_ram_bytes']:
        raise RuntimeError('available_ram_floor')
    if sample['swap_used_bytes'] > limits['max_swap_bytes']:
        raise RuntimeError('swap_limit')
    if sample['tree_rss_bytes'] > limits['max_tree_rss_bytes']:
        raise RuntimeError('tree_rss_limit')
    size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file() and not path.is_symlink())
    if size > limits['new_artifacts_and_temporaries_cap_bytes']:
        raise RuntimeError('new_artifact_budget')


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=15)
    return not result.stdout.strip()


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()


def verify_content(plan):
    # Deliberately separate from metadata preflight: this reads the final inputs.
    rows = []
    for row in plan['inputs']:
        path = Path(row['path'])
        md5 = hashlib.md5(usedforsecurity=False)
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                md5.update(block); sha.update(block)
        if row['frozen_md5'] and md5.hexdigest() != row['frozen_md5']:
            raise ValueError('frozen input MD5 mismatch')
        rows.append({**row, 'sha256': sha.hexdigest()})
    for model, row in plan['checkpoints'].items():
        if sha256(row['path']) != row['sha256']:
            raise ValueError(f'checkpoint SHA mismatch: {model}')
    for path, digest in plan['source_hashes'].items():
        if sha256(path) != digest:
            raise ValueError(f'source changed after freeze: {path}')
    return rows


def execute(plan, plan_sha, source, output, verified_inputs):
    # This function is intentionally unreachable without authorize().
    if not gpu_idle():
        raise RuntimeError('GPU compute process already active')
    output.mkdir(parents=True, exist_ok=False)
    def write_json(path, payload):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path)
    write_json(output / 'plan-copy.json', plan)
    write_json(output / 'verified-inputs.json', verified_inputs)
    cohort = {'evaluation_cases': plan['cohort']['case_ids_in_frozen_order']}
    write_json(output / 'cohort.json', cohort)
    source_config = json.loads((source / 'resolved-config.json').read_text())
    config = {'seed': source_config['seed'], 'base_checkpoint': plan['checkpoints']['B0']['path'],
              'evaluation_cache': source_config['evaluation_cache']}
    write_json(output / 'config.json', config)
    for model in ('A', 'B'):
        arm = output / f'arm_{model}'
        arm.mkdir()
        (arm / 'best.checkpoint.pt').symlink_to(plan['checkpoints'][model]['path'])
    limits = plan['resources']
    evaluator = Path(__file__).with_name('stunet_campaign_evaluate.py')
    started = time.monotonic()
    campaign = {'schema': 'shadowtrainer.frozen-final-run/v1', 'status': 'running',
                'plan_sha256': plan_sha, 'models': {},
                'cohort_reuse_acknowledged': True, 'telemetry_samples': 0}
    write_json(output / 'campaign-summary.json', campaign)
    with (output / 'resources.jsonl').open('x') as telemetry:
        try:
            for model in plan['work']['model_order']:
                success = False
                attempts = []
                for attempt in (1, 2):
                    if time.monotonic() - started >= limits['global_time_cap_seconds'] - limits['closing_reserve_seconds']:
                        raise RuntimeError('global_closing_reserve')
                    command = [sys.executable, str(evaluator), '--config', str(output / 'config.json'),
                               '--session', str(output), '--model', model]
                    log_path = output / f'evaluate-{model}-attempt-{attempt}.log'
                    attempt_started = time.monotonic()
                    process = None
                    reason = None
                    try:
                        with log_path.open('x') as log:
                            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
                            while process.poll() is None:
                                sample = resource_snapshot(process.pid)
                                telemetry.write(json.dumps({'model': model, 'attempt': attempt, **sample}) + '\n')
                                telemetry.flush(); campaign['telemetry_samples'] += 1
                                enforce_resources(sample, limits, output)
                                if time.monotonic() - attempt_started > limits['invocation_process_cap_seconds']:
                                    reason = 'invocation_time_cap'
                                    raise RuntimeError(reason)
                                if time.monotonic() - started >= limits['global_time_cap_seconds'] - limits['closing_reserve_seconds']:
                                    reason = 'global_closing_reserve'
                                    raise RuntimeError(reason)
                                try: process.wait(timeout=1)
                                except subprocess.TimeoutExpired: pass
                        if process.returncode:
                            reason = f'exit_{process.returncode}'
                        else:
                            summary_path = output / f'evaluation/evaluation/{model}/summary.json'
                            summary = json.loads(summary_path.read_text())
                            cases = summary.get('cases', [])
                            expected = plan['cohort']['case_ids_in_frozen_order']
                            if (summary.get('status') != 'success' or [row['case_id'] for row in cases] != expected
                                    or not all(row.get('provenance_verified') for row in cases)):
                                reason = 'incomplete_or_unverified_summary'
                            else:
                                success = True
                                campaign['models'][model] = {'summary': str(summary_path),
                                  'summary_sha256': sha256(summary_path), 'cases': len(cases),
                                  'aggregate': summary['aggregate'], 'attempts': [*attempts, attempt]}
                    except Exception as exc:
                        reason = reason or str(exc)
                        if process is not None: stop_process(process)
                    attempts.append({'attempt': attempt, 'elapsed_seconds': time.monotonic() - attempt_started,
                                     'status': 'success' if success else 'failed', 'reason': reason})
                    if success: break
                    # Deterministic evaluator/data errors are not made better by retrying.
                    if reason not in ('invocation_time_cap',): break
                if not success:
                    raise RuntimeError(f'model_{model}_incomplete:{attempts[-1]["reason"]}')
                campaign['models'][model]['attempt_details'] = attempts
                write_json(output / 'campaign-summary.json', campaign)
            campaign.update(status='success', elapsed_seconds=time.monotonic() - started)
        except Exception as exc:
            campaign.update(status='incomplete', error=str(exc), elapsed_seconds=time.monotonic() - started)
        finally:
            write_json(output / 'campaign-summary.json', campaign)
    return 0 if campaign['status'] == 'success' else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-final', action='store_true')
    parser.add_argument('--acknowledge-cohort-reuse', action='store_true')
    parser.add_argument('--confirm-plan-sha')
    args = parser.parse_args()
    plan_sha = sha256(args.plan)
    plan = load_plan(args.plan)
    preflight = metadata_preflight(plan, args.source.resolve(), args.output.resolve())
    response = {'schema': 'shadowtrainer.frozen-final-preflight/v1', 'plan_sha256': plan_sha,
                'mode': 'execute' if args.execute_final else 'metadata_preflight', **preflight}
    if preflight['status'] != 'pass':
        print(json.dumps(response, indent=2)); return 2
    authorize(args, plan_sha)
    if not args.execute_final:
        print(json.dumps(response, indent=2)); return 0
    source = args.source.resolve()
    source_config = json.loads((source / 'resolved-config.json').read_text())
    lock_path = Path(source_config['laboratory_root']) / '.stage23_supervisor.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified = verify_content(plan)
        return execute(plan, plan_sha, source, args.output.resolve(), verified)


if __name__ == '__main__':
    raise SystemExit(main())
