import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'research'))
import run_frozen_final_evaluation as supervisor


class FrozenFinalSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / 'r2'; source.mkdir()
        self.source, self.output = source, root / 'new-output'
        checkpoints = {}
        inputs = []
        for model in ('B0', 'A', 'B'):
            path = root / f'{model}.pt'; path.write_bytes(model.encode())
            checkpoints[model] = {'path': str(path), 'sha256': supervisor.sha256(path),
                                  'size_bytes': path.stat().st_size, 'selection': 'fixture'}
        for case in range(6):
            for kind in ('image', 'label'):
                path = root / f'{case}-{kind}.bin'; path.write_bytes(f'{case}-{kind}'.encode())
                inputs.append({'case_id': f'C{case}', 'kind': kind, 'path': str(path),
                               'frozen_size_bytes': path.stat().st_size, 'frozen_md5': None,
                               'exists': True, 'stat_size_matches': True})
        self.plan = {'schema': 'shadowtrainer.frozen-evaluation-protocol/v1',
                     'status': 'prepared_not_authorized_for_execution', 'source_session': 'r2',
                     'cohort': {'case_ids_in_frozen_order': [f'C{x}' for x in range(6)],
                                'same_final_cohort_as_r1': True,
                                'historical_independence': 'not_established_by_schedule_only_audit'},
                     'inputs': inputs, 'checkpoints': checkpoints, 'source_hashes': {},
                     'work': {'model_order': ['B0','A','B'], 'new_output_required': True},
                     'metrics': {}, 'resources': {'frozen_input_bytes': 12, 'referenced_checkpoint_bytes': 6,
                       'reserved_overhead_bytes': 10, 'new_artifacts_and_temporaries_cap_bytes': 12,
                       'combined_budget_bytes': 40, 'physical_disk_floor_bytes': 20 * 2**30},
                     'gates_before_execution': []}

    def test_valid_metadata_preflight_does_not_read_hashes_or_create_output(self):
        with patch.object(supervisor.shutil, 'disk_usage', return_value=type('D', (), {'free': 30 * 2**30})()):
            result = supervisor.metadata_preflight(self.plan, self.source, self.output)
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['content_hashes_read'])
        self.assertFalse(self.output.exists())

    def test_existing_output_rejected(self):
        self.output.mkdir()
        with patch.object(supervisor.shutil, 'disk_usage', return_value=type('D', (), {'free': 30 * 2**30})()):
            self.assertIn('output_exists', supervisor.metadata_preflight(self.plan, self.source, self.output)['issues'])

    def test_bad_cohort_and_order_rejected(self):
        for mutation in ('duplicate', 'order'):
            plan = copy.deepcopy(self.plan)
            if mutation == 'duplicate': plan['cohort']['case_ids_in_frozen_order'][-1] = 'C0'
            else: plan['work']['model_order'] = ['A', 'B0', 'B']
            with self.assertRaises(ValueError): supervisor.load_plan(self._write(plan))

    def test_execution_needs_both_acknowledgements(self):
        args = type('A', (), {'execute_final': True, 'acknowledge_cohort_reuse': False,
                              'confirm_plan_sha': 'abc'})()
        with self.assertRaises(PermissionError): supervisor.authorize(args, 'abc')
        args.acknowledge_cohort_reuse = True; args.confirm_plan_sha = 'wrong'
        with self.assertRaises(PermissionError): supervisor.authorize(args, 'abc')

    def test_checkpoint_change_detected_only_in_content_phase(self):
        Path(self.plan['checkpoints']['A']['path']).write_bytes(b'changed')
        self.plan['checkpoints']['A']['size_bytes'] = len(b'changed')
        with patch.object(supervisor.shutil, 'disk_usage', return_value=type('D', (), {'free': 30 * 2**30})()):
            self.assertEqual(supervisor.metadata_preflight(self.plan, self.source, self.output)['status'], 'pass')
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA'):
            supervisor.verify_content(self.plan)

    def test_low_disk_rejected(self):
        with patch.object(supervisor.shutil, 'disk_usage', return_value=type('D', (), {'free': 20 * 2**30})()):
            result = supervisor.metadata_preflight(self.plan, self.source, self.output)
        self.assertIn('physical_disk_admission', result['issues'])

    def test_resource_guards(self):
        limits = {'physical_disk_floor_bytes': 100, 'min_available_ram_bytes': 50,
                  'max_swap_bytes': 10, 'max_tree_rss_bytes': 20,
                  'new_artifacts_and_temporaries_cap_bytes': 1000}
        good = {'disk_free_bytes': 100, 'ram_available_bytes': 50,
                'swap_used_bytes': 10, 'tree_rss_bytes': 20}
        supervisor.enforce_resources(good, limits, self.output)
        for key, value in [('disk_free_bytes', 99), ('ram_available_bytes', 49),
                           ('swap_used_bytes', 11), ('tree_rss_bytes', 21)]:
            sample = {**good, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                supervisor.enforce_resources(sample, limits, self.output)

    def test_execution_path_still_requires_authorize(self):
        args = type('A', (), {'execute_final': False, 'acknowledge_cohort_reuse': False,
                              'confirm_plan_sha': None})()
        supervisor.authorize(args, 'abc')
        self.assertFalse(self.output.exists())

    def _write(self, plan):
        path = Path(self.temp.name) / 'plan.json'; path.write_text(json.dumps(plan)); return path


if __name__ == '__main__': unittest.main()
