"""CPU integration of the real evaluator; synthetic data and injected tiny model.

Run in the existing scientific environment (scipy/matplotlib required).
CUDA calls and model construction are replaced, NOT metric/receipt/CLI logic.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'research'))
if importlib.util.find_spec('scipy') is None:
    raise unittest.SkipTest('Scientific evaluator requires scipy; run with the scientific environment, not the minimal MVP environment.')
import stunet_campaign_evaluate as evaluator
from prediction_provenance import receipt_path


class EvaluatorFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='shadow-evaluator-flow-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = 'synthetic_001'
        self.cache = self.root / 'cache'
        folder = self.cache / self.case
        folder.mkdir(parents=True)
        self.image = folder / 'imaging.nii.gz'
        self.label = folder / 'segmentation.nii.gz'
        data = np.zeros((8, 9, 10), dtype=np.float32)
        target = np.zeros(data.shape, dtype=np.uint8)
        target[2:6, 2:6, 2:6] = 1
        target[3:5, 3:5, 3:5] = 2
        nib.save(nib.Nifti1Image(data, np.eye(4)), self.image)
        nib.save(nib.Nifti1Image(target, np.eye(4)), self.label)
        self.model = torch.nn.Conv3d(1, 4, kernel_size=1)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.copy_(torch.tensor([0., 1., 0., 0.]))
        self.checkpoint = self.root / 'checkpoint.pt'
        torch.save({'model_state': self.model.state_dict()}, self.checkpoint)
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'seed': 1, 'base_checkpoint': str(self.checkpoint),
                                          'training_cache': str(self.cache)}))
        # No final-cohort key exists: accidental access must fail.
        (self.root / 'cohort.json').write_text(json.dumps({'development_cases': [self.case]}))
        self.output = self.root / 'evaluation/development/B0'
        self.prediction = self.output / f'predictions/{self.case}.nii.gz'
        self.summary = self.output / 'summary.json'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(sys, 'argv', ['evaluator', '--config', str(self.config),
                                                          '--session', str(self.root), '--model', 'B0', '--development']))
        cpu = torch.device('cpu')
        self.stack.enter_context(patch.object(evaluator, 'evaluation_device', return_value=cpu))
        self.stack.enter_context(patch.object(evaluator.torch.amp, 'autocast', side_effect=lambda *a, **k: nullcontext()))
        for name in ('manual_seed_all', 'empty_cache', 'reset_peak_memory_stats', 'synchronize'):
            self.stack.enter_context(patch.object(evaluator.torch.cuda, name))
        for name in ('max_memory_allocated', 'max_memory_reserved'):
            self.stack.enter_context(patch.object(evaluator.torch.cuda, name, return_value=0))
        self.stack.enter_context(patch.object(evaluator.torch.cuda, 'get_device_name', return_value='synthetic-CPU-not-GPU'))
        self.builder = self.stack.enter_context(patch.object(evaluator, 'build_model', return_value=(self.model, None, [], ['synthetic-conv1x1'])))
        self.stack.enter_context(patch.object(evaluator, 'normalize_logits', side_effect=lambda x: x))
        # Overlay cosmetics are not part of the receipt/control-flow acceptance.
        self.stack.enter_context(patch.object(evaluator, 'save_overlay'))

    def run_main(self):
        with redirect_stdout(io.StringIO()):
            return evaluator.main()

    def initial(self):
        code = self.run_main()
        self.assertEqual(code, 0, self.summary.read_text())
        self.assertTrue(receipt_path(self.prediction).is_file())
        result = json.loads(self.summary.read_text())
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['cases'][0]['provenance_verified'])
        self.assertEqual(result['gpu'], 'synthetic-CPU-not-GPU')
        return result

    def rejected_before_model_or_summary_write(self):
        before = self.summary.read_bytes()
        self.builder.reset_mock()
        with self.assertRaises(ValueError):
            self.run_main()
        self.builder.assert_not_called()
        self.assertEqual(self.summary.read_bytes(), before)

    def test_initial_and_valid_resume(self):
        first = self.initial()
        self.assertFalse(list((self.root / 'decoded_cache').glob('*.f32')))
        saved = self.prediction.read_bytes()
        with patch.object(evaluator, 'evaluate_case', side_effect=AssertionError('must not infer again')):
            self.assertEqual(self.run_main(), 0)
        second = json.loads(self.summary.read_text())
        self.assertEqual(first['aggregate'], second['aggregate'])
        self.assertEqual(self.prediction.read_bytes(), saved)
        self.assertTrue(second['cases'][0]['resumed_from_prediction'])

    def test_surface_limit_is_below_memory_intensive_final_volumes(self):
        self.assertLessEqual(47 * 512 * 512, evaluator.SURFACE_VOXEL_LIMIT)
        self.assertGreater(90 * 512 * 512, evaluator.SURFACE_VOXEL_LIMIT)

    def test_unsealed_legacy_refused(self):
        self.initial()
        receipt_path(self.prediction).unlink()  # only this test's own temporary fixture
        self.rejected_before_model_or_summary_write()

    def test_corrupt_prediction_refused(self):
        self.initial()
        self.prediction.write_bytes(b'truncated synthetic fixture')
        self.rejected_before_model_or_summary_write()

    def test_changed_image_refused(self):
        self.initial()
        nib.save(nib.Nifti1Image(np.ones((8, 9, 10)), np.eye(4)), self.image)
        self.rejected_before_model_or_summary_write()

    def test_changed_checkpoint_refused(self):
        self.initial()
        self.checkpoint.write_bytes(b'changed synthetic checkpoint')
        self.rejected_before_model_or_summary_write()

    def test_changed_protocol_refused(self):
        self.initial()
        config = json.loads(self.config.read_text())
        config['seed'] = 2
        self.config.write_text(json.dumps(config))
        self.rejected_before_model_or_summary_write()

    def test_interrupted_seal_is_not_silently_recovered(self):
        with patch.object(evaluator, 'seal', side_effect=RuntimeError('synthetic interruption')):
            self.assertEqual(self.run_main(), 1)
        self.assertTrue(self.prediction.is_file())
        self.assertFalse(receipt_path(self.prediction).exists())
        self.rejected_before_model_or_summary_write()


if __name__ == '__main__':
    unittest.main()
