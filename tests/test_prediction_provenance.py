import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'research'))
from prediction_provenance import binding, seal, verify, receipt_path
from audit_development_masks import geometry, overlap_counts


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image, self.label, self.prediction = [self.root / name for name in ('image', 'label', 'prediction')]
        for path in (self.image, self.label, self.prediction):
            path.write_bytes(path.name.encode())
        self.expected = binding('a' * 64, self.image, self.label, {'version': 1})

    def test_valid_receipt(self):
        seal(self.prediction, self.expected)
        self.assertEqual(verify(self.prediction, self.expected)['binding'], self.expected)

    def test_legacy_refused_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'legacy'):
            verify(self.prediction, self.expected)
        self.assertEqual(self.prediction.read_bytes(), b'prediction')
        self.assertFalse(receipt_path(self.prediction).exists())

    def test_changed_bindings(self):
        seal(self.prediction, self.expected)
        for key in ('checkpoint_sha256', 'image_sha256', 'label_sha256', 'protocol'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance'):
                verify(self.prediction, {**self.expected, key: 'changed'})

    def test_modified_prediction(self):
        seal(self.prediction, self.expected)
        self.prediction.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            verify(self.prediction, self.expected)

    def test_no_receipt_overwrite(self):
        seal(self.prediction, self.expected)
        original = receipt_path(self.prediction).read_bytes()
        with self.assertRaises(ValueError):
            seal(self.prediction, self.expected)
        self.assertEqual(receipt_path(self.prediction).read_bytes(), original)

    def test_truncated_receipt(self):
        receipt_path(self.prediction).write_text('{')
        with self.assertRaises(json.JSONDecodeError):
            verify(self.prediction, self.expected)


class MaskAuditTests(unittest.TestCase):
    def image(self, array, affine=None):
        return nib.Nifti1Image(np.asarray(array, dtype=np.float32), np.eye(4) if affine is None else affine)

    def test_counts_against_dense_reference(self):
        rng = np.random.default_rng(7)
        p, t = [rng.integers(0, 4, size=(5, 7, 19)) for _ in range(2)]
        measured = overlap_counts(self.image(p), self.image(t), check=lambda: None)
        for label in (1, 2, 3):
            item = measured['classes'][str(label)]
            self.assertEqual(item['prediction_voxels'], int((p == label).sum()))
            self.assertEqual(item['target_voxels'], int((t == label).sum()))
            self.assertEqual(item['intersection_voxels'], int(((p == label) & (t == label)).sum()))
        self.assertEqual(measured['hec']['kidney_and_masses']['intersection_voxels'], int(((p > 0) & (t > 0)).sum()))

    def test_unknown_fractional_and_nan_labels(self):
        for value in (4, .5, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                overlap_counts(self.image([[[value]]]), self.image([[[0]]]), check=lambda: None)

    def test_geometry_and_shifted_affine(self):
        image = self.image(np.zeros((2, 3, 4)))
        geometry(image, image, image)
        shifted = np.eye(4)
        shifted[0, 3] = 10
        with self.assertRaisesRegex(ValueError, 'affine'):
            geometry(image, image, self.image(np.zeros((2, 3, 4)), shifted))

    def test_shape_mismatch(self):
        image = self.image(np.zeros((2, 3, 4)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            geometry(image, image, self.image(np.zeros((2, 3, 5))))
