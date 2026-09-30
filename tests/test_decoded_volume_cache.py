import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'research'))
import decoded_volume_cache as cache


class DecodedVolumeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'image.nii.gz'
        self.data = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
        nib.save(nib.Nifti1Image(self.data, np.eye(4)), self.path)
        self.source_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.image = nib.load(self.path)
        self.folder = self.root / 'decoded'

    def build(self):
        with patch.object(cache, 'validate_resources'):
            return cache.materialize(self.image, self.source_sha, self.folder)

    def test_decode_and_reuse_exact(self):
        first = self.build()
        self.assertTrue(np.array_equal(first, self.data))
        self.assertTrue(first.flags.f_contiguous)
        second = self.build()
        self.assertTrue(np.array_equal(second, self.data))
        self.assertTrue(second.flags.f_contiguous)
        self.assertEqual(len(list(self.folder.glob('*.f32'))), 1)

    def test_scaled_compressed_nifti_preserves_proxy_values(self):
        scaled_path = self.root / 'scaled.nii.gz'
        image = nib.Nifti1Image(np.arange(7 * 9 * 11, dtype=np.int16).reshape(7, 9, 11), np.eye(4))
        image.header.set_slope_inter(0.5, -100.0)
        nib.save(image, scaled_path)
        source_sha = hashlib.sha256(scaled_path.read_bytes()).hexdigest()
        loaded = nib.load(scaled_path)
        with patch.object(cache, 'validate_resources'):
            decoded = cache.materialize(loaded, source_sha, self.folder)
        self.assertTrue(np.array_equal(decoded, np.asarray(loaded.dataobj, dtype=np.float32)))

    def test_tampering_rejected(self):
        self.build()
        target = self.folder / f'{self.source_sha}.f32'
        with target.open('r+b') as stream:
            stream.write(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.build()

    def test_missing_receipt_rejected(self):
        self.build()
        (self.folder / f'{self.source_sha}.receipt.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.build()

    def test_changed_source_digest_uses_distinct_cache(self):
        self.build()
        changed = 'f' * 64
        with patch.object(cache, 'validate_resources'):
            cache.materialize(self.image, changed, self.folder)
        self.assertEqual(len(list(self.folder.glob('*.f32'))), 2)

    def test_resource_rejection_before_decode(self):
        with patch.object(cache, 'validate_resources', side_effect=RuntimeError('disk')):
            with self.assertRaisesRegex(RuntimeError, 'disk'):
                cache.materialize(self.image, self.source_sha, self.folder)
        self.assertFalse(list(self.folder.glob('*.f32')))

    def test_verified_cache_can_be_evicted_and_rebuilt(self):
        first = self.build()
        del first
        self.assertTrue(cache.evict(self.folder, self.source_sha))
        self.assertFalse(list(self.folder.iterdir()))
        self.assertTrue(np.array_equal(self.build(), self.data))

    def test_modified_cache_is_not_silently_evicted(self):
        self.build()
        target = self.folder / f'{self.source_sha}.f32'
        with target.open('r+b') as stream:
            stream.write(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            cache.evict(self.folder, self.source_sha)
        self.assertTrue(target.exists())
