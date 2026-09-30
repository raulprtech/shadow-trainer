"""Lightweight numerical checks; no patient data, torch or GPU."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dev_audit', Path(__file__).parents[1] / 'research/audit_development_evidence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DevelopmentAuditTests(unittest.TestCase):
    def item(self, p, t, i, dice, precision, recall):
        return dict(prediction_voxels=p, target_voxels=t, intersection_voxels=i,
                    dice=dice, precision=precision, recall=recall)

    def test_nonempty(self):
        self.assertEqual(audit.counts(self.item(10, 20, 5, 1 / 3, .5, .25))['recall'], .25)

    def test_both_empty_remains_missing_dice(self):
        self.assertIsNone(audit.counts(self.item(0, 0, 0, None, 1, 1))['dice'])

    def test_one_empty(self):
        self.assertEqual(audit.counts(self.item(0, 20, 0, 0, 0, 0))['dice'], 0)
        self.assertIsNone(audit.counts(self.item(20, 0, 0, 0, 0, None))['recall'])

    def test_invalid_counts(self):
        for p, t, i in ((-1, 2, 0), (1, 1, 2), (1.5, 2, 1)):
            with self.assertRaises(ValueError):
                audit.counts(self.item(p, t, i, 0, 0, 0))

    def test_wrong_or_nonfinite_metric(self):
        for dice in (0.9, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                audit.counts(self.item(10, 20, 5, dice, .5, .25))


if __name__ == '__main__':
    unittest.main()
