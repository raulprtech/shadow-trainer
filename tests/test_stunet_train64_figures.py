from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "research"))
from render_stunet_train64_heldout import chosen_slice, reference_crop


class ReferenceSelectionTests(unittest.TestCase):
    def test_tumor_slice_wins_over_larger_renal_slice(self):
        target = np.zeros((300, 300, 3), dtype=np.uint8)
        target[10:100, 10:100, 0] = 1
        target[120:150, 120:150, 2] = 2
        self.assertEqual(chosen_slice(target), 2)

    def test_renal_fallback_and_bounded_crop(self):
        target = np.zeros((300, 300, 3), dtype=np.uint8)
        target[280:300, 280:300, 1] = 1
        self.assertEqual(chosen_slice(target), 1)
        x, y = reference_crop(target[:, :, 1])
        self.assertEqual((x.stop - x.start, y.stop - y.start), (256, 256))
        self.assertEqual((x.stop, y.stop), (300, 300))


if __name__ == "__main__":
    unittest.main()
