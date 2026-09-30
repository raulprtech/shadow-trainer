import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "research"))
import complete_stunet_train64_heldout as campaign


class HeldoutCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name)
        self.cases = [f"case_{number:05d}" for number in range(6)]
        self.selection = {"evaluation_cases": self.cases,
                          "candidate_checkpoint_sha256": "digest"}
        self.config = {"base_checkpoint_sha256": "digest"}
        self.protocol = {"surface_voxel_limit": 0, "patch": 128, "stride": 64}
        for model in ("B0", "A"):
            folder = self.session / "evaluation/evaluation" / model
            (folder / "predictions").mkdir(parents=True)
            rows = []
            for case_id in self.cases:
                prediction = folder / "predictions" / f"{case_id}.nii.gz"
                prediction.write_bytes(b"fixture")
                score = 0.2 if model == "B0" else 0.3
                rows.append({
                    "case_id": case_id, "prediction": str(prediction),
                    "prediction_sha256": "digest", "provenance_verified": True,
                    "classes": {str(number): {"dice": score} for number in (1, 2, 3)},
                    "hec": {name: {"dice": score} for name in
                            ("kidney_and_masses", "kidney_mass", "tumor")},
                })
            summary = {"status": "success", "model": model, "cohort": "evaluation",
                       "cases_requested": self.cases, "checkpoint_sha256": "digest",
                       "protocol": self.protocol, "cases": rows}
            (folder / "summary.json").write_text(json.dumps(summary))

    def _audit(self):
        with (patch.object(campaign, "frozen_inputs",
                           return_value=(self.selection, {}, self.config)),
              patch.object(campaign, "sha256_file", return_value="digest")):
            return campaign.audit(self.session, self.session / "candidate.pt",
                                  verify_receipts=False)

    def test_complete_pair_has_six_ordered_deltas(self):
        result = self._audit()
        self.assertEqual(result["case_count"], 6)
        self.assertEqual(result["metrics"]["classes.tumor"]["improved_cases"], 6)
        self.assertAlmostEqual(result["metrics"]["classes.tumor"]["mean_paired_delta"], 0.1)

    def test_partial_candidate_cannot_publish_pair(self):
        path = self.session / "evaluation/evaluation/A/summary.json"
        summary = json.loads(path.read_text())
        summary["cases"] = summary["cases"][:1]
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._audit()

    def test_protocol_mismatch_cannot_publish_pair(self):
        path = self.session / "evaluation/evaluation/A/summary.json"
        summary = json.loads(path.read_text())
        summary["protocol"]["surface_voxel_limit"] = 20_000_000
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "protocol"):
            self._audit()


if __name__ == "__main__":
    unittest.main()
