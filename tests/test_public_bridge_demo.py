import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "export_bridge_demo", Path(__file__).parents[1] / "research/export_bridge_demo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicBridgeDemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "source").mkdir()
        self.source = self.root / "source/bridge.json"
        self.record = {
            "schema_version": "shadowtrainer.bridge-smoke/v1",
            "classification": "synthetic_cpu_integration_only",
            "status": "success", "training_steps_limit": 2, "cuda_visible_devices": "",
            "checks": list(module.CHECKS),
            "private": {"token": "SECRET_SENTINEL", "path": "/home/private/patient"},
        }
        self.source.write_text(json.dumps(self.record))

    def test_sanitized_and_hashed(self):
        original = self.source.read_bytes()
        output = self.root / "public"
        module.export(self.source, output)
        manifest = json.loads((output / "manifest.json").read_text())
        for name, entry in manifest["artifacts"].items():
            raw = (output / name).read_bytes()
            self.assertEqual(entry["bytes"], len(raw))
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertNotIn(b"SECRET_SENTINEL", raw)
            self.assertNotIn(b"/home/private", raw)
        self.assertEqual(original, self.source.read_bytes())

    def test_no_overwrite(self):
        output = self.root / "public"
        module.export(self.source, output)
        with self.assertRaises(FileExistsError):
            module.export(self.source, output)

    def test_no_output_inside_source(self):
        with self.assertRaises(ValueError):
            module.export(self.source, self.source.parent / "output")

    def test_unknown_status_and_untrusted_check(self):
        for key, value in (("status", "unknown"), ("checks", ["<script>SECRET</script>"]),
                           ("training_steps_limit", True)):
            data = dict(self.record)
            data[key] = value
            self.source.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                module.export(self.source, self.root / "public")
            self.assertFalse((self.root / "public").exists())

    def test_nonfinite_and_duplicate_rejected(self):
        for raw in ('{"bad":NaN}', '{"status":"success","status":"unknown"}'):
            self.source.write_text(raw)
            with self.assertRaises(ValueError):
                module.export(self.source, self.root / "public")

    def test_symlink_and_oversize_rejected(self):
        link = self.root / "linked.json"
        link.symlink_to(self.source)
        with self.assertRaises(ValueError):
            module.export(link, self.root / "public")
        self.source.write_text(" " * 65537)
        with self.assertRaises(ValueError):
            module.export(self.source, self.root / "public")


if __name__ == "__main__":
    unittest.main()
