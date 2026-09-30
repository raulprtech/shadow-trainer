import importlib.util
from pathlib import Path


def test_phase_logs_cannot_overwrite_bridge_artifacts(tmp_path):
    source = Path(__file__).resolve().parents[1] / "research" / "run_bridge_smoke.py"
    spec = importlib.util.spec_from_file_location("bridge_smoke", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = tmp_path / "receipt.json"
    module.save(artifact, {"run_status": "success"})
    module.save(module.phase_log_path(tmp_path, "receipt"), {"returncode": 0})
    assert module.load(artifact) == {"run_status": "success"}
    for name in ("build", "plan", "execute", "report", "receipt"):
        assert module.phase_log_path(tmp_path, name) != tmp_path / (name + ".json")
