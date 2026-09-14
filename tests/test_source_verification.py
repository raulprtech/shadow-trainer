import pytest

from shadow_trainer.errors import IntegrityError
from shadow_trainer.source_verification import verify_local_manifest

from conftest import write_case_source


def test_verify_local_manifest_and_detect_changed_source(tmp_path):
    source, manifest = write_case_source(tmp_path, count=2, size=16)
    result = verify_local_manifest(manifest, source)
    assert result["status"] == "verified"
    assert result["case_count"] == 2
    assert result["file_count"] == 2
    assert result["total_bytes"] == 32

    (source / "case_1.bin").write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="expected"):
        verify_local_manifest(manifest, source)
