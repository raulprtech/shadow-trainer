import json
from pathlib import Path

import pytest

from shadow_trainer.errors import ConfigurationError, IntegrityError
from shadow_trainer.staging import CaseStager, LocalSource, Source, load_manifest

from conftest import write_case_source


def test_lru_never_exceeds_budget(tmp_path):
    source, manifest = write_case_source(tmp_path, count=3, size=16)
    stager = CaseStager(manifest, tmp_path / "cache", 20, LocalSource(source))
    stager.stage_batch(["case_0"])
    stager.stage_batch(["case_1"])
    assert stager.status()["occupancy_bytes"] <= 20
    assert stager.status()["cached_cases"] == ["case_1"]


def test_protected_batch_over_budget_is_rejected(tmp_path):
    source, manifest = write_case_source(tmp_path, count=2, size=16)
    stager = CaseStager(manifest, tmp_path / "cache", 20, LocalSource(source))
    with pytest.raises(IntegrityError, match="protected"):
        stager.stage_batch(["case_0", "case_1"])


def test_manifest_path_traversal_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "files": [
                            {
                                "name": "../outside",
                                "source": "safe",
                                "size_bytes": 1,
                            }
                        ],
                    }
                ]
            }
        )
    )
    with pytest.raises(ConfigurationError, match="inside"):
        load_manifest(manifest)


class WrongSizeSource(Source):
    def materialize(self, source: str, destination: Path) -> None:
        destination.write_bytes(b"x")


def test_failed_download_is_not_promoted(tmp_path):
    _, manifest = write_case_source(tmp_path, count=1, size=16)
    cache = tmp_path / "cache"
    stager = CaseStager(manifest, cache, 64, WrongSizeSource())
    with pytest.raises(IntegrityError, match="expected"):
        stager.stage_batch(["case_0"])
    assert not (cache / "case_0").exists()
    assert not (cache / ".case_0.partial").exists()
