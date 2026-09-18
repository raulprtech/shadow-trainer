import json

import pytest

from shadow_trainer.config import JobConfig
from shadow_trainer.errors import ConfigurationError

from conftest import write_job


def test_job_contract_loads(tmp_path):
    config = JobConfig.load(write_job(tmp_path))
    assert config.schema_version == "shadowtrainer.job/v1"
    assert config.strategy == "auto"
    assert config.resources.cache_bytes == 64


def test_wrong_schema_is_rejected(tmp_path):
    path = write_job(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigurationError, match="schema_version"):
        JobConfig.load(path)


def test_unsafe_job_id_is_rejected(tmp_path):
    path = write_job(tmp_path)
    payload = json.loads(path.read_text())
    payload["job_id"] = "../escape"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigurationError, match="job_id"):
        JobConfig.load(path)


def test_v2_clinical_provenance_is_validated_and_preserved(tmp_path):
    path = write_job(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = "shadowtrainer.job/v2"
    payload["provenance"] = {
        "schema_version": "clinical-nigma.shadow-provenance/v1",
        "bridge_version": "2",
        "experiment_id": payload["job_id"],
        "experiment_digest": "a" * 64,
        "execution_split": "development",
        "split_ref": "split://renal/development/v1",
        "manifest_sha256": "b" * 64,
        "variant_id": "stunet-s@stage20",
    }
    path.write_text(json.dumps(payload))
    config = JobConfig.load(path)
    assert config.schema_version == "shadowtrainer.job/v2"
    assert config.public_dict()["provenance"] == payload["provenance"]


def test_v2_requires_sealed_provenance(tmp_path):
    path = write_job(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = "shadowtrainer.job/v2"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigurationError, match="provenance"):
        JobConfig.load(path)
