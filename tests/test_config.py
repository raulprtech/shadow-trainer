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
