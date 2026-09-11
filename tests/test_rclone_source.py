import subprocess
from unittest.mock import patch

import pytest

from shadow_trainer.errors import IntegrityError
from shadow_trainer.staging import RcloneSource


@patch("shadow_trainer.staging.Path.is_file", return_value=True)
@patch("shadow_trainer.staging.os.access", return_value=True)
@patch("shadow_trainer.staging.shutil.which", return_value="/fake/rclone")
def test_rclone_success_invokes_bounded_command(_which, _access, _is_file, tmp_path):
    source = RcloneSource("remote:dataset")

    def success(command, **kwargs):
        (tmp_path / "output").write_bytes(b"ok")
        return subprocess.CompletedProcess(command, 0)

    with patch("shadow_trainer.staging.subprocess.run", side_effect=success) as run:
        source.materialize("case/file.bin", tmp_path / "output")
    command = run.call_args.args[0]
    assert command[0] == "/fake/rclone"
    assert "remote:dataset/case/file.bin" in command
    assert "--retries" in command


@patch("shadow_trainer.staging.Path.is_file", return_value=True)
@patch("shadow_trainer.staging.os.access", return_value=True)
@patch("shadow_trainer.staging.shutil.which", return_value="/fake/rclone")
def test_rclone_timeout_is_a_stable_integrity_error(_which, _access, _is_file, tmp_path):
    source = RcloneSource("remote:dataset")
    with patch(
        "shadow_trainer.staging.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["rclone"], 600),
    ):
        with pytest.raises(IntegrityError, match="TimeoutExpired"):
            source.materialize("case/file.bin", tmp_path / "output")
