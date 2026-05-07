from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
import unittest
from unittest.mock import patch

from train.platform.autodl import AutoDLPlatform


class AutoDLPlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = AutoDLPlatform(host="demo.autodl", port=2222, user="root", key_path="~/.ssh/id_ed25519")
        self.job_config = {
            "dataset_path": "/mnt/data/demo",
            "epochs": 3,
            "checkpoint_path": "/mnt/checkpoints/run-001",
            "checkpoint_frequency": 1,
            "gpu_count": 1,
        }

    def test_dry_run_reports_connection_and_workdir(self) -> None:
        with patch.dict(os.environ, {"AUTODL_WORKDIR": "/root/EVO_Train"}, clear=False):
            buffer = StringIO()
            with redirect_stdout(buffer):
                self.platform.submit({**self.job_config, "dry_run": True})
            payload = json.loads(buffer.getvalue())

        self.assertEqual(payload["host"], "demo.autodl")
        self.assertEqual(payload["port"], 2222)
        self.assertEqual(payload["user"], "root")
        self.assertEqual(payload["workdir"], "/root/EVO_Train")
        self.assertIn("python train.py", payload["command"])

    def test_submit_uses_pid_and_log_files(self) -> None:
        commands: list[str] = []

        def fake_exec(command: str) -> tuple[int, str, str]:
            commands.append(command)
            return 0, "job-123", ""

        with patch.object(self.platform, "_exec", side_effect=fake_exec):
            job_id = self.platform.submit({**self.job_config, "job_name": "run-001"})

        self.assertEqual(job_id, "job-123")
        remote_command = commands[0]
        self.assertIn("/tmp/evo_train_", remote_command)
        self.assertIn(".pid", remote_command)
        self.assertIn(".log", remote_command)
        self.assertIn(".exit", remote_command)
        self.assertIn(".stopped", remote_command)

    def test_submit_uses_workdir_from_job_config(self) -> None:
        commands: list[str] = []

        def fake_exec(command: str) -> tuple[int, str, str]:
            commands.append(command)
            return 0, "job-456", ""

        with patch.object(self.platform, "_exec", side_effect=fake_exec):
            self.platform.submit({**self.job_config, "workdir": "/root/project"})

        self.assertIn("cd /root/project", commands[0])

    def test_status_reads_pid_file(self) -> None:
        with patch.object(self.platform, "_exec", return_value=(0, "Running", "")) as mocked_exec:
            status = self.platform.status("job-789")

        self.assertEqual(status, "Running")
        self.assertIn(".pid", mocked_exec.call_args.args[0])
        self.assertIn(".exit", mocked_exec.call_args.args[0])
        self.assertIn(".stopped", mocked_exec.call_args.args[0])

    def test_status_can_report_success(self) -> None:
        with patch.object(self.platform, "_exec", return_value=(0, "Succeeded", "")):
            status = self.platform.status("job-790")

        self.assertEqual(status, "Succeeded")

    def test_status_can_report_failure(self) -> None:
        with patch.object(self.platform, "_exec", return_value=(0, "Failed", "")):
            status = self.platform.status("job-791")

        self.assertEqual(status, "Failed")

    def test_status_can_report_stopped(self) -> None:
        with patch.object(self.platform, "_exec", return_value=(0, "STOPPED", "")):
            status = self.platform.status("job-792")

        self.assertEqual(status, "STOPPED")

    def test_stop_removes_pid_file(self) -> None:
        with patch.object(self.platform, "_exec", return_value=(0, "", "")) as mocked_exec:
            self.platform.stop("job-999")

        command = mocked_exec.call_args.args[0]
        self.assertIn(".pid", command)
        self.assertIn("rm -f", command)
        self.assertIn(".stopped", command)


if __name__ == "__main__":
    unittest.main()
