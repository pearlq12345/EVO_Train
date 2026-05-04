from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sql_lite import sql_pack
from train import server_function


def _job_body(*, job_id: str = "dlc-job-1", status: str = "Running", reason_message: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id,
        display_name="demo-job",
        status=status,
        sub_status="",
        reason_code="",
        reason_message=reason_message,
        workspace_id="ws-1",
        resource_id="rg-1",
        user_command="python train.py",
    )


class ServerFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        sql_pack.DB_PATH = Path(self.temp_dir.name) / "tasks.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_start_training_submits_aliyun_job_and_persists_metadata(self) -> None:
        request = json.dumps(
            {
                "username": "pearl",
                "taskName": "run-001",
                "action": "开始训练",
                "provider": "aliyun",
                "datasetPath": "/mnt/data/demo",
                "epochs": 3,
                "checkpointPath": "/mnt/checkpoints/run-001",
                "checkpointFrequency": 1,
                "gpuCount": 1,
            },
            ensure_ascii=False,
        )

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(server_function.start_train, "create_job", return_value=_job_body(job_id="job-123")),
        ):
            response = server_function.handle_request(request)

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(len(response["tasks"]), 1)
        task = response["tasks"][0]
        self.assertEqual(task["taskName"], "run-001")
        self.assertEqual(task["provider"], "aliyun")
        self.assertEqual(task["jobId"], "job-123")
        self.assertEqual(task["checkpointPath"], "/mnt/checkpoints/run-001")
        self.assertEqual(task["datasetPath"], "/mnt/data/demo")
        self.assertEqual(task["status"], "Submitted")

    def test_sync_refreshes_remote_status(self) -> None:
        created = sql_pack.sql_add_user_task(
            "pearl",
            "run-002",
            status="Submitted",
            provider="aliyun",
            remote_job_id="job-234",
            checkpoint_path="/mnt/checkpoints/run-002",
            dataset_path="/mnt/data/demo",
        )
        self.assertTrue(created)

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(server_function.start_train, "fetch_job_body", return_value=_job_body(job_id="job-234", status="Running")),
        ):
            response = server_function.handle_request(
                json.dumps({"username": "pearl", "action": "任务同步"}, ensure_ascii=False)
            )

        self.assertEqual(response["message"], "sync success")
        self.assertEqual(response["tasks"][0]["status"], "Running")
        self.assertEqual(response["tasks"][0]["jobId"], "job-234")

    def test_stop_training_marks_remote_task_stopped(self) -> None:
        created = sql_pack.sql_add_user_task(
            "pearl",
            "run-003",
            status="Running",
            provider="aliyun",
            remote_job_id="job-345",
        )
        self.assertTrue(created)

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(server_function.start_train, "stop_job"),
        ):
            response = server_function.handle_request(
                json.dumps(
                    {"username": "pearl", "taskName": "run-003", "action": "结束训练"},
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "stop task success")
        self.assertEqual(response["tasks"][0]["status"], "STOPPED")

    def test_delete_task_removes_row(self) -> None:
        self.assertTrue(sql_pack.sql_add_user_task("pearl", "run-004"))

        response = server_function.handle_request(
            json.dumps({"username": "pearl", "taskName": "run-004", "action": "删除任务"}, ensure_ascii=False)
        )

        self.assertEqual(response["message"], "delete task success")
        self.assertEqual(response["tasks"], [])

    def test_rejects_unsupported_provider(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "taskName": "run-005",
                    "action": "开始训练",
                    "provider": "autodl",
                    "datasetPath": "/mnt/data/demo",
                    "epochs": 3,
                    "checkpointPath": "/mnt/checkpoints/run-005",
                    "checkpointFrequency": 1,
                    "gpuCount": 1,
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "unsupported provider: autodl")
        self.assertEqual(response["tasks"], [])


if __name__ == "__main__":
    unittest.main()
