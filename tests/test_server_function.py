from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from train import server_function


class FakeTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], dict[str, str]] = {}
        self._counter = 0

    def _stamp(self) -> str:
        self._counter += 1
        return f"2026-05-07 12:00:{self._counter:02d}"

    def get_all(self, username: str) -> list[dict[str, str]]:
        tasks = [task.copy() for (task_username, _), task in self._tasks.items() if task_username == username]
        tasks.sort(key=lambda item: (item["createdAt"], item["taskName"]))
        return tasks

    def get(self, username: str, task_name: str) -> dict[str, str] | None:
        task = self._tasks.get((username, task_name))
        return task.copy() if task is not None else None

    def add(
        self,
        username: str,
        task_name: str,
        *,
        status: str = "",
        provider: str = "",
        remote_job_id: str = "",
        checkpoint_path: str = "",
        dataset_path: str = "",
        last_error: str = "",
    ) -> bool:
        key = (username, task_name)
        if key in self._tasks:
            return False
        timestamp = self._stamp()
        self._tasks[key] = {
            "taskName": task_name,
            "status": status,
            "provider": provider,
            "jobId": remote_job_id,
            "checkpointPath": checkpoint_path,
            "datasetPath": dataset_path,
            "error": last_error,
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }
        return True

    def update(self, username: str, task_name: str, **fields: str) -> bool:
        key = (username, task_name)
        if key not in self._tasks:
            return False
        task = self._tasks[key]
        mapping = {
            "status": "status",
            "provider": "provider",
            "remote_job_id": "jobId",
            "checkpoint_path": "checkpointPath",
            "dataset_path": "datasetPath",
            "last_error": "error",
        }
        for field_name, field_value in fields.items():
            task[mapping[field_name]] = str(field_value)
        task["updatedAt"] = self._stamp()
        return True

    def delete(self, username: str, task_name: str) -> bool:
        return self._tasks.pop((username, task_name), None) is not None


class ServerFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeTaskStore()
        self.patches = [
            patch.object(server_function, "sql_add_user_task", side_effect=self.store.add),
            patch.object(server_function, "sql_get_user_all_task", side_effect=self.store.get_all),
            patch.object(server_function, "sql_get_user_task", side_effect=self.store.get),
            patch.object(server_function, "sql_update_user_task", side_effect=self.store.update),
            patch.object(server_function, "sql_delete_user_task", side_effect=self.store.delete),
            patch.object(server_function.start_train, "load_env"),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()

    def test_start_training_submits_autodl_job_and_persists_metadata(self) -> None:
        platform = MagicMock()
        platform.submit.return_value = "autodl-job-1"
        request = json.dumps(
            {
                "username": "pearl",
                "taskName": "run-001",
                "action": "开始训练",
                "provider": "autodl",
                "datasetPath": "/mnt/data/demo",
                "epochs": 3,
                "checkpointPath": "/mnt/checkpoints/run-001",
                "checkpointFrequency": 1,
                "gpuCount": 1,
            },
            ensure_ascii=False,
        )

        with patch.object(server_function, "get_platform", return_value=platform) as mocked_factory:
            response = server_function.handle_request(request)

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(response["tasks"][0]["provider"], "autodl")
        self.assertEqual(response["tasks"][0]["jobId"], "autodl-job-1")
        mocked_factory.assert_called_once_with("autodl", region_id="cn-hangzhou")
        job_config = platform.submit.call_args.args[0]
        self.assertEqual(job_config["dataset_path"], "/mnt/data/demo")
        self.assertEqual(job_config["checkpoint_path"], "/mnt/checkpoints/run-001")

    def test_start_training_accepts_aliyun_alias(self) -> None:
        platform = MagicMock()
        platform.submit.return_value = "aliyun-job-1"

        with patch.object(server_function, "get_platform", return_value=platform) as mocked_factory:
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "run-002",
                        "action": "开始训练",
                        "provider": "aliyun_dlc",
                        "datasetPath": "/mnt/data/demo",
                        "epochs": 3,
                        "checkpointPath": "/mnt/checkpoints/run-002",
                        "checkpointFrequency": 1,
                        "gpuCount": 1,
                        "region": "cn-hangzhou",
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(response["tasks"][0]["provider"], "aliyun")
        mocked_factory.assert_called_once_with("aliyun", region_id="cn-hangzhou")

    def test_sync_refreshes_task_status_through_platform(self) -> None:
        self.store.add(
            "pearl",
            "run-003",
            status="Submitted",
            provider="autodl",
            remote_job_id="remote-003",
            checkpoint_path="/mnt/checkpoints/run-003",
            dataset_path="/mnt/data/demo",
        )
        platform = MagicMock()
        platform.metadata.return_value = {"status": "Running", "last_error": ""}

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps({"username": "pearl", "action": "任务同步"}, ensure_ascii=False)
            )

        self.assertEqual(response["message"], "sync success")
        self.assertEqual(response["tasks"][0]["status"], "Running")

    def test_sync_persists_platform_error_message(self) -> None:
        self.store.add(
            "pearl",
            "run-003b",
            status="Submitted",
            provider="autodl",
            remote_job_id="remote-003b",
            checkpoint_path="/mnt/checkpoints/run-003b",
            dataset_path="/mnt/data/demo",
        )
        platform = MagicMock()
        platform.metadata.return_value = {"status": "Failed", "last_error": "RuntimeError: boom"}

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps({"username": "pearl", "action": "任务同步"}, ensure_ascii=False)
            )

        self.assertEqual(response["tasks"][0]["status"], "Failed")
        self.assertEqual(response["tasks"][0]["error"], "RuntimeError: boom")

    def test_stop_training_calls_platform_and_marks_task_stopped(self) -> None:
        self.store.add(
            "pearl",
            "run-004",
            status="Running",
            provider="autodl",
            remote_job_id="remote-004",
        )
        platform = MagicMock()

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps({"username": "pearl", "taskName": "run-004", "action": "结束训练"}, ensure_ascii=False)
            )

        platform.stop.assert_called_once_with("remote-004")
        self.assertEqual(response["message"], "stop task success")
        self.assertEqual(response["tasks"][0]["status"], "STOPPED")

    def test_rejects_unknown_provider(self) -> None:
        with patch.object(server_function, "get_platform", side_effect=ValueError("Unknown training platform: foo")):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "run-005",
                        "action": "开始训练",
                        "provider": "foo",
                        "datasetPath": "/mnt/data/demo",
                        "epochs": 3,
                        "checkpointPath": "/mnt/checkpoints/run-005",
                        "checkpointFrequency": 1,
                        "gpuCount": 1,
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "unsupported provider: foo")
        self.assertEqual(response["tasks"], [])


if __name__ == "__main__":
    unittest.main()
