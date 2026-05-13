from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sql_lite import sql_pack
from thread_pool.thread_pool import ThreadPool, TrainTaskEvent
from train import server_function
from train.platform.autodl import AutoDLApiClient, AutoDLPlatform


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
        sql_pack.sql_set_user_balance("pearl", 2000)
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
        self.assertEqual(task["hourlyPriceCents"], "1000")
        self.assertEqual(task["billingStatus"], "frozen")
        self.assertEqual(sql_pack.sql_get_wallet("pearl")["frozenCents"], "1000")
        self.assertEqual(response["wallet"]["balanceCents"], "2000")
        self.assertEqual(response["wallet"]["frozenCents"], "1000")

    def test_wallet_and_billing_actions_expose_user_balance_and_records(self) -> None:
        recharge = server_function.handle_request(
            json.dumps(
                {"username": "pearl", "action": "管理员充值", "balanceCents": 2500},
                ensure_ascii=False,
            )
        )

        self.assertEqual(recharge["message"], "set balance success")
        self.assertEqual(recharge["wallet"]["balanceCents"], "2500")
        self.assertEqual(recharge["wallet"]["availableCents"], "2500")
        self.assertEqual(recharge["billingRecords"][0]["kind"], "admin_set_balance")

        wallet = server_function.handle_request(
            json.dumps({"username": "pearl", "action": "余额查询"}, ensure_ascii=False)
        )
        self.assertEqual(wallet["message"], "wallet query success")
        self.assertEqual(wallet["wallet"]["balanceCents"], "2500")

        records = server_function.handle_request(
            json.dumps({"username": "pearl", "action": "账单查询"}, ensure_ascii=False)
        )
        self.assertEqual(records["message"], "billing records query success")
        self.assertEqual(records["billingRecords"][0]["amountCents"], "2500")

    def test_set_price_action_changes_training_freeze_amount(self) -> None:
        price_set = server_function.handle_request(
            json.dumps(
                {"action": "价格设置", "provider": "aliyun", "gpuSpec": "ecs.gn7i-c8g1.2xlarge", "hourlyPriceCents": 1800},
                ensure_ascii=False,
            )
        )
        self.assertEqual(price_set["message"], "set gpu price success")
        price_query = server_function.handle_request(
            json.dumps({"action": "价格查询", "provider": "aliyun"}, ensure_ascii=False)
        )
        self.assertEqual(price_query["message"], "price query success")
        self.assertEqual(price_query["prices"][0]["hourlyPriceCents"], "1800")
        sql_pack.sql_set_user_balance("pearl", 2000)

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(server_function.start_train, "create_job", return_value=_job_body(job_id="job-price")),
        ):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "run-price",
                        "action": "开始训练",
                        "provider": "aliyun",
                        "datasetPath": "/mnt/data/demo",
                        "epochs": 3,
                        "checkpointPath": "/mnt/checkpoints/run-price",
                        "checkpointFrequency": 1,
                        "gpuCount": 1,
                        "gpuSpec": "ecs.gn7i-c8g1.2xlarge",
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(response["tasks"][0]["hourlyPriceCents"], "1800")
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "2000")
        self.assertEqual(wallet["frozenCents"], "1800")

    def test_sync_refreshes_remote_status(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "run-002", 1000, "test setup"))
        created = sql_pack.sql_add_user_task(
            "pearl",
            "run-002",
            status="Submitted",
            provider="aliyun",
            remote_job_id="job-234",
            checkpoint_path="/mnt/checkpoints/run-002",
            dataset_path="/mnt/data/demo",
            hourly_price_cents=1000,
            frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), 1),
            started_at=sql_pack.utc_now_text(),
            billing_status="frozen",
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
        self.assertIn("datasetDir", response)
        self.assertEqual(response["tasks"][0]["status"], "Running")
        self.assertEqual(response["tasks"][0]["jobId"], "job-234")

    def test_query_status_refreshes_single_task_for_upstream_compatibility(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "run-status", 1000, "test setup"))
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "run-status",
                status="Submitted",
                provider="aliyun",
                remote_job_id="job-status",
                hourly_price_cents=1000,
                frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), 1),
                started_at=sql_pack.utc_now_text(),
                billing_status="frozen",
            )
        )

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(server_function.start_train, "fetch_job_body", return_value=_job_body(job_id="job-status", status="Running")),
        ):
            response = server_function.handle_request(
                json.dumps(
                    {"username": "pearl", "taskName": "run-status", "action": "查询状态"},
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "run-status: Running")
        self.assertEqual(response["tasks"][0]["status"], "Running")

    def test_query_download_directory_and_logs_match_upstream_actions(self) -> None:
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "run-files",
                status="Failed",
                provider="autodl",
                remote_job_id="instance-1::runner-1",
                checkpoint_path="/root/autodl-tmp/evo_train/output/run-files",
                last_error="line one\nline two",
            )
        )

        directory = server_function.handle_request(
            json.dumps(
                {"username": "pearl", "taskName": "run-files", "action": "查询下载目录"},
                ensure_ascii=False,
            )
        )
        logs = server_function.handle_request(
            json.dumps(
                {"username": "pearl", "taskName": "run-files", "action": "请求用户日志"},
                ensure_ascii=False,
            )
        )

        self.assertEqual(directory["message"], "query download directory success")
        self.assertEqual(directory["downloadPath"], "/root/autodl-tmp/evo_train/output/run-files")
        self.assertEqual(directory["checkpoints"], ["/root/autodl-tmp/evo_train/output/run-files"])
        self.assertEqual(logs["message"], "query user logs success")
        self.assertEqual(logs["logs"], ["line one", "line two"])

    def test_stop_training_marks_remote_task_stopped(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "run-003", 1000, "test setup"))
        created = sql_pack.sql_add_user_task(
            "pearl",
            "run-003",
            status="Running",
            provider="aliyun",
            remote_job_id="job-345",
            hourly_price_cents=1000,
            frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), 1),
            started_at=sql_pack.utc_now_text(),
            billing_status="frozen",
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
        self.assertEqual(response["tasks"][0]["billingStatus"], "settled")
        self.assertEqual(response["tasks"][0]["actualCostCents"], "1000")
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "1000")
        self.assertEqual(wallet["frozenCents"], "0")

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

    def test_start_training_supports_autodl_with_same_billing_gate(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 3000)
        platform = SimpleNamespace(submit=lambda job_config: "autodl-job-1")

        with (
            patch.dict("os.environ", {"EVO_TRAIN_ALLOW_RAW_COMMAND": "true"}, clear=False),
            patch.object(server_function, "get_platform", return_value=platform) as mocked_factory,
        ):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "autodl-run",
                        "action": "开始训练",
                        "provider": "autodl",
                        "command": "python eval.py --suite libero_object_task",
                        "workdir": "/root/autodl-tmp/evf",
                        "hourlyPriceCents": 1200,
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "create task success")
        mocked_factory.assert_called_once_with("autodl", region_id="cn-hangzhou")
        task = response["tasks"][0]
        self.assertEqual(task["provider"], "autodl")
        self.assertEqual(task["jobId"], "autodl-job-1")
        self.assertEqual(task["hourlyPriceCents"], "1200")
        self.assertEqual(response["wallet"]["frozenCents"], "1200")

    def test_raw_command_is_disabled_unless_explicitly_allowed(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 3000)

        with patch.dict("os.environ", {}, clear=True):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "raw-command",
                        "action": "开始训练",
                        "provider": "autodl",
                        "command": "python train.py",
                        "workdir": "/root/autodl-tmp/evf",
                        "hourlyPriceCents": 1200,
                    },
                    ensure_ascii=False,
                )
            )

        self.assertIn("raw command is disabled", response["message"])
        self.assertEqual(sql_pack.sql_get_wallet("pearl")["frozenCents"], "0")

    def test_ai_plan_generates_workflow_recipe_for_roboclaw(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "message": "我想在metaworld上跑pick-place，20个epoch，训练后评估10个episode",
                    "provider": "autodl",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(response["plan"]["workflow"], "evf_metaworld")
        self.assertEqual(response["plan"]["params"]["envName"], "pick-place-v2")
        self.assertEqual(response["plan"]["params"]["epochs"], 20)
        self.assertEqual(
            [stage["name"] for stage in response["plan"]["stages"]],
            ["prepare_data", "train", "evaluate", "collect_artifacts"],
        )
        self.assertIn("--benchmark metaworld", response["plan"]["stages"][1]["command"])
        self.assertIn("__EVO_STAGE_START__", response["plan"]["command"])
        self.assertTrue(response["plan"]["needsConfirmation"])
        self.assertEqual(response["plan"]["missingFields"], [])
        self.assertEqual(response["plan"]["estimatedHours"], "1")
        self.assertEqual(response["plan"]["estimatedMinimumCostCents"], "1000")
        self.assertTrue(response["plan"]["readyToStart"])

    def test_ai_plan_reports_missing_fields_and_warnings(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "evf_metaworld",
                    "params": {"epochs": 120, "batchSize": 256, "learningRate": 0.02},
                    "provider": "autodl",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertIn("envName", response["plan"]["missingFields"])
        self.assertFalse(response["plan"]["readyToStart"])
        self.assertEqual(response["plan"]["estimatedHours"], "3")
        self.assertGreaterEqual(len(response["plan"]["warnings"]), 3)

    def test_start_training_rejects_workflow_with_missing_fields(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 3000)

        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "taskName": "workflow-missing",
                    "action": "开始训练",
                    "provider": "autodl",
                    "workflow": "evf_metaworld",
                    "params": {"epochs": 5},
                },
                ensure_ascii=False,
            )
        )

        self.assertIn("missing workflow fields: envName", response["message"])
        self.assertEqual(sql_pack.sql_get_wallet("pearl")["frozenCents"], "0")

    def test_start_training_materializes_workflow_before_provider_submit(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 3000)
        submitted_configs: list[dict[str, object]] = []

        def submit(job_config: dict[str, object]) -> str:
            submitted_configs.append(job_config)
            return "workflow-job-1"

        platform = SimpleNamespace(submit=submit)

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "workflow-run",
                        "action": "开始训练",
                        "provider": "autodl",
                        "workflow": "evf_libero",
                        "params": {
                            "suite": "libero_object_task",
                            "taskId": 2,
                            "epochs": 5,
                            "evalEpisodes": 3,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(response["tasks"][0]["provider"], "autodl")
        self.assertEqual(response["tasks"][0]["datasetPath"], "/root/autodl-tmp/datasets/libero/libero_object_task")
        self.assertIn("__EVO_STAGE_START__", submitted_configs[0]["command"])
        self.assertIn("--benchmark libero", submitted_configs[0]["command"])
        self.assertIn("--task-id 2", submitted_configs[0]["command"])
        self.assertTrue(submitted_configs[0]["autodl_managed"])

    def test_custom_project_plan_keeps_general_training_simple(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "custom_project",
                    "params": {
                        "repoUrl": "https://github.com/example/project.git",
                        "trainCommand": "python train.py --config configs/demo.yaml",
                        "evalCommand": "python eval.py --ckpt outputs/latest.pt",
                        "artifactPath": "/root/autodl-tmp/custom_project/outputs",
                    },
                    "provider": "autodl",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(response["plan"]["workflow"], "custom_project")
        self.assertEqual(response["plan"]["missingFields"], [])
        self.assertEqual(
            [stage["name"] for stage in response["plan"]["stages"]],
            ["prepare_code", "setup_env", "prepare_data", "train", "evaluate", "collect_artifacts"],
        )
        self.assertIn("git clone", response["plan"]["stages"][0]["command"])
        self.assertIn("python train.py", response["plan"]["stages"][3]["command"])

    def test_custom_project_requires_repo_and_train_command(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "custom_project",
                    "params": {},
                    "provider": "autodl",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(response["plan"]["missingFields"], ["repoUrl", "trainCommand"])
        self.assertFalse(response["plan"]["readyToStart"])

    def test_billing_scan_stops_autodl_task_when_next_hour_cannot_be_frozen(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 1000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "autodl-run-2", 1000, "test setup"))
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "autodl-run-2",
                status="Running",
                provider="autodl",
                remote_job_id="autodl-job-2",
                hourly_price_cents=1000,
                frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), -1),
                started_at=sql_pack.add_hours_text(sql_pack.utc_now_text(), -1),
                billing_status="frozen",
            )
        )
        platform = SimpleNamespace(
            metadata=lambda job_id: {"status": "Running", "last_error": ""},
            stop=lambda job_id: None,
        )

        with patch.object(server_function, "get_platform", return_value=platform) as mocked_factory:
            stats = server_function.scan_billing_tasks()

        self.assertEqual(stats["checked"], 1)
        self.assertGreaterEqual(mocked_factory.call_count, 1)
        task = sql_pack.sql_get_user_task("pearl", "autodl-run-2")
        self.assertEqual(task["status"], "STOPPED")
        self.assertEqual(task["billingStatus"], "settled")
        self.assertEqual(sql_pack.sql_get_wallet("pearl")["balanceCents"], "0")

    def test_autodl_managed_platform_powers_on_instance_and_returns_composite_job_id(self) -> None:
        class FakeApi:
            def __init__(self) -> None:
                self.powered_on: list[str] = []

            def wallet_balance(self) -> dict[str, str]:
                return {"assets": "5000", "accumulate": "0", "voucherBalance": "0"}

            def status(self, instance_uuid: str) -> str:
                return "running"

            def power_on(self, instance_uuid: str, start_command: str | None = None) -> None:
                self.powered_on.append(instance_uuid)

        platform = AutoDLPlatform(host="demo.autodl", port=22, user="root", key_path="/tmp/key", api_client=FakeApi())
        with patch.dict("os.environ", {"AUTODL_TOKEN": "token"}, clear=False):
            with patch.object(platform, "_exec", return_value=(0, "runner-1", "")):
                job_id = platform.submit(
                    {
                        "command": "python eval.py",
                        "workdir": "/root/autodl-tmp/evf",
                        "autodl_instance_uuid": "pro-1",
                    }
                )

        self.assertEqual(job_id, "pro-1::runner-1")

    def test_autodl_managed_platform_can_use_snapshot_ssh_connection(self) -> None:
        class FakeApi:
            def snapshot(self, instance_uuid: str) -> dict[str, object]:
                return {
                    "proxy_host": "connect.autodl.example",
                    "ssh_port": 34222,
                    "root_password": "secret",
                }

        platform = AutoDLPlatform(api_client=FakeApi())

        with patch.dict("os.environ", {}, clear=True):
            host, port, user, key_path, password = platform._connection_settings("pro-1")

        self.assertEqual(host, "connect.autodl.example")
        self.assertEqual(port, 34222)
        self.assertEqual(user, "root")
        self.assertIsNone(key_path)
        self.assertEqual(password, "secret")

    def test_platform_balance_query_reports_low_autodl_balance(self) -> None:
        class FakeClient:
            def wallet_balance(self) -> dict[str, str]:
                return {"assets": "500", "accumulate": "100", "voucherBalance": "0"}

        with patch.object(server_function, "AutoDLApiClient", return_value=FakeClient()):
            response = server_function.handle_request(
                json.dumps(
                    {"action": "平台余额查询", "provider": "autodl", "minimumAssets": 1000},
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "platform balance query success")
        self.assertEqual(response["balance"]["assets"], "500")
        self.assertTrue(response["lowBalance"])

    def test_platform_balance_query_returns_error_json_when_token_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            response = server_function.handle_request(
                json.dumps(
                    {"action": "平台余额查询", "provider": "autodl", "minimumAssets": 1000},
                    ensure_ascii=False,
                )
            )

        self.assertIn("platform balance query failed", response["message"])

    def test_gpu_sku_query_applies_ten_percent_service_fee(self) -> None:
        skus = json.dumps(
            [
                {
                    "skuId": "sku-4090",
                    "provider": "autodl",
                    "displayName": "RTX 4090 48G",
                    "gpuSpec": "4090-48g",
                    "autodlGpuSpecUuid": "v-48g",
                    "autodlImageUuid": "image-1",
                    "cudaVFrom": 118,
                    "costHourlyCents": 901,
                }
            ]
        )

        with patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus, "EVO_TRAIN_SERVICE_FEE_RATE": "0.10"}, clear=False):
            response = server_function.handle_request(
                json.dumps({"action": "GPU规格查询", "provider": "autodl"}, ensure_ascii=False)
            )

        self.assertEqual(response["message"], "gpu sku query success")
        self.assertEqual(response["skus"][0]["skuId"], "sku-4090")
        self.assertEqual(response["skus"][0]["hourlyPriceCents"], "992")
        self.assertEqual(response["skus"][0]["serviceFeeRate"], "0.10")
        self.assertTrue(response["skus"][0]["readyToStart"])

    def test_gpu_sku_query_hides_incomplete_default_skus(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            response = server_function.handle_request(
                json.dumps({"action": "GPU规格查询", "provider": "autodl"}, ensure_ascii=False)
            )

        self.assertEqual(response["message"], "gpu sku query success")
        self.assertEqual(response["skus"], [])

        debug_response = server_function.handle_request(
            json.dumps({"action": "GPU规格查询", "provider": "autodl", "includeIncomplete": True}, ensure_ascii=False)
        )
        self.assertGreater(len(debug_response["skus"]), 0)
        self.assertFalse(debug_response["skus"][0]["readyToStart"])

    def test_autodl_start_training_maps_sku_to_real_create_fields_and_price(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        skus = json.dumps(
            [
                {
                    "skuId": "sku-4090",
                    "provider": "autodl",
                    "displayName": "RTX 4090 48G",
                    "gpuSpec": "4090-48g",
                    "autodlGpuSpecUuid": "v-48g",
                    "autodlImageUuid": "image-1",
                    "cudaVFrom": 118,
                    "gpuCount": 1,
                    "costHourlyCents": 901,
                }
            ]
        )
        captured: dict[str, object] = {}

        class FakePlatform:
            def submit(self, job_config: dict[str, object]) -> str:
                captured.update(job_config)
                return "pro-1::runner-1"

        request = {
            "username": "pearl",
            "taskName": "autodl-sku",
            "action": "开始训练",
            "provider": "autodl",
            "skuId": "sku-4090",
            "workflow": "custom_project",
            "params": {
                "repoUrl": "https://example.com/repo.git",
                "setupCommand": "true",
                "trainCommand": "python train.py",
                "evalCommand": "python eval.py",
                "artifactCommand": "true",
                "workdir": "/root/autodl-tmp/project",
            },
        }

        with (
            patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus, "EVO_TRAIN_SERVICE_FEE_RATE": "0.10"}, clear=False),
            patch.object(server_function, "get_platform", return_value=FakePlatform()),
        ):
            response = server_function.handle_request(json.dumps(request, ensure_ascii=False))

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(captured["autodl_gpu_spec_uuid"], "v-48g")
        self.assertEqual(captured["autodl_image_uuid"], "image-1")
        self.assertEqual(captured["autodl_cuda_v_from"], 118)
        self.assertEqual(captured["gpu_count"], 1)
        self.assertEqual(response["tasks"][0]["hourlyPriceCents"], "992")

    def test_autodl_start_training_rejects_incomplete_sku_before_billing(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        skus = json.dumps(
            [
                {
                    "skuId": "sku-incomplete",
                    "provider": "autodl",
                    "displayName": "RTX 4090 48G",
                    "gpuSpec": "4090-48g",
                    "autodlGpuSpecUuid": "v-48g",
                    "costHourlyCents": 900,
                }
            ]
        )
        request = {
            "username": "pearl",
            "taskName": "autodl-bad-sku",
            "action": "开始训练",
            "provider": "autodl",
            "skuId": "sku-incomplete",
            "workflow": "custom_project",
            "params": {
                "repoUrl": "https://example.com/repo.git",
                "trainCommand": "python train.py",
            },
        }

        with patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus}, clear=False):
            response = server_function.handle_request(json.dumps(request, ensure_ascii=False))

        self.assertIn("AutoDL sku is missing required fields", response["message"])
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "2000")
        self.assertEqual(wallet["frozenCents"], "0")

    def test_autodl_api_create_instance_accepts_string_data_and_sends_sku_fields(self) -> None:
        requests: list[dict[str, object]] = []

        class FakeClient(AutoDLApiClient):
            def __init__(self) -> None:
                pass

            def _request(self, method: str, path: str, body: dict[str, object] | None = None) -> dict[str, object]:
                requests.append({"method": method, "path": path, "body": body or {}})
                return {"code": "Success", "data": "pro-created"}

        instance_uuid = FakeClient().create_instance(
            {
                "gpu_count": 1,
                "autodl_gpu_spec_uuid": "v-48g",
                "autodl_image_uuid": "image-1",
                "autodl_cuda_v_from": 118,
                "autodl_data_centers": "westDC3,beijingDC2",
                "job_name": "EVO task",
                "autodl_start_command": "sleep 1",
            }
        )

        self.assertEqual(instance_uuid, "pro-created")
        body = requests[0]["body"]
        self.assertEqual(body["gpu_spec_uuid"], "v-48g")
        self.assertEqual(body["image_uuid"], "image-1")
        self.assertEqual(body["cuda_v_from"], 118)
        self.assertEqual(body["data_center_list"], ["westDC3", "beijingDC2"])
        self.assertEqual(body["instance_name"], "EVO task")
        self.assertEqual(body["start_command"], "sleep 1")

    def test_thread_pool_returns_json_when_handler_crashes(self) -> None:
        responses: list[str] = []
        pool = ThreadPool(4, task_handler=lambda request_text: (_ for _ in ()).throw(SystemExit("boom")))
        pool._handle_event(
            0,
            TrainTaskEvent(
                client_id="client-1",
                request_text="{}",
                response_callback=responses.append,
            ),
        )

        self.assertEqual(json.loads(responses[0])["message"], "internal error: boom")

    def test_admin_actions_require_admin_token_when_configured(self) -> None:
        with patch.dict("os.environ", {"EVO_TRAIN_ADMIN_TOKEN": "admin-secret"}, clear=False):
            rejected = server_function.handle_request(
                json.dumps(
                    {"username": "pearl", "action": "管理员充值", "balanceCents": 2500},
                    ensure_ascii=False,
                )
            )
            accepted = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "action": "管理员充值",
                        "balanceCents": 2500,
                        "adminToken": "admin-secret",
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(rejected["message"], "unauthorized admin request")
        self.assertEqual(accepted["message"], "set balance success")
        self.assertEqual(accepted["wallet"]["balanceCents"], "2500")

    def test_user_actions_require_client_token_when_configured(self) -> None:
        with patch.dict("os.environ", {"EVO_TRAIN_CLIENT_TOKEN": "client-secret"}, clear=False):
            rejected = server_function.handle_request(
                json.dumps({"username": "pearl", "action": "余额查询"}, ensure_ascii=False)
            )
            accepted = server_function.handle_request(
                json.dumps(
                    {"username": "pearl", "action": "余额查询", "apiToken": "client-secret"},
                    ensure_ascii=False,
                )
            )

        self.assertEqual(rejected["message"], "unauthorized request")
        self.assertEqual(accepted["message"], "wallet query success")

    def test_autodl_instance_booting_status_is_not_settled_as_terminal(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "autodl-booting", 1000, "test setup"))
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "autodl-booting",
                status="Running",
                provider="autodl",
                remote_job_id="pro-1::runner-booting",
                hourly_price_cents=1000,
                frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), 1),
                started_at=sql_pack.utc_now_text(),
                billing_status="frozen",
            )
        )
        platform = SimpleNamespace(metadata=lambda job_id: {"status": "Instance:booting", "last_error": ""})

        with patch.object(server_function, "get_platform", return_value=platform):
            stats = server_function.scan_billing_tasks()

        self.assertEqual(stats["checked"], 1)
        task = sql_pack.sql_get_user_task("pearl", "autodl-booting")
        self.assertEqual(task["status"], "Instance:booting")
        self.assertEqual(task["billingStatus"], "frozen")
        self.assertEqual(sql_pack.sql_get_wallet("pearl")["frozenCents"], "1000")

    def test_result_download_uses_provider_chunk_protocol(self) -> None:
        sql_pack.sql_add_user_task(
            "pearl",
            "autodl-artifact",
            status="Succeeded",
            provider="autodl",
            remote_job_id="pro-1::runner-1",
            checkpoint_path="/root/autodl-tmp/evo_train/output",
        )
        platform = SimpleNamespace(
            download_artifact_chunk=lambda job_id, artifact_path, offset, chunk_size: {
                "artifactPath": artifact_path,
                "archivePath": "/root/autodl-tmp/evo_train/jobs/evo_train_runner-1/artifact.tar.gz",
                "offset": offset,
                "nextOffset": offset + chunk_size,
                "chunkSize": chunk_size,
                "totalBytes": 4096,
                "done": False,
                "dataBase64": "YWJj",
            }
        )

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "autodl-artifact",
                        "action": "结果下载",
                        "offset": 0,
                        "chunkSize": 3,
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "download artifact success")
        self.assertEqual(response["artifact"]["artifactPath"], "/root/autodl-tmp/evo_train/output")
        self.assertEqual(response["artifact"]["dataBase64"], "YWJj")

    def test_start_training_rejects_insufficient_balance(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 999)

        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "taskName": "run-006",
                    "action": "开始训练",
                    "provider": "aliyun",
                    "datasetPath": "/mnt/data/demo",
                    "epochs": 3,
                    "checkpointPath": "/mnt/checkpoints/run-006",
                    "checkpointFrequency": 1,
                    "gpuCount": 1,
                },
                ensure_ascii=False,
            )
        )

        self.assertIn("insufficient balance", response["message"])
        self.assertEqual(response["tasks"], [])
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "999")
        self.assertEqual(wallet["frozenCents"], "0")

    def test_sync_settles_succeeded_task_billing(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "run-007", 1000, "test setup"))
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "run-007",
                status="Running",
                provider="aliyun",
                remote_job_id="job-777",
                hourly_price_cents=1000,
                frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), 1),
                started_at=sql_pack.utc_now_text(),
                billing_status="frozen",
            )
        )

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(
                server_function.start_train,
                "fetch_job_body",
                return_value=_job_body(job_id="job-777", status="Succeeded"),
            ),
        ):
            response = server_function.handle_request(
                json.dumps({"username": "pearl", "action": "任务同步"}, ensure_ascii=False)
            )

        task = response["tasks"][0]
        self.assertEqual(task["status"], "Succeeded")
        self.assertEqual(task["billingStatus"], "settled")
        self.assertEqual(task["actualCostCents"], "1000")
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "1000")
        self.assertEqual(wallet["frozenCents"], "0")

    def test_billing_scan_stops_running_task_when_next_hour_cannot_be_frozen(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 1000)
        self.assertTrue(sql_pack.sql_freeze_user_balance("pearl", "run-008", 1000, "test setup"))
        self.assertTrue(
            sql_pack.sql_add_user_task(
                "pearl",
                "run-008",
                status="Running",
                provider="aliyun",
                remote_job_id="job-888",
                hourly_price_cents=1000,
                frozen_until=sql_pack.add_hours_text(sql_pack.utc_now_text(), -1),
                started_at=sql_pack.add_hours_text(sql_pack.utc_now_text(), -1),
                billing_status="frozen",
            )
        )

        with (
            patch.object(server_function.start_train, "load_env"),
            patch.object(server_function.start_train, "create_client", return_value=object()),
            patch.object(
                server_function.start_train,
                "fetch_job_body",
                return_value=_job_body(job_id="job-888", status="Running"),
            ),
            patch.object(server_function.start_train, "stop_job") as mocked_stop,
        ):
            stats = server_function.scan_billing_tasks()

        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["updated"], 1)
        mocked_stop.assert_called_once()
        task = sql_pack.sql_get_user_task("pearl", "run-008")
        self.assertEqual(task["status"], "STOPPED")
        self.assertEqual(task["billingStatus"], "settled")
        self.assertEqual(task["actualCostCents"], "1000")
        self.assertEqual(task["error"], "insufficient balance for next training hour")
        wallet = sql_pack.sql_get_wallet("pearl")
        self.assertEqual(wallet["balanceCents"], "0")
        self.assertEqual(wallet["frozenCents"], "0")


if __name__ == "__main__":
    unittest.main()
