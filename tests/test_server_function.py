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

    def test_rlinf_vla_plan_materializes_real_remote_command(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "repoUrl": "https://github.com/RLinf/RLinf.git",
                        "configName": "libero_pi0_grpo_smoke",
                        "algorithm": "grpo",
                        "modelFamily": "pi0",
                        "benchmark": "libero",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "maxSteps": 1000,
                        "evalEpisodes": 2,
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["workflow"], "rlinf_vla")
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(
            [stage["name"] for stage in plan["stages"]],
            ["prepare_code", "setup_env", "preflight", "write_contract", "train_rlinf_vla", "collect_artifacts"],
        )
        self.assertIn("git clone", plan["stages"][0]["command"])
        self.assertIn("test -f examples/embodiment/train_embodied_agent.py", plan["stages"][2]["command"])
        self.assertIn("run_contract.json", plan["stages"][3]["command"])
        self.assertIn("python examples/embodiment/train_embodied_agent.py", plan["stages"][4]["command"])
        self.assertIn("--config-name libero_pi0_grpo_smoke", plan["stages"][4]["command"])
        self.assertIn("runner.max_steps=1000", plan["stages"][4]["command"])
        self.assertIn("actor.model.model_type=pi0", plan["stages"][4]["command"])

    def test_start_training_materializes_rlinf_vla_for_autodl(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 3000)
        submitted_configs: list[dict[str, object]] = []

        def submit(job_config: dict[str, object]) -> str:
            submitted_configs.append(job_config)
            return "instance-1::rlinf-job-1"

        platform = SimpleNamespace(submit=submit)

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "rlinf-vla-smoke",
                        "action": "开始训练",
                        "provider": "autodl",
                        "workflow": "rlinf_vla",
                        "params": {
                            "repoUrl": "https://github.com/RLinf/RLinf.git",
                            "configName": "libero_pi0_grpo_smoke",
                            "datasetPath": "/root/autodl-tmp/datasets/libero",
                            "artifactPath": "/root/autodl-tmp/evo_train/jobs/rlinf-vla-smoke/artifacts",
                            "maxSteps": 1000,
                            "evalEpisodes": 2,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(response["tasks"][0]["provider"], "autodl")
        self.assertEqual(response["tasks"][0]["datasetPath"], "/root/autodl-tmp/datasets/libero")
        self.assertEqual(
            response["tasks"][0]["checkpointPath"],
            "/root/autodl-tmp/evo_train/jobs/rlinf-vla-smoke/artifacts",
        )
        self.assertIn("__EVO_STAGE_START__", submitted_configs[0]["command"])
        self.assertIn("train_rlinf_vla", submitted_configs[0]["command"])
        self.assertIn("libero_pi0_grpo_smoke", submitted_configs[0]["command"])
        self.assertTrue(submitted_configs[0]["autodl_managed"])

    def test_rlinf_vla_project_backend_matches_dexbotic_style_interface(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "launchMode": "project_backend",
                        "repoUrl": "https://github.com/dexmal/dexbotic.git",
                        "workdir": "/root/autodl-tmp/dexbotic",
                        "configName": "libero_goal_ppo_dexbotic_pi0",
                        "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
                        "rlinfExtModule": "dexbotic.rl.rlinf_registry",
                        "suite": "libero_goal",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/dexbotic-rl/artifacts",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(plan["params"]["launchMode"], "project_backend")
        self.assertEqual(plan["params"]["launcherModule"], "dexbotic.rl.model_rl_libero_pi0")
        self.assertEqual(plan["params"]["rlinfExtModule"], "dexbotic.rl.rlinf_registry")
        backend_interface = plan["params"]["backendInterface"]
        self.assertEqual(backend_interface["backendKind"], "rlinf")
        self.assertEqual(backend_interface["registryInjection"]["env"], "RLINF_EXT_MODULE")
        self.assertIn("rlinf", backend_interface["preflight"]["imports"])
        self.assertIn("VLA_RL_CONTRACT_PATH", backend_interface["envExports"])
        self.assertIn("run_contract.json", backend_interface["artifactContract"]["contractFile"])
        self.assertIn("import importlib", plan["stages"][2]["command"])
        self.assertIn("dexbotic.rl.rlinf_registry", plan["stages"][2]["command"])
        self.assertIn("import importlib", plan["stages"][2]["command"])
        self.assertIn("rlinf", plan["stages"][2]["command"])
        self.assertIn("run_contract.json", plan["stages"][3]["command"])
        self.assertIn("backendInterface", plan["stages"][3]["command"])
        self.assertIn("export RLINF_EXT_MODULE=dexbotic.rl.rlinf_registry", plan["stages"][4]["command"])
        self.assertIn("python -m dexbotic.rl.model_rl_libero_pi0", plan["stages"][4]["command"])
        self.assertIn("--suite=libero_goal", plan["stages"][4]["command"])
        self.assertIn("--dataset_path /root/autodl-tmp/datasets/libero", plan["stages"][4]["command"])
        self.assertIn("--artifact_path /root/autodl-tmp/evo_train/jobs/dexbotic-rl/artifacts", plan["stages"][4]["command"])

    def test_rlinf_vla_project_backend_requires_launcher_module(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "params": {
                        "launchMode": "project_backend",
                        "configName": "libero_goal_ppo_dexbotic_pi0",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertIn("launcherModule", response["plan"]["missingFields"])
        self.assertFalse(response["plan"]["readyToStart"])

    def test_vla_rl_plan_preserves_configured_backend_interface(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "vla_rl_backend",
                    "params": {
                        "backendKind": "mybackend",
                        "launchMode": "project_backend",
                        "repoUrl": "https://github.com/example/mybackend.git",
                        "workdir": "/root/autodl-tmp/mybackend",
                        "configName": "mybackend_smoke",
                        "launcherModule": "mybackend.train",
                        "backendExtModule": "mybackend.registry",
                        "datasetPath": "/root/autodl-tmp/datasets/demo",
                        "backendInterface": {
                            "interfaceVersion": "vla-rl-backend/v1",
                            "workflow": "vla_rl_backend",
                            "useLauncherContract": True,
                            "requiredParams": ["repoUrl", "launcherModule", "datasetPath", "artifactPath"],
                            "preflightImports": ["launcherModule", "backendExtModule"],
                            "preflightChecks": ["import mybackend"],
                            "usePreflightCommands": True,
                            "preflightCommands": ["test -d {datasetPath}", "mkdir -p {artifactPath}"],
                            "envExports": {
                                "MYBACKEND_DATASET": "datasetPath",
                                "MYBACKEND_MODE": "literal:smoke",
                            },
                            "launcherContract": {
                                "python_module": "python -m {launcherModule} --data {datasetPath} --out {artifactPath}"
                            },
                            "artifactContract": {"contractFile": "run_contract.json"},
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

        backend_interface = response["plan"]["params"]["backendInterface"]
        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(backend_interface["backendKind"], "mybackend")
        self.assertEqual(backend_interface["preflightChecks"], ["import mybackend"])
        self.assertNotIn("artifactPath", response["plan"]["missingFields"])
        self.assertIn("mybackend.registry", response["plan"]["stages"][2]["command"])
        self.assertIn("test -d /root/autodl-tmp/datasets/demo", response["plan"]["stages"][2]["command"])
        self.assertIn("mkdir -p /root/autodl-tmp/mybackend/outputs", response["plan"]["stages"][2]["command"])
        self.assertIn("export MYBACKEND_DATASET=/root/autodl-tmp/datasets/demo", response["plan"]["stages"][4]["command"])
        self.assertIn("export MYBACKEND_MODE=smoke", response["plan"]["stages"][4]["command"])
        self.assertIn("backendInterface", response["plan"]["stages"][3]["command"])
        self.assertIn("python -m mybackend.train --data /root/autodl-tmp/datasets/demo --out /root/autodl-tmp/mybackend/outputs", response["plan"]["stages"][4]["command"])

    def test_vla_rl_plan_uses_backend_interface_required_params(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "vla_rl_backend",
                    "params": {
                        "backendKind": "mybackend",
                        "launchMode": "project_backend",
                        "repoUrl": "https://github.com/example/mybackend.git",
                        "workdir": "/root/autodl-tmp/mybackend",
                        "configName": "mybackend_smoke",
                        "launcherModule": "mybackend.train",
                        "datasetPath": "/root/autodl-tmp/datasets/demo",
                        "backendInterface": {
                            "interfaceVersion": "vla-rl-backend/v1",
                            "workflow": "vla_rl_backend",
                            "requiredParams": ["rewardModule"],
                            "preflightImports": ["launcherModule", "rewardModule"],
                            "artifactContract": {"contractFile": "run_contract.json"},
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertIn("rewardModule", response["plan"]["missingFields"])
        self.assertFalse(response["plan"]["readyToStart"])

    def test_vla_rl_plan_rejects_unsupported_interface_launcher_kind(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "vla_rl_backend",
                    "params": {
                        "backendKind": "scriptbackend",
                        "launchMode": "project_backend",
                        "launcherKind": "python_module",
                        "repoUrl": "https://github.com/example/scriptbackend.git",
                        "workdir": "/root/autodl-tmp/scriptbackend",
                        "configName": "scriptbackend_smoke",
                        "launcherModule": "scriptbackend.train",
                        "datasetPath": "/root/autodl-tmp/datasets/demo",
                        "backendInterface": {
                            "interfaceVersion": "vla-rl-backend/v1",
                            "workflow": "vla_rl_backend",
                            "launcherKinds": ["python_script"],
                            "requiredParams": ["scriptPath"],
                            "artifactContract": {"contractFile": "run_contract.json"},
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertIn("launcherKind", response["plan"]["missingFields"])
        self.assertIn("scriptPath", response["plan"]["missingFields"])

    def test_vla_rl_plan_loads_backend_interface_from_env(self) -> None:
        configured = json.dumps(
            {
                "envbackend": {
                    "interfaceVersion": "vla-rl-backend/v1",
                    "workflow": "vla_rl_backend",
                    "preflightChecks": ["import envbackend"],
                    "artifactContract": {"contractFile": "run_contract.json"},
                }
            }
        )
        with patch.dict("os.environ", {"EVO_TRAIN_VLA_BACKEND_INTERFACES_JSON": configured}, clear=False):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "action": "AI配置训练",
                        "workflow": "vla_rl_backend",
                        "params": {
                            "backendKind": "envbackend",
                            "launchMode": "project_backend",
                            "repoUrl": "https://github.com/example/envbackend.git",
                            "workdir": "/root/autodl-tmp/envbackend",
                            "configName": "envbackend_smoke",
                            "launcherModule": "envbackend.train",
                            "datasetPath": "/root/autodl-tmp/datasets/demo",
                        },
                    },
                    ensure_ascii=False,
                )
            )

        backend_interface = response["plan"]["params"]["backendInterface"]
        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(backend_interface["backendKind"], "envbackend")
        self.assertEqual(backend_interface["preflightChecks"], ["import envbackend"])

    def test_rlinf_vla_infers_project_backend_when_launcher_module_is_present(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "params": {
                        "repoUrl": "https://github.com/dexmal/dexbotic.git",
                        "workdir": "/root/autodl-tmp/dexbotic",
                        "configName": "libero_goal_ppo_dexbotic_pi0",
                        "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
                        "rlinfExtModule": "dexbotic.rl.rlinf_registry",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        self.assertEqual(response["plan"]["params"]["launchMode"], "project_backend")
        self.assertIn("python -m dexbotic.rl.model_rl_libero_pi0", response["plan"]["stages"][4]["command"])

    def test_message_about_dexbotic_rlinf_uses_project_backend_defaults(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "message": "用 Dexbotic 接 RLinf backend 在 LIBERO 上做 pi0 PPO 后训练",
                    "provider": "autodl",
                    "params": {
                        "configName": "libero_goal_ppo_dexbotic_pi0",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["workflow"], "rlinf_vla")
        self.assertEqual(plan["params"]["launchMode"], "project_backend")
        self.assertEqual(plan["params"]["repoUrl"], "https://github.com/dexmal/dexbotic.git")
        self.assertEqual(plan["params"]["launcherModule"], "dexbotic.rl.model_rl_libero_pi0")
        self.assertEqual(plan["params"]["rlinfExtModule"], "dexbotic.rl.rlinf_registry")
        self.assertEqual(plan["params"]["suite"], "libero_goal")
        self.assertIn("algorithm.name=ppo", plan["params"]["overrides"])

    def test_project_backend_supports_deepspeed_simplevla_rl_launcher(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "launchMode": "project_backend",
                        "backendKind": "dexbotic",
                        "launcherKind": "deepspeed_script",
                        "repoUrl": "https://github.com/dexmal/dexbotic.git",
                        "workdir": "/root/autodl-tmp/dexbotic",
                        "scriptPath": "playground/benchmarks/libero/libero_simplevla_rl.py",
                        "task": "train",
                        "sftModelPath": "/root/autodl-tmp/checkpoints/pi0-sft",
                        "datasetName": "libero_10",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/simplevla-rl/artifacts",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(plan["params"]["backendKind"], "dexbotic")
        self.assertEqual(plan["params"]["launcherKind"], "deepspeed_script")
        self.assertIn("test -f playground/benchmarks/libero/libero_simplevla_rl.py", plan["stages"][2]["command"])
        self.assertIn("import importlib", plan["stages"][2]["command"])
        self.assertNotIn("import torch", plan["stages"][2]["command"])
        self.assertIn("deepspeed playground/benchmarks/libero/libero_simplevla_rl.py", plan["stages"][4]["command"])
        self.assertIn("--task train", plan["stages"][4]["command"])
        self.assertIn("--sft_model_path /root/autodl-tmp/checkpoints/pi0-sft", plan["stages"][4]["command"])
        self.assertIn("--dataset_name libero_10", plan["stages"][4]["command"])

    def test_project_backend_supports_optional_evaluation_stage(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "launchMode": "project_backend",
                        "launcherKind": "python_module",
                        "repoUrl": "https://github.com/example/roboclaw-vla.git",
                        "workdir": "/root/autodl-tmp/roboclaw-vla",
                        "configName": "libero_goal_pi0_ppo",
                        "launcherModule": "roboclaw_vla.rl.launcher",
                        "rlinfExtModule": "roboclaw_vla.rl.registry",
                        "preflightModules": ["roboclaw_vla.rl.adapters"],
                        "evalModule": "roboclaw_vla.rl.evaluate",
                        "suite": "libero_goal",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/roboclaw-vla/artifacts",
                        "launcherArgs": ["--profile", "smoke"],
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(
            [stage["name"] for stage in plan["stages"]],
            ["prepare_code", "setup_env", "preflight", "write_contract", "train_rlinf_vla", "evaluate", "collect_artifacts"],
        )
        self.assertIn("roboclaw_vla.rl.adapters", plan["stages"][2]["command"])
        self.assertIn("run_contract.json", plan["stages"][3]["command"])
        self.assertIn("--profile smoke", plan["stages"][4]["command"])
        self.assertIn("python -m roboclaw_vla.rl.evaluate", plan["stages"][5]["command"])
        self.assertIn("--checkpoint_path", plan["stages"][5]["command"])

    def test_roboclaw_grpo_profile_uses_real_import_preflight_and_config_template(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "builtinTrainingProfile": "roboclaw_grpo_backend",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "checkpointPath": "/root/autodl-tmp/checkpoints/pi0",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/roboclaw-grpo/artifacts",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(plan["params"]["builtinTrainingProfile"], "roboclaw_grpo_backend")
        self.assertEqual(plan["params"]["algorithm"], "grpo")
        self.assertEqual(plan["params"]["groupSize"], 8)
        self.assertEqual(plan["params"]["placementStrategy"], "single_node")
        self.assertEqual(plan["params"]["configName"], "libero_10_grpo_roboclaw")
        self.assertEqual(plan["params"]["configPath"], "roboclaw_vla/config/rl/libero_10_grpo_roboclaw.yaml")
        self.assertIn("test -f roboclaw_vla/config/rl/libero_10_grpo_roboclaw.yaml", plan["stages"][2]["command"])
        self.assertIn("importlib.import_module", plan["stages"][2]["command"])
        self.assertIn("rlinf", plan["stages"][2]["command"])
        self.assertIn("roboclaw_vla.rl.registry", plan["stages"][2]["command"])
        self.assertIn("roboclaw_vla.rl.launcher", plan["stages"][2]["command"])
        self.assertIn("algorithm.group_size=8", plan["params"]["overrides"])
        self.assertIn('"placementStrategy": "single_node"', plan["stages"][3]["command"])

    def test_pi0_grpo_request_selects_roboclaw_grpo_profile(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "modelFamily": "pi0",
                        "algorithm": "grpo",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "checkpointPath": "/root/autodl-tmp/checkpoints/pi0",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["params"]["builtinTrainingProfile"], "roboclaw_grpo_backend")
        self.assertEqual(plan["params"]["configName"], "libero_10_grpo_roboclaw")
        self.assertEqual(plan["params"]["groupSize"], 8)

    def test_rynnvla_request_selects_lerobot_project_backend(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "provider": "autodl",
                    "params": {
                        "modelFamily": "rynnvla",
                        "datasetPath": "/root/autodl-tmp/datasets/rynnvla",
                        "checkpointPath": "/root/autodl-tmp/checkpoints/rynnvla",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["params"]["builtinTrainingProfile"], "rynnvla_lerobot")
        self.assertEqual(plan["params"]["backendKind"], "lerobot")
        self.assertEqual(plan["params"]["policyFamily"], "rynnvla")
        self.assertEqual(plan["params"]["repoUrl"], "https://github.com/alibaba-damo-academy/RynnVLA-001.git")
        self.assertEqual(plan["params"]["scriptPath"], "train.py")
        self.assertEqual(plan["params"]["configName"], "lerobot_exp")

    def test_roboclaw_grpo_hydra_defaults_chain_exists(self) -> None:
        root = Path(__file__).resolve().parents[1] / "roboclaw" / "config" / "rl"

        for relative_path in (
            "libero_10_grpo_roboclaw.yaml",
            "env/libero_10.yaml",
            "model/roboclaw_pi0.yaml",
            "training_backend/fsdp.yaml",
        ):
            self.assertTrue((root / relative_path).is_file(), relative_path)

        main_config = (root / "libero_10_grpo_roboclaw.yaml").read_text(encoding="utf-8")
        env_config = (root / "env" / "libero_10.yaml").read_text(encoding="utf-8")
        model_config = (root / "model" / "roboclaw_pi0.yaml").read_text(encoding="utf-8")
        fsdp_config = (root / "training_backend" / "fsdp.yaml").read_text(encoding="utf-8")
        self.assertIn("env/libero_10@env.train", main_config)
        self.assertIn("env/libero_10@env.eval", main_config)
        self.assertIn("model/roboclaw_pi0@actor.model", main_config)
        self.assertIn("training_backend/fsdp@actor.fsdp_config", main_config)
        self.assertIn("total_num_envs: 16", env_config)
        self.assertIn("is_eval: true", env_config)
        self.assertIn("add_value_head: true", model_config)
        self.assertIn("precision: bfloat16", model_config)
        self.assertNotIn("torch_dtype", model_config)
        self.assertIn("enable_gradient_accumulation: true", fsdp_config)
        self.assertIn("mixed_precision:", fsdp_config)
        self.assertIn("use_orig_params: false", fsdp_config)

    def test_generic_vla_rl_backend_does_not_force_rlinf_import(self) -> None:
        for backend_kind in (
            "lerobot",
            "dexbotic",
            "custom",
        ):
            with self.subTest(backend_kind=backend_kind):
                response = server_function.handle_request(
                    json.dumps(
                        {
                            "username": "pearl",
                            "action": "AI配置训练",
                            "workflow": "vla_rl_backend",
                            "provider": "autodl",
                            "params": {
                                "backendKind": backend_kind,
                                "launchMode": "project_backend",
                                "launcherKind": "python_module",
                                "repoUrl": f"https://github.com/example/{backend_kind}-train.git",
                                "workdir": f"/root/autodl-tmp/{backend_kind}-train",
                                "configName": f"{backend_kind}_smoke",
                                "launcherModule": f"project.{backend_kind}.launch",
                                "backendExtModule": f"project.{backend_kind}.registry",
                                "datasetPath": f"/root/autodl-tmp/datasets/{backend_kind}",
                            },
                        },
                        ensure_ascii=False,
                    )
                )

                self.assertEqual(response["message"], "plan generated")
                plan = response["plan"]
                self.assertEqual(plan["workflow"], "vla_rl_backend")
                self.assertEqual(plan["missingFields"], [])
                self.assertEqual(plan["params"]["backendKind"], backend_kind)
                self.assertEqual(plan["params"]["backendExtModule"], f"project.{backend_kind}.registry")
                self.assertEqual([stage["name"] for stage in plan["stages"]][4], "train_vla_rl_backend")
                self.assertIn(f"VLA_RL_BACKEND_KIND={backend_kind}", plan["stages"][4]["command"])
                self.assertIn(f"VLA_RL_BACKEND_EXT_MODULE=project.{backend_kind}.registry", plan["stages"][4]["command"])
                self.assertIn(f"project.{backend_kind}.registry", plan["stages"][2]["command"])
                self.assertNotIn("importlib.import_module('rlinf')", plan["stages"][2]["command"])

    def test_vla_release_capabilities_are_normalized_into_contract_fields(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "message": "用 Pi0.5 co-training 在 XLeRobot 上做 action expert 和 LLM 联合优化，Blackwell 镜像",
                    "provider": "autodl",
                    "params": {
                        "configName": "xlerobot_pi05_cotrain",
                        "launcherModule": "project.training.pi05_cotrain",
                        "backendExtModule": "project.training.registry",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["workflow"], "rlinf_vla")
        self.assertEqual(plan["params"]["modelFamily"], "pi0.5")
        self.assertEqual(plan["params"]["trainingMode"], "co_training")
        self.assertEqual(plan["params"]["coTrainingTargets"], ["action_expert", "llm"])
        self.assertEqual(plan["params"]["robotAdapter"], "xlerobot")
        self.assertEqual(plan["params"]["imageProfile"], "blackwell")
        stage_commands = "\n".join(stage["command"] for stage in plan["stages"])
        self.assertIn("VLA_RL_MODEL_FAMILY=pi0.5", stage_commands)
        self.assertIn("VLA_RL_TRAINING_MODE=co_training", stage_commands)

    def test_vla_release_model_aliases_cover_navigation_and_gr00t(self) -> None:
        nav = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "message": "用 Uni-NaVid 做导航 VLA+RL 后训练",
                    "params": {"configName": "uni_navid_nav_smoke"},
                },
                ensure_ascii=False,
            )
        )
        gr00t = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "message": "GR00TN1 在 SO-101 上接 RLinf backend",
                    "params": {"configName": "gr00tn1_so101_smoke"},
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(nav["plan"]["workflow"], "rlinf_vla")
        self.assertEqual(nav["plan"]["params"]["modelFamily"], "uni-navid")
        self.assertEqual(nav["plan"]["params"]["trainingMode"], "rl_post_train")
        self.assertEqual(gr00t["plan"]["params"]["modelFamily"], "gr00tn1")
        self.assertEqual(gr00t["plan"]["params"]["robotAdapter"], "so-101")

    def test_builtin_training_profile_fills_verified_dexbotic_dm0_launcher(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "params": {
                        "modelFamily": "dm0",
                        "configName": "libero_goal_ppo_dexbotic_dm0",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/dm0-rl/artifacts",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(plan["params"]["builtinTrainingProfile"], "dexbotic_dm0_rlinf")
        self.assertEqual(plan["params"]["launcherModule"], "dexbotic.rl.model_rl_libero_dm0")
        self.assertEqual(plan["params"]["rlinfExtModule"], "dexbotic.rl.rlinf_registry")
        self.assertEqual(plan["params"]["trainingMode"], "rl_post_train")
        self.assertIn("python -m dexbotic.rl.model_rl_libero_dm0", plan["stages"][4]["command"])

    def test_builtin_training_profile_supports_simplevla_deepspeed_route(self) -> None:
        response = server_function.handle_request(
            json.dumps(
                {
                    "username": "pearl",
                    "action": "AI配置训练",
                    "workflow": "rlinf_vla",
                    "params": {
                        "builtinTrainingProfile": "dexbotic_simplevla_rl",
                        "datasetPath": "/root/autodl-tmp/datasets/libero",
                        "datasetName": "libero_10",
                        "sftModelPath": "/root/autodl-tmp/checkpoints/pi0-sft",
                        "artifactPath": "/root/autodl-tmp/evo_train/jobs/simplevla/artifacts",
                    },
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(response["message"], "plan generated")
        plan = response["plan"]
        self.assertEqual(plan["missingFields"], [])
        self.assertEqual(plan["params"]["backendKind"], "dexbotic")
        self.assertEqual(plan["params"]["launcherKind"], "deepspeed_script")
        self.assertEqual(plan["params"]["scriptPath"], "playground/benchmarks/libero/libero_simplevla_rl.py")
        self.assertIn("deepspeed playground/benchmarks/libero/libero_simplevla_rl.py", plan["stages"][4]["command"])

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

    def test_autodl_remote_command_creates_log_directory_before_nohup_redirect(self) -> None:
        platform = AutoDLPlatform(host="demo.autodl", port=22, user="root", key_path="/tmp/key")

        command = platform._build_remote_command("runner-1", "python3 -V", "/root")

        job_dir = "/root/autodl-tmp/evo_train/jobs/evo_train_runner-1"
        self.assertTrue(command.startswith(f"mkdir -p {job_dir}; nohup "))
        self.assertIn(f">{job_dir}/run.log", command)

    def test_autodl_exec_drains_output_before_waiting_for_exit_status(self) -> None:
        events: list[str] = []

        class FakeChannel:
            def recv_exit_status(self) -> int:
                events.append("exit")
                return 0

        class FakeStream:
            def __init__(self, label: str, payload: bytes) -> None:
                self.label = label
                self.payload = payload
                self.channel = FakeChannel()

            def read(self) -> bytes:
                events.append(self.label)
                return self.payload

        class FakeClient:
            def exec_command(self, command: str):
                events.append(f"exec:{command}")
                return None, FakeStream("stdout", b"ok\n"), FakeStream("stderr", b"")

            def close(self) -> None:
                events.append("close")

        platform = AutoDLPlatform(host="demo.autodl", port=22, user="root", key_path="/tmp/key")

        with patch.object(platform, "_connect", return_value=FakeClient()):
            exit_status, stdout, stderr = platform._exec("python3 -V")

        self.assertEqual((exit_status, stdout, stderr), (0, "ok", ""))
        self.assertEqual(events, ["exec:python3 -V", "stdout", "stderr", "exit", "close"])

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

    def test_autodl_image_query_returns_ready_images(self) -> None:
        images = json.dumps(
            [
                {
                    "imageId": "pytorch-cu121",
                    "displayName": "PyTorch CUDA 12.1",
                    "autodlImageUuid": "image-cu121",
                    "cudaVFrom": 121,
                }
            ]
        )

        with patch.dict("os.environ", {"AUTODL_IMAGES_JSON": images}, clear=True):
            response = server_function.handle_request(
                json.dumps({"action": "AutoDL镜像查询"}, ensure_ascii=False)
            )

        self.assertEqual(response["message"], "autodl image query success")
        self.assertEqual(response["images"][0]["imageId"], "pytorch-cu121")
        self.assertEqual(response["images"][0]["autodlImageUuid"], "image-cu121")
        self.assertEqual(response["images"][0]["cudaVFrom"], "121")
        self.assertTrue(response["images"][0]["readyToStart"])

    def test_runtime_match_scores_gpu_and_image_capabilities(self) -> None:
        skus = json.dumps(
            [
                {
                    "skuId": "autodl-4090d",
                    "provider": "autodl",
                    "displayName": "RTX 4090D",
                    "gpuSpec": "4090d",
                    "autodlGpuSpecUuid": "4090D",
                    "gpuCount": 1,
                    "costHourlyCents": 900,
                    "gpuMemoryGb": 24,
                    "supportedBackends": ["rlinf", "lerobot"],
                    "supportedModels": ["pi0", "dm0"],
                    "supportedBenchmarks": ["libero", "maniskill"],
                    "supportedAlgorithms": ["grpo", "ppo"],
                    "supportedTrainingModes": ["rl_post_train"],
                    "capabilities": ["cuda121", "rollout_video"],
                    "simFrameworks": ["mujoco"],
                },
                {
                    "skuId": "autodl-small",
                    "provider": "autodl",
                    "displayName": "Small GPU",
                    "gpuSpec": "small",
                    "autodlGpuSpecUuid": "small",
                    "gpuCount": 1,
                    "costHourlyCents": 100,
                    "gpuMemoryGb": 8,
                    "supportedBackends": ["lerobot"],
                    "supportedModels": ["act"],
                    "supportedBenchmarks": ["metaworld"],
                },
            ]
        )
        images = json.dumps(
            [
                {
                    "imageId": "vla-rlinf-cu121",
                    "displayName": "VLA RLinf CUDA 12.1",
                    "autodlImageUuid": "image-rlinf",
                    "cudaVFrom": 121,
                    "supportedBackends": ["rlinf"],
                    "supportedModels": ["pi0", "dm0"],
                    "supportedBenchmarks": ["libero", "maniskill"],
                    "supportedAlgorithms": ["grpo", "ppo"],
                    "supportedTrainingModes": ["rl_post_train"],
                    "frameworks": ["rlinf", "roboclaw"],
                    "capabilities": ["cuda121", "libero_assets"],
                    "datasetFormats": ["libero"],
                },
                {
                    "imageId": "lerobot-cu121",
                    "displayName": "LeRobot CUDA 12.1",
                    "autodlImageUuid": "image-lerobot",
                    "cudaVFrom": 121,
                    "supportedBackends": ["lerobot"],
                    "supportedModels": ["act"],
                    "supportedBenchmarks": ["metaworld"],
                    "frameworks": ["lerobot"],
                },
            ]
        )

        with patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus, "AUTODL_IMAGES_JSON": images}, clear=True):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "action": "训练运行时匹配",
                        "provider": "autodl",
                        "params": {
                            "backendKind": "rlinf",
                            "modelFamily": "pi0",
                            "benchmark": "libero",
                            "algorithm": "grpo",
                            "trainingMode": "rl_post_train",
                            "requiredCapabilities": ["cuda121", "mujoco", "libero_assets"],
                            "minGpuMemoryGb": 24,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "runtime match success")
        self.assertTrue(response["readyToStart"])
        self.assertEqual(response["matches"][0]["sku"]["skuId"], "autodl-4090d")
        self.assertEqual(response["matches"][0]["image"]["imageId"], "vla-rlinf-cu121")
        self.assertTrue(response["matches"][0]["compatible"])
        self.assertIn("gpu memory ok: 24GB", response["matches"][0]["reasons"])
        self.assertIn("capabilities ok: cuda121, mujoco, libero_assets", response["matches"][0]["reasons"])
        self.assertTrue(any(match["blockingReasons"] for match in response["matches"][1:]))

    def test_runtime_match_blocks_incompatible_gpu_and_image_pair(self) -> None:
        skus = json.dumps(
            [
                {
                    "skuId": "autodl-small",
                    "provider": "autodl",
                    "displayName": "Small GPU",
                    "autodlGpuSpecUuid": "small",
                    "gpuCount": 1,
                    "costHourlyCents": 100,
                    "gpuMemoryGb": 8,
                    "supportedBackends": ["lerobot"],
                    "supportedModels": ["act"],
                    "supportedBenchmarks": ["metaworld"],
                    "capabilities": ["cuda121"],
                }
            ]
        )
        images = json.dumps(
            [
                {
                    "imageId": "lerobot-cu121",
                    "displayName": "LeRobot CUDA 12.1",
                    "autodlImageUuid": "image-lerobot",
                    "cudaVFrom": 121,
                    "supportedBackends": ["lerobot"],
                    "supportedModels": ["act"],
                    "supportedBenchmarks": ["metaworld"],
                    "capabilities": ["cuda121"],
                }
            ]
        )

        with patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus, "AUTODL_IMAGES_JSON": images}, clear=True):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "action": "训练运行时匹配",
                        "provider": "autodl",
                        "params": {
                            "backendKind": "rlinf",
                            "modelFamily": "pi0",
                            "benchmark": "libero",
                            "requiredCapabilities": ["cuda121", "libero_assets"],
                            "minGpuMemoryGb": 24,
                        },
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "runtime match success")
        self.assertFalse(response["readyToStart"])
        self.assertEqual(len(response["matches"]), 1)
        match = response["matches"][0]
        self.assertFalse(match["compatible"])
        self.assertIn("sku does not support backend: rlinf", match["blockingReasons"])
        self.assertIn("image does not support benchmark: libero", match["blockingReasons"])
        self.assertIn("gpu memory too small: 8GB < 24GB", match["blockingReasons"])
        self.assertIn("missing capabilities: libero_assets", match["blockingReasons"])

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

    def test_autodl_start_training_combines_gpu_sku_and_image_id(self) -> None:
        sql_pack.sql_set_user_balance("pearl", 2000)
        skus = json.dumps(
            [
                {
                    "skuId": "sku-4090",
                    "provider": "autodl",
                    "displayName": "RTX 4090 48G",
                    "gpuSpec": "4090-48g",
                    "autodlGpuSpecUuid": "v-48g",
                    "gpuCount": 1,
                    "costHourlyCents": 901,
                }
            ]
        )
        images = json.dumps(
            [
                {
                    "imageId": "pytorch-cu121",
                    "displayName": "PyTorch CUDA 12.1",
                    "autodlImageUuid": "image-cu121",
                    "cudaVFrom": 121,
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
            "taskName": "autodl-sku-image",
            "action": "开始训练",
            "provider": "autodl",
            "skuId": "sku-4090",
            "imageId": "pytorch-cu121",
            "workflow": "custom_project",
            "params": {
                "repoUrl": "https://example.com/repo.git",
                "trainCommand": "python train.py",
            },
        }

        with (
            patch.dict("os.environ", {"AUTODL_GPU_SKUS_JSON": skus, "AUTODL_IMAGES_JSON": images}, clear=False),
            patch.object(server_function, "get_platform", return_value=FakePlatform()),
        ):
            response = server_function.handle_request(json.dumps(request, ensure_ascii=False))

        self.assertEqual(response["message"], "create task success")
        self.assertEqual(captured["autodl_gpu_spec_uuid"], "v-48g")
        self.assertEqual(captured["autodl_image_uuid"], "image-cu121")
        self.assertEqual(captured["autodl_cuda_v_from"], 121)

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

        self.assertIn("missing required field: imageId", response["message"])
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

    def test_loss_download_uses_provider_chunk_protocol(self) -> None:
        sql_pack.sql_add_user_task(
            "pearl",
            "autodl-loss",
            status="Succeeded",
            provider="autodl",
            remote_job_id="pro-1::runner-loss",
            checkpoint_path="/root/autodl-tmp/evo_train/output",
        )
        platform = SimpleNamespace(
            download_artifact_chunk=lambda job_id, artifact_path, offset, chunk_size: {
                "artifactPath": artifact_path,
                "archivePath": "/root/autodl-tmp/evo_train/jobs/evo_train_runner-loss/artifact.tar.gz",
                "offset": offset,
                "nextOffset": offset + chunk_size,
                "chunkSize": chunk_size,
                "totalBytes": 128,
                "done": True,
                "dataBase64": "bG9zcw==",
            }
        )

        with patch.object(server_function, "get_platform", return_value=platform):
            response = server_function.handle_request(
                json.dumps(
                    {
                        "username": "pearl",
                        "taskName": "autodl-loss",
                        "action": "下载损失",
                        "offset": 0,
                        "chunkSize": 4,
                    },
                    ensure_ascii=False,
                )
            )

        self.assertEqual(response["message"], "download loss success")
        self.assertEqual(response["artifact"]["artifactPath"], "/root/autodl-tmp/evo_train/output/loss/loss.txt")
        self.assertEqual(response["artifact"]["dataBase64"], "bG9zcw==")

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
