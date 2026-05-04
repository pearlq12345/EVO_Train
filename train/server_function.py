#!/usr/bin/env python3
"""Training task business functions for the TCP server."""

from __future__ import annotations

import argparse
import json
from typing import Any

from sql_lite.sql_pack import (
    sql_add_user_task,
    sql_delete_user_task,
    sql_get_user_all_task,
    sql_get_user_task,
    sql_update_user_task,
)
from train import start_train


TERMINAL_STATUSES = start_train.DONE_STATUSES | start_train.FAILED_STATUSES | {"STOPPED"}


def _request_string(request: dict[str, Any], key: str) -> str:
    return str(request.get(key) or "").strip()


def _request_optional_string(request: dict[str, Any], key: str) -> str | None:
    value = _request_string(request, key)
    return value or None


def _request_int(request: dict[str, Any], key: str, *, required: bool = False, default: int | None = None) -> int | None:
    value = request.get(key)
    if value is None or value == "":
        if required:
            raise ValueError(f"missing required field: {key}")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field: {key}") from exc


def _request_bool(request: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = request.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "y", "on"}


def _request_mounts(request: dict[str, Any]) -> list[str] | None:
    mounts = request.get("mount")
    if mounts is None or mounts == "":
        return None
    if isinstance(mounts, list):
        return [str(item).strip() for item in mounts if str(item).strip()]
    return [str(mounts).strip()]


def _build_submit_args(request: dict[str, Any], task_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        region=_request_optional_string(request, "region") or "cn-hangzhou",
        env_file=_request_optional_string(request, "envFile"),
        dataset_path=_request_string(request, "datasetPath"),
        epochs=_request_int(request, "epochs", required=True),
        checkpoint_path=_request_string(request, "checkpointPath"),
        checkpoint_frequency=_request_int(request, "checkpointFrequency", required=True),
        command=_request_optional_string(request, "command"),
        command_template=_request_optional_string(request, "commandTemplate"),
        job_name=_request_optional_string(request, "jobName") or task_name,
        workspace_id=_request_optional_string(request, "workspaceId"),
        resource_id=_request_optional_string(request, "resourceId"),
        image=_request_optional_string(request, "image"),
        description=_request_optional_string(request, "description"),
        ecs_spec=_request_optional_string(request, "ecsSpec"),
        gpu_count=_request_int(request, "gpuCount", required=True),
        gpu_type=_request_optional_string(request, "gpuType"),
        cpu=_request_int(request, "cpu"),
        memory=_request_int(request, "memory"),
        max_running_minutes=_request_int(request, "maxRunningMinutes"),
        mount=_request_mounts(request),
        wait=_request_bool(request, "wait"),
        timeout=_request_int(request, "timeout", default=86400),
        interval=_request_int(request, "interval", default=30),
        dry_run=_request_bool(request, "dryRun"),
    )


def _create_aliyun_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if sql_get_user_task(username, task_name) is not None:
        return "create task failed", sql_get_user_all_task(username)

    provider = _request_optional_string(request, "provider") or "aliyun"
    if provider != "aliyun":
        return f"unsupported provider: {provider}", sql_get_user_all_task(username)

    try:
        args = _build_submit_args(request, task_name)
        if not args.dataset_path:
            raise ValueError("missing required field: datasetPath")
        if not args.checkpoint_path:
            raise ValueError("missing required field: checkpointPath")
        start_train.load_env(args.env_file)
        client = start_train.create_client(args.region)
        body = start_train.create_job(client, args)
        if body is None:
            return "dry run only", sql_get_user_all_task(username)
        remote_job_id = str(getattr(body, "job_id", "") or "")
        if not remote_job_id:
            return "create task failed: missing remote job id", sql_get_user_all_task(username)
        created = sql_add_user_task(
            username,
            task_name,
            status="Submitted",
            provider="aliyun",
            remote_job_id=remote_job_id,
            checkpoint_path=args.checkpoint_path,
            dataset_path=args.dataset_path,
        )
        if not created:
            return "create task failed", sql_get_user_all_task(username)
        return "create task success", sql_get_user_all_task(username)
    except (ValueError, SystemExit) as exc:
        return str(exc), sql_get_user_all_task(username)
    except Exception as exc:
        return f"create task failed: {exc}", sql_get_user_all_task(username)


def _refresh_aliyun_task(task: dict[str, str], env_file: str | None = None, region: str | None = None) -> None:
    if task.get("provider") != "aliyun":
        return
    remote_job_id = task.get("jobId", "")
    status = task.get("status", "")
    if not remote_job_id or status in TERMINAL_STATUSES:
        return

    start_train.load_env(env_file)
    client = start_train.create_client(region or "cn-hangzhou")
    body = start_train.fetch_job_body(client, remote_job_id, need_detail=False)
    payload = start_train.build_status_payload(body)
    sql_update_user_task(
        task["username"],  # type: ignore[index]
        task["taskName"],
        status=payload["status"] or status,
        last_error=payload["reason_message"] or "",
    )


def _refresh_user_tasks(username: str, request: dict[str, Any]) -> list[dict[str, str]]:
    tasks = sql_get_user_all_task(username)
    if not tasks:
        return tasks

    env_file = _request_optional_string(request, "envFile")
    region = _request_optional_string(request, "region") or "cn-hangzhou"
    for task in tasks:
        task["username"] = username  # internal helper context, stripped before response
        try:
            _refresh_aliyun_task(task, env_file=env_file, region=region)
        except (SystemExit, Exception) as exc:
            sql_update_user_task(
                username,
                task["taskName"],
                last_error=str(exc),
            )
    refreshed = sql_get_user_all_task(username)
    return refreshed


def _stop_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return "stop task failed", sql_get_user_all_task(username)

    if task.get("provider") == "aliyun" and task.get("jobId") and task.get("status") not in TERMINAL_STATUSES:
        try:
            env_file = _request_optional_string(request, "envFile")
            region = _request_optional_string(request, "region") or "cn-hangzhou"
            start_train.load_env(env_file)
            client = start_train.create_client(region)
            start_train.stop_job(client, task["jobId"])
            sql_update_user_task(username, task_name, status="STOPPED", last_error="")
            return "stop task success", sql_get_user_all_task(username)
        except (SystemExit, Exception) as exc:
            sql_update_user_task(username, task_name, last_error=str(exc))
            return f"stop task failed: {exc}", sql_get_user_all_task(username)

    updated = sql_update_user_task(username, task_name, status="STOPPED")
    return ("stop task success" if updated else "stop task failed"), sql_get_user_all_task(username)


def handle_request(text: str) -> dict[str, Any]:
    """Handle one complete JSON request and return a response dict."""
    try:
        request = json.loads(text)
    except json.JSONDecodeError:
        return {"message": "invalid json", "tasks": []}

    username = _request_string(request, "username")
    task_name = _request_string(request, "taskName")
    action = _request_string(request, "action")

    if action == "任务同步":
        return {"message": "sync success", "tasks": _refresh_user_tasks(username, request)}

    if not username or not task_name:
        return {"message": "invalid request", "tasks": sql_get_user_all_task(username)}

    if action == "开始训练":
        message, tasks = _create_aliyun_task(username, task_name, request)
    elif action == "结束训练":
        message, tasks = _stop_task(username, task_name, request)
    elif action == "删除任务":
        message = "delete task success" if sql_delete_user_task(username, task_name) else "delete task failed"
        tasks = sql_get_user_all_task(username)
    else:
        message = "invalid action"
        tasks = sql_get_user_all_task(username)

    return {"message": message, "tasks": tasks}


def handle_request_text(text: str) -> str:
    """Handle one complete JSON request and return a JSON response string."""
    return json.dumps(handle_request(text), ensure_ascii=False)
