#!/usr/bin/env python3
"""Training task business functions for the TCP server."""

from __future__ import annotations

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
from train.platform.factory import get_platform


TERMINAL_STATUSES = {"Succeeded", "Succeed", "SUCCESS", "SUCCEEDED", "Failed", "FAILED", "Stopped", "STOPPED", "Deleted", "DELETED"}


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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _request_mounts(request: dict[str, Any]) -> list[str] | None:
    mounts = request.get("mount")
    if mounts is None or mounts == "":
        return None
    if isinstance(mounts, list):
        return [str(item).strip() for item in mounts if str(item).strip()]
    mount = str(mounts).strip()
    return [mount] if mount else None


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or "").strip().lower()
    if not normalized or normalized == "aliyun_dlc":
        return "aliyun"
    return normalized


def _build_job_config(request: dict[str, Any], task_name: str) -> dict[str, Any]:
    return {
        "region": _request_optional_string(request, "region") or "cn-hangzhou",
        "env_file": _request_optional_string(request, "envFile"),
        "dataset_path": _request_string(request, "datasetPath"),
        "epochs": _request_int(request, "epochs", required=True),
        "checkpoint_path": _request_string(request, "checkpointPath"),
        "checkpoint_frequency": _request_int(request, "checkpointFrequency", required=True),
        "command": _request_optional_string(request, "command"),
        "command_template": _request_optional_string(request, "commandTemplate"),
        "job_name": _request_optional_string(request, "jobName") or task_name,
        "workspace_id": _request_optional_string(request, "workspaceId"),
        "resource_id": _request_optional_string(request, "resourceId"),
        "image": _request_optional_string(request, "image"),
        "description": _request_optional_string(request, "description"),
        "ecs_spec": _request_optional_string(request, "ecsSpec"),
        "gpu_count": _request_int(request, "gpuCount", required=True),
        "gpu_type": _request_optional_string(request, "gpuType"),
        "cpu": _request_int(request, "cpu"),
        "memory": _request_int(request, "memory"),
        "max_running_minutes": _request_int(request, "maxRunningMinutes"),
        "mount": _request_mounts(request),
        "wait": _request_bool(request, "wait"),
        "timeout": _request_int(request, "timeout", default=86400),
        "interval": _request_int(request, "interval", default=30),
        "dry_run": _request_bool(request, "dryRun"),
    }


def _resolve_platform(provider: str, request: dict[str, Any]) -> Any:
    return get_platform(provider, region_id=_request_optional_string(request, "region") or "cn-hangzhou")


def _create_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if sql_get_user_task(username, task_name) is not None:
        return "create task failed", sql_get_user_all_task(username)

    provider = _normalize_provider(_request_optional_string(request, "provider"))
    try:
        platform = _resolve_platform(provider, request)
    except ValueError:
        return f"unsupported provider: {provider}", sql_get_user_all_task(username)

    try:
        job_config = _build_job_config(request, task_name)
        if not job_config["dataset_path"]:
            raise ValueError("missing required field: datasetPath")
        if not job_config["checkpoint_path"]:
            raise ValueError("missing required field: checkpointPath")

        start_train.load_env(job_config["env_file"])
        remote_job_id = platform.submit(job_config)
        if job_config["dry_run"]:
            return "dry run only", sql_get_user_all_task(username)
        if not remote_job_id:
            return "create task failed: missing remote job id", sql_get_user_all_task(username)

        created = sql_add_user_task(
            username,
            task_name,
            status="Submitted",
            provider=provider,
            remote_job_id=remote_job_id,
            checkpoint_path=str(job_config["checkpoint_path"] or ""),
            dataset_path=str(job_config["dataset_path"] or ""),
        )
        if not created:
            return "create task failed", sql_get_user_all_task(username)
        return "create task success", sql_get_user_all_task(username)
    except (ValueError, SystemExit) as exc:
        return str(exc), sql_get_user_all_task(username)
    except Exception as exc:
        return f"create task failed: {exc}", sql_get_user_all_task(username)


def _refresh_task(username: str, task: dict[str, str], request: dict[str, Any]) -> None:
    provider = _normalize_provider(task.get("provider"))
    remote_job_id = task.get("jobId", "")
    status = task.get("status", "")
    if not remote_job_id or status in TERMINAL_STATUSES:
        return

    start_train.load_env(_request_optional_string(request, "envFile"))
    platform = _resolve_platform(provider, request)
    metadata = platform.metadata(remote_job_id)
    sql_update_user_task(
        username,
        task["taskName"],
        status=metadata.get("status") or status,
        last_error=metadata.get("last_error", ""),
    )


def _refresh_user_tasks(username: str, request: dict[str, Any]) -> list[dict[str, str]]:
    tasks = sql_get_user_all_task(username)
    for task in tasks:
        try:
            _refresh_task(username, task, request)
        except Exception as exc:
            sql_update_user_task(username, task["taskName"], last_error=str(exc))
    return sql_get_user_all_task(username)


def _stop_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return "stop task failed", sql_get_user_all_task(username)

    remote_job_id = task.get("jobId", "")
    if remote_job_id and task.get("status") not in TERMINAL_STATUSES:
        provider = _normalize_provider(task.get("provider"))
        try:
            start_train.load_env(_request_optional_string(request, "envFile"))
            platform = _resolve_platform(provider, request)
            platform.stop(remote_job_id)
            sql_update_user_task(username, task_name, status="STOPPED", last_error="")
            return "stop task success", sql_get_user_all_task(username)
        except Exception as exc:
            sql_update_user_task(username, task_name, last_error=str(exc))
            return f"stop task failed: {exc}", sql_get_user_all_task(username)

    updated = sql_update_user_task(username, task_name, status="STOPPED", last_error="")
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
        message, tasks = _create_task(username, task_name, request)
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
