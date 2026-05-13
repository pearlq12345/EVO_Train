#!/usr/bin/env python3
"""Training task business functions for the TCP server."""

from __future__ import annotations

import argparse
import hmac
import json
import os
from decimal import Decimal, ROUND_CEILING
from typing import Any

from sql_lite.sql_pack import (
    add_hours_text,
    parse_timestamp,
    sql_charge_frozen_balance,
    sql_add_user_task,
    sql_delete_user_task,
    sql_freeze_user_balance,
    sql_get_billing_watch_tasks,
    sql_get_hourly_price,
    sql_get_gpu_prices,
    sql_get_task_frozen_cents,
    sql_get_user_all_task,
    sql_get_user_task,
    sql_get_wallet,
    sql_get_billing_records,
    sql_refund_frozen_balance,
    sql_set_gpu_price,
    sql_set_user_balance,
    sql_update_user_task,
    utc_now_text,
)
from train import start_train
from train.platform.autodl import AutoDLApiClient
from train.platform.factory import get_platform, normalize_provider
from train.workflows import build_training_plan, materialize_training_request


TERMINAL_STATUSES = start_train.DONE_STATUSES | start_train.FAILED_STATUSES | {"STOPPED"}
TERMINAL_INSTANCE_STATUSES = {
    "Instance:stopped",
    "Instance:stopping",
    "Instance:shutdown",
    "Instance:released",
    "Instance:deleted",
    "Instance:not_found",
}
ADMIN_ACTIONS = {"管理员充值", "价格设置", "平台余额查询"}
USER_ACTIONS = {
    "余额查询",
    "账单查询",
    "任务同步",
    "开始训练",
    "结束训练",
    "删除任务",
    "结果下载",
    "AI配置训练",
    "查询状态",
    "查询下载目录",
    "请求用户日志",
    "GPU规格查询",
}


DEFAULT_AUTODL_SKUS = [
    {
        "skuId": "autodl-h800-80g",
        "provider": "autodl",
        "displayName": "H800 80G",
        "gpuSpec": "h800-80g",
        "autodlGpuSpecUuid": "h800",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-4090-48g",
        "provider": "autodl",
        "displayName": "RTX 4090 48G",
        "gpuSpec": "4090-48g",
        "autodlGpuSpecUuid": "v-48g",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-pro6000-96g",
        "provider": "autodl",
        "displayName": "PRO6000 96G",
        "gpuSpec": "pro6000-96g",
        "autodlGpuSpecUuid": "pro6000-p",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-4080s-32g",
        "provider": "autodl",
        "displayName": "RTX 4080(S) 32G",
        "gpuSpec": "4080s-32g",
        "autodlGpuSpecUuid": "v-32g-p",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-3090-48g",
        "provider": "autodl",
        "displayName": "RTX 3090 48G",
        "gpuSpec": "3090-48g",
        "autodlGpuSpecUuid": "v-48g-350w",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-5090-32g",
        "provider": "autodl",
        "displayName": "RTX 5090 32G",
        "gpuSpec": "5090-32g",
        "autodlGpuSpecUuid": "5090-p",
        "gpuCount": 1,
        "enabled": True,
    },
    {
        "skuId": "autodl-4090d",
        "provider": "autodl",
        "displayName": "RTX 4090D",
        "gpuSpec": "4090d",
        "autodlGpuSpecUuid": "4090D",
        "gpuCount": 1,
        "enabled": True,
    },
]


def _is_terminal_status(status: str) -> bool:
    return status in TERMINAL_STATUSES or status in TERMINAL_INSTANCE_STATUSES


def _request_string(request: dict[str, Any], key: str) -> str:
    return str(request.get(key) or "").strip()


def _request_optional_string(request: dict[str, Any], key: str) -> str | None:
    value = _request_string(request, key)
    return value or None


def _request_auth_token(request: dict[str, Any]) -> str:
    return (
        _request_string(request, "apiToken")
        or _request_string(request, "token")
        or _request_string(request, "adminToken")
    )


def _token_matches(provided: str, expected: str | None) -> bool:
    if not expected:
        return True
    return bool(provided) and hmac.compare_digest(provided, expected)


def _authorization_error(request: dict[str, Any], action: str) -> dict[str, Any] | None:
    provided = _request_auth_token(request)
    if action in ADMIN_ACTIONS:
        admin_token = os.environ.get("EVO_TRAIN_ADMIN_TOKEN")
        if not _token_matches(provided, admin_token):
            return {"message": "unauthorized admin request", "tasks": []}
        return None
    if action in USER_ACTIONS:
        client_token = os.environ.get("EVO_TRAIN_CLIENT_TOKEN")
        admin_token = os.environ.get("EVO_TRAIN_ADMIN_TOKEN")
        if client_token and not (
            _token_matches(provided, client_token)
            or (admin_token is not None and _token_matches(provided, admin_token))
        ):
            return {"message": "unauthorized request", "tasks": []}
    return None


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


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _service_fee_rate() -> Decimal:
    raw = os.environ.get("EVO_TRAIN_SERVICE_FEE_RATE", "0.10")
    try:
        rate = Decimal(raw)
    except Exception:
        rate = Decimal("0.10")
    return max(rate, Decimal("0"))


def _sale_price_from_cost(cost_cents: int) -> int:
    if cost_cents <= 0:
        return 0
    multiplier = Decimal("1") + _service_fee_rate()
    return int((Decimal(cost_cents) * multiplier).quantize(Decimal("1"), rounding=ROUND_CEILING))


def _load_autodl_skus() -> list[dict[str, Any]]:
    raw = os.environ.get("AUTODL_GPU_SKUS_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid AUTODL_GPU_SKUS_JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("AUTODL_GPU_SKUS_JSON must be a JSON array")
        skus = [dict(item) for item in parsed if isinstance(item, dict)]
    else:
        skus = [dict(item) for item in DEFAULT_AUTODL_SKUS]

    default_image_uuid = os.environ.get("AUTODL_IMAGE_UUID", "")
    default_cuda_v_from = os.environ.get("AUTODL_CUDA_V_FROM", "")
    default_data_centers = os.environ.get("AUTODL_DATA_CENTER_LIST", "")
    for sku in skus:
        sku.setdefault("provider", "autodl")
        sku.setdefault("enabled", True)
        if default_image_uuid and not sku.get("autodlImageUuid"):
            sku["autodlImageUuid"] = default_image_uuid
        if default_cuda_v_from and not sku.get("cudaVFrom"):
            sku["cudaVFrom"] = int(default_cuda_v_from)
        if default_data_centers and not sku.get("autodlDataCenters"):
            sku["autodlDataCenters"] = default_data_centers
        cost = int(sku.get("costHourlyCents") or 0)
        if cost > 0 and not sku.get("hourlyPriceCents"):
            sku["hourlyPriceCents"] = _sale_price_from_cost(cost)
    return skus


def _public_sku(sku: dict[str, Any]) -> dict[str, str | bool]:
    cost = int(sku.get("costHourlyCents") or 0)
    sale = int(sku.get("hourlyPriceCents") or _sale_price_from_cost(cost) or 0)
    response: dict[str, str | bool] = {
        "skuId": str(sku.get("skuId") or ""),
        "provider": str(sku.get("provider") or "autodl"),
        "displayName": str(sku.get("displayName") or sku.get("gpuSpec") or sku.get("skuId") or ""),
        "gpuSpec": str(sku.get("gpuSpec") or sku.get("skuId") or ""),
        "gpuCount": str(sku.get("gpuCount") or 1),
        "hourlyPriceCents": str(sale),
        "serviceFeeRate": str(_service_fee_rate()),
        "enabled": bool(sku.get("enabled", True)),
    }
    if cost > 0:
        response["costHourlyCents"] = str(cost)
    return response


def _gpu_sku_response(request: dict[str, Any]) -> dict[str, Any]:
    provider = normalize_provider(_request_optional_string(request, "provider") or "autodl")
    include_disabled = _request_bool(request, "includeDisabled")
    try:
        skus = [
            _public_sku(sku)
            for sku in _load_autodl_skus()
            if normalize_provider(str(sku.get("provider") or "autodl")) == provider
            and (include_disabled or bool(sku.get("enabled", True)))
        ]
    except ValueError as exc:
        return {"message": f"gpu sku query failed: {exc}", "provider": provider, "skus": []}
    return {"message": "gpu sku query success", "provider": provider, "skus": skus}


def _find_autodl_sku(request: dict[str, Any]) -> dict[str, Any] | None:
    sku_id = _request_optional_string(request, "skuId")
    gpu_spec = _request_optional_string(request, "gpuSpec")
    if not sku_id and not gpu_spec:
        return None
    for sku in _load_autodl_skus():
        if not bool(sku.get("enabled", True)):
            continue
        if sku_id and str(sku.get("skuId") or "") == sku_id:
            return sku
        if gpu_spec and str(sku.get("gpuSpec") or "") == gpu_spec:
            return sku
    return None


def _apply_autodl_sku(request: dict[str, Any]) -> dict[str, Any]:
    if normalize_provider(_request_optional_string(request, "provider") or "aliyun") != "autodl":
        return request
    sku = _find_autodl_sku(request)
    if sku is None:
        if _request_optional_string(request, "skuId"):
            raise ValueError(f"unknown or disabled AutoDL skuId: {_request_string(request, 'skuId')}")
        return request
    enriched = dict(request)
    enriched["gpuSpec"] = sku.get("gpuSpec") or enriched.get("gpuSpec")
    enriched["gpuCount"] = sku.get("gpuCount") or enriched.get("gpuCount") or 1
    enriched["autodlGpuSpecUuid"] = sku.get("autodlGpuSpecUuid") or enriched.get("autodlGpuSpecUuid")
    enriched["autodlImageUuid"] = sku.get("autodlImageUuid") or enriched.get("autodlImageUuid")
    enriched["autodlDataCenters"] = sku.get("autodlDataCenters") or enriched.get("autodlDataCenters")
    enriched["cudaVFrom"] = sku.get("cudaVFrom") or enriched.get("cudaVFrom")
    cost = int(sku.get("costHourlyCents") or 0)
    enriched["hourlyPriceCents"] = sku.get("hourlyPriceCents") or _sale_price_from_cost(cost) or enriched.get("hourlyPriceCents")
    return enriched


def _request_mounts(request: dict[str, Any]) -> list[str] | None:
    mounts = request.get("mount")
    if mounts is None or mounts == "":
        return None
    if isinstance(mounts, list):
        return [str(item).strip() for item in mounts if str(item).strip()]
    return [str(mounts).strip()]


def _hourly_price_cents(provider: str, request: dict[str, Any]) -> int:
    explicit_price = _request_int(request, "hourlyPriceCents")
    if explicit_price is not None:
        return explicit_price
    gpu_spec = _request_optional_string(request, "gpuSpec") or _request_optional_string(request, "ecsSpec") or "default"
    return sql_get_hourly_price(provider, gpu_spec)


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


def _build_job_config(request: dict[str, Any], task_name: str, provider: str) -> dict[str, Any]:
    command = _request_optional_string(request, "command")
    return {
        "provider": provider,
        "region": _request_optional_string(request, "region") or "cn-hangzhou",
        "env_file": _request_optional_string(request, "envFile"),
        "dataset_path": _request_string(request, "datasetPath"),
        "epochs": _request_int(request, "epochs", required=command is None),
        "checkpoint_path": _request_string(request, "checkpointPath"),
        "checkpoint_frequency": _request_int(request, "checkpointFrequency", required=command is None),
        "command": command,
        "command_template": _request_optional_string(request, "commandTemplate"),
        "job_name": _request_optional_string(request, "jobName") or task_name,
        "workspace_id": _request_optional_string(request, "workspaceId"),
        "resource_id": _request_optional_string(request, "resourceId"),
        "image": _request_optional_string(request, "image"),
        "description": _request_optional_string(request, "description"),
        "ecs_spec": _request_optional_string(request, "ecsSpec"),
        "gpu_count": _request_int(request, "gpuCount", required=command is None),
        "gpu_type": _request_optional_string(request, "gpuType"),
        "cpu": _request_int(request, "cpu"),
        "memory": _request_int(request, "memory"),
        "max_running_minutes": _request_int(request, "maxRunningMinutes"),
        "mount": _request_mounts(request),
        "wait": _request_bool(request, "wait"),
        "timeout": _request_int(request, "timeout", default=86400),
        "interval": _request_int(request, "interval", default=30),
        "dry_run": _request_bool(request, "dryRun"),
        "workdir": _request_optional_string(request, "workdir"),
        "autodl_managed": _request_bool(request, "autodlManaged"),
        "autodl_instance_uuid": _request_optional_string(request, "autodlInstanceUuid"),
        "autodl_start_command": _request_optional_string(request, "autodlStartCommand"),
        "autodl_gpu_spec_uuid": _request_optional_string(request, "autodlGpuSpecUuid"),
        "autodl_image_uuid": _request_optional_string(request, "autodlImageUuid"),
        "autodl_data_centers": _request_optional_string(request, "autodlDataCenters"),
        "autodl_cuda_v_from": _request_int(request, "cudaVFrom"),
        "expand_system_disk_by_gb": _request_int(request, "expandSystemDiskGb"),
    }


def _create_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if sql_get_user_task(username, task_name) is not None:
        return "create task failed", sql_get_user_all_task(username)
    if request.get("command") and not request.get("workflow") and not _env_bool("EVO_TRAIN_ALLOW_RAW_COMMAND"):
        return (
            "raw command is disabled; submit workflow + params or set EVO_TRAIN_ALLOW_RAW_COMMAND=true",
            sql_get_user_all_task(username),
        )
    try:
        request = materialize_training_request(request)
        request = _apply_autodl_sku(request)
    except (ValueError, TypeError) as exc:
        return f"create task failed: {exc}", sql_get_user_all_task(username)

    provider = normalize_provider(_request_optional_string(request, "provider"))
    try:
        platform = get_platform(provider, region_id=_request_optional_string(request, "region") or "cn-hangzhou")
    except ValueError:
        return f"unsupported provider: {provider}", sql_get_user_all_task(username)

    hourly_price_cents = _hourly_price_cents(provider, request)
    if hourly_price_cents <= 0:
        return "invalid hourly price", sql_get_user_all_task(username)
    if not sql_freeze_user_balance(username, task_name, hourly_price_cents, "freeze first training hour"):
        wallet = sql_get_wallet(username)
        return (
            f"insufficient balance: need at least {hourly_price_cents} cents for 1 hour, "
            f"available {wallet['availableCents']} cents",
            sql_get_user_all_task(username),
        )

    try:
        job_config = _build_job_config(request, task_name, provider)
        if not job_config["command"] and not job_config["dataset_path"]:
            raise ValueError("missing required field: datasetPath")
        if not job_config["command"] and not job_config["checkpoint_path"]:
            raise ValueError("missing required field: checkpointPath")
        remote_job_id = platform.submit(job_config)
        if not remote_job_id and job_config["dry_run"]:
            sql_refund_frozen_balance(username, task_name, hourly_price_cents, "dry run")
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
            hourly_price_cents=hourly_price_cents,
            frozen_until=add_hours_text(utc_now_text(), 1),
            started_at=utc_now_text(),
            billing_status="frozen",
        )
        if not created:
            sql_refund_frozen_balance(username, task_name, hourly_price_cents, "task row create failed")
            return "create task failed", sql_get_user_all_task(username)
        return "create task success", sql_get_user_all_task(username)
    except (ValueError, SystemExit) as exc:
        sql_refund_frozen_balance(username, task_name, hourly_price_cents, "task create failed")
        return str(exc), sql_get_user_all_task(username)
    except Exception as exc:
        sql_refund_frozen_balance(username, task_name, hourly_price_cents, "task create failed")
        return f"create task failed: {exc}", sql_get_user_all_task(username)


def _refresh_platform_task(task: dict[str, str], env_file: str | None = None, region: str | None = None) -> None:
    provider = normalize_provider(task.get("provider"))
    remote_job_id = task.get("jobId", "")
    status = task.get("status", "")
    if _is_terminal_status(status):
        _settle_task_billing(task["username"], task["taskName"], status)
        return
    if not remote_job_id:
        return

    platform = get_platform(provider, region_id=region or "cn-hangzhou")
    metadata = platform.metadata(remote_job_id)
    sql_update_user_task(
        task["username"],  # type: ignore[index]
        task["taskName"],
        status=metadata.get("status") or status,
        last_error=metadata.get("last_error", ""),
    )
    refreshed_status = metadata.get("status") or status
    if _is_terminal_status(refreshed_status):
        _settle_task_billing(task["username"], task["taskName"], refreshed_status)
    elif refreshed_status == "Running":
        _ensure_next_billing_hour(task["username"], task["taskName"], env_file=env_file, region=region)


def _ensure_next_billing_hour(username: str, task_name: str, env_file: str | None = None, region: str | None = None) -> None:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return
    frozen_until = parse_timestamp(task.get("frozenUntil"))
    if frozen_until is None or frozen_until > parse_timestamp(utc_now_text()):
        return
    hourly_price_cents = int(task.get("hourlyPriceCents") or "0")
    if hourly_price_cents <= 0:
        return
    if sql_freeze_user_balance(username, task_name, hourly_price_cents, "freeze next training hour"):
        sql_update_user_task(username, task_name, frozen_until=add_hours_text(task["frozenUntil"], 1))
        return

    try:
        provider = normalize_provider(task.get("provider"))
        platform = get_platform(provider, region_id=region or "cn-hangzhou")
        platform.stop(task["jobId"])
    finally:
        sql_update_user_task(
            username,
            task_name,
            status="STOPPED",
            stopped_at=utc_now_text(),
            billing_status="stopped_insufficient_balance",
            last_error="insufficient balance for next training hour",
        )
        _settle_task_billing(username, task_name, "STOPPED")


def _settle_task_billing(username: str, task_name: str, terminal_status: str) -> None:
    task = sql_get_user_task(username, task_name)
    if task is None or task.get("billingStatus") == "settled":
        return
    hourly_price_cents = int(task.get("hourlyPriceCents") or "0")
    if hourly_price_cents <= 0:
        sql_update_user_task(username, task_name, billing_status="settled")
        return
    frozen_cents = sql_get_task_frozen_cents(username, task_name)
    charge_cents = frozen_cents or hourly_price_cents
    sql_charge_frozen_balance(username, task_name, charge_cents, f"settle terminal task: {terminal_status}")
    sql_update_user_task(
        username,
        task_name,
        stopped_at=utc_now_text(),
        actual_cost_cents=charge_cents,
        billing_status="settled",
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
            _refresh_platform_task(task, env_file=env_file, region=region)
        except (SystemExit, Exception) as exc:
            sql_update_user_task(
                username,
                task["taskName"],
                last_error=str(exc),
            )
    refreshed = sql_get_user_all_task(username)
    return refreshed


def scan_billing_tasks(env_file: str | None = None, region: str | None = None) -> dict[str, int]:
    """Reconcile billing and enforce wallet limits for all active paid tasks."""
    stats = {"checked": 0, "updated": 0, "errors": 0}
    for task in sql_get_billing_watch_tasks():
        stats["checked"] += 1
        username = task.get("username", "")
        task_name = task.get("taskName", "")
        try:
            before = sql_get_user_task(username, task_name)
            _refresh_platform_task(task, env_file=env_file, region=region or "cn-hangzhou")
            after = sql_get_user_task(username, task_name)
            if before != after:
                stats["updated"] += 1
        except (SystemExit, Exception) as exc:
            stats["errors"] += 1
            if username and task_name:
                sql_update_user_task(username, task_name, last_error=str(exc))
    return stats


def _stop_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return "stop task failed", sql_get_user_all_task(username)

    if task.get("jobId") and not _is_terminal_status(task.get("status", "")):
        try:
            provider = normalize_provider(task.get("provider"))
            region = _request_optional_string(request, "region") or "cn-hangzhou"
            platform = get_platform(provider, region_id=region)
            platform.stop(task["jobId"])
            _settle_task_billing(username, task_name, "STOPPED")
            sql_update_user_task(username, task_name, status="STOPPED", last_error="")
            return "stop task success", sql_get_user_all_task(username)
        except (SystemExit, Exception) as exc:
            sql_update_user_task(username, task_name, last_error=str(exc))
            return f"stop task failed: {exc}", sql_get_user_all_task(username)

    _settle_task_billing(username, task_name, "STOPPED")
    updated = sql_update_user_task(username, task_name, status="STOPPED")
    return ("stop task success" if updated else "stop task failed"), sql_get_user_all_task(username)


def _download_task_artifact(username: str, task_name: str, request: dict[str, Any]) -> dict[str, Any]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return _with_wallet(username, "download artifact failed", sql_get_user_all_task(username))
    artifact_path = (
        _request_optional_string(request, "artifactPath")
        or _request_optional_string(request, "downloadPath")
        or task.get("checkpointPath")
    )
    if not artifact_path:
        return _with_wallet(username, "download artifact failed: missing artifactPath", sql_get_user_all_task(username))
    try:
        provider = normalize_provider(task.get("provider"))
        region = _request_optional_string(request, "region") or "cn-hangzhou"
        offset = _request_int(request, "offset", default=0) or 0
        chunk_size = _request_int(request, "chunkSize", default=1024 * 1024) or 1024 * 1024
        platform = get_platform(provider, region_id=region)
        artifact = platform.download_artifact_chunk(
            task["jobId"],
            artifact_path,
            offset=offset,
            chunk_size=chunk_size,
        )
    except (SystemExit, Exception) as exc:
        sql_update_user_task(username, task_name, last_error=str(exc))
        response = _with_wallet(username, f"download artifact failed: {exc}", sql_get_user_all_task(username))
        response["artifact"] = {}
        return response
    response = _with_wallet(username, "download artifact success", sql_get_user_all_task(username))
    response["artifact"] = artifact
    return response


def _list_user_dataset_dirs(username: str) -> list[str]:
    """Return local dataset directory names in the shape used by upstream EVO-Train."""
    root_template = os.environ.get("EVO_TRAIN_USER_DATA_ROOT", "~/usrdata/{username}")
    root_path = os.path.expanduser(root_template.format(username=username))
    if not os.path.isdir(root_path):
        return []
    try:
        return sorted(
            name
            for name in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, name))
        )
    except OSError:
        return []


def _status_response(username: str, task_name: str, request: dict[str, Any]) -> dict[str, Any]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return _with_wallet(username, f"{task_name}: query status failed, job id does not exist.", sql_get_user_all_task(username))

    task["username"] = username
    try:
        _refresh_platform_task(task, env_file=_request_optional_string(request, "envFile"), region=_request_optional_string(request, "region"))
    except (SystemExit, Exception) as exc:
        sql_update_user_task(username, task_name, last_error=str(exc))
        return _with_wallet(username, f"{task_name}: query status failed: {exc}", sql_get_user_all_task(username))

    refreshed = sql_get_user_task(username, task_name) or task
    message = f"{task_name}: {refreshed.get('status', '')}".strip()
    return _with_wallet(username, message, sql_get_user_all_task(username))


def _download_directory_response(username: str, task_name: str) -> dict[str, Any]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        response = _with_wallet(username, f"{task_name}: query download directory failed, job id does not exist.", sql_get_user_all_task(username))
        response["downloadPath"] = ""
        response["downloadSize"] = 0
        response["checkpoints"] = []
        return response

    download_path = task.get("checkpointPath", "")
    response = _with_wallet(username, "query download directory success" if download_path else "query download directory failed", sql_get_user_all_task(username))
    response["downloadPath"] = download_path
    response["downloadSize"] = 0
    response["checkpoints"] = [download_path] if download_path else []
    return response


def _user_logs_response(username: str, task_name: str) -> dict[str, Any]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        response = _with_wallet(username, f"{task_name}: query logs failed, job id does not exist.", sql_get_user_all_task(username))
        response["logs"] = []
        return response

    last_error = task.get("error", "")
    response = _with_wallet(username, "query user logs success", sql_get_user_all_task(username))
    response["logs"] = last_error.splitlines() if last_error else []
    return response


def _wallet_response(username: str) -> dict[str, Any]:
    return {
        "message": "wallet query success",
        "wallet": sql_get_wallet(username),
        "tasks": sql_get_user_all_task(username),
    }


def _billing_records_response(username: str) -> dict[str, Any]:
    return {
        "message": "billing records query success",
        "wallet": sql_get_wallet(username),
        "billingRecords": sql_get_billing_records(username),
        "tasks": sql_get_user_all_task(username),
    }


def _admin_set_balance(username: str, request: dict[str, Any]) -> dict[str, Any]:
    amount = _request_int(request, "balanceCents", required=True)
    if amount is None or amount < 0:
        return {"message": "invalid balance", "wallet": sql_get_wallet(username), "tasks": sql_get_user_all_task(username)}
    sql_set_user_balance(username, amount)
    return {
        "message": "set balance success",
        "wallet": sql_get_wallet(username),
        "billingRecords": sql_get_billing_records(username),
        "tasks": sql_get_user_all_task(username),
    }


def _admin_set_price(request: dict[str, Any]) -> dict[str, Any]:
    provider = _request_optional_string(request, "provider") or "aliyun"
    gpu_spec = _request_optional_string(request, "gpuSpec") or _request_optional_string(request, "ecsSpec") or "default"
    amount = _request_int(request, "hourlyPriceCents", required=True)
    if amount is None or amount <= 0:
        return {"message": "invalid hourly price"}
    sql_set_gpu_price(provider, gpu_spec, amount)
    return {
        "message": "set gpu price success",
        "provider": provider,
        "gpuSpec": gpu_spec,
        "hourlyPriceCents": str(sql_get_hourly_price(provider, gpu_spec)),
    }


def _price_response(request: dict[str, Any]) -> dict[str, Any]:
    provider = _request_optional_string(request, "provider")
    return {
        "message": "price query success",
        "prices": sql_get_gpu_prices(provider),
    }


def _platform_balance_response(request: dict[str, Any]) -> dict[str, Any]:
    provider = normalize_provider(_request_optional_string(request, "provider") or "autodl")
    if provider != "autodl":
        return {"message": f"unsupported platform balance provider: {provider}"}
    try:
        balance = AutoDLApiClient().wallet_balance()
        minimum_assets = int(_request_int(request, "minimumAssets", default=0) or 0)
        if minimum_assets <= 0:
            minimum_assets = int(os.environ.get("AUTODL_MIN_ASSETS", "0"))
        assets = int(balance["assets"])
        return {
            "message": "platform balance query success",
            "provider": provider,
            "balance": balance,
            "minimumAssets": str(minimum_assets),
            "lowBalance": assets < minimum_assets if minimum_assets > 0 else False,
        }
    except (SystemExit, Exception) as exc:
        return {
            "message": f"platform balance query failed: {exc}",
            "provider": provider,
            "balance": {},
            "minimumAssets": str(_request_int(request, "minimumAssets", default=0) or 0),
            "lowBalance": True,
        }


def _ai_training_plan_response(username: str, request: dict[str, Any]) -> dict[str, Any]:
    try:
        plan = build_training_plan(request)
    except (ValueError, TypeError) as exc:
        return {
            "message": f"plan generation failed: {exc}",
            "wallet": sql_get_wallet(username),
            "plan": {},
            "tasks": sql_get_user_all_task(username),
        }
    return {
        "message": "plan generated",
        "wallet": sql_get_wallet(username),
        "plan": plan.to_response(),
        "tasks": sql_get_user_all_task(username),
    }


def _with_wallet(username: str, message: str, tasks: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "message": message,
        "wallet": sql_get_wallet(username),
        "tasks": tasks,
    }


def handle_request(text: str) -> dict[str, Any]:
    """Handle one complete JSON request and return a response dict."""
    try:
        request = json.loads(text)
    except json.JSONDecodeError:
        return {"message": "invalid json", "tasks": []}

    username = _request_string(request, "username")
    task_name = _request_string(request, "taskName")
    action = _request_string(request, "action")
    auth_error = _authorization_error(request, action)
    if auth_error is not None:
        return auth_error

    if action == "价格设置":
        return _admin_set_price(request)
    if action == "价格查询":
        return _price_response(request)
    if action == "平台余额查询":
        return _platform_balance_response(request)
    if action == "GPU规格查询":
        return _gpu_sku_response(request)

    if action in {"余额查询", "账单查询", "管理员充值"} and not username:
        return {"message": "invalid request", "tasks": []}

    if action == "余额查询":
        return _wallet_response(username)
    if action == "账单查询":
        return _billing_records_response(username)
    if action == "管理员充值":
        return _admin_set_balance(username, request)
    if action == "AI配置训练":
        return _ai_training_plan_response(username, request)

    if action == "任务同步":
        response = _with_wallet(username, "sync success", _refresh_user_tasks(username, request))
        response["datasetDir"] = _list_user_dataset_dirs(username)
        return response

    if not username or not task_name:
        return {"message": "invalid request", "tasks": sql_get_user_all_task(username)}

    if action == "开始训练":
        message, tasks = _create_task(username, task_name, request)
    elif action == "结束训练":
        message, tasks = _stop_task(username, task_name, request)
    elif action == "结果下载":
        return _download_task_artifact(username, task_name, request)
    elif action == "查询状态":
        return _status_response(username, task_name, request)
    elif action == "查询下载目录":
        return _download_directory_response(username, task_name)
    elif action == "请求用户日志":
        return _user_logs_response(username, task_name)
    elif action == "删除任务":
        message = "delete task success" if sql_delete_user_task(username, task_name) else "delete task failed"
        tasks = sql_get_user_all_task(username)
    else:
        message = "invalid action"
        tasks = sql_get_user_all_task(username)

    return _with_wallet(username, message, tasks)


def handle_request_text(text: str) -> str:
    """Handle one complete JSON request and return a JSON response string."""
    return json.dumps(handle_request(text), ensure_ascii=False)


def handle_download_task(event: Any) -> dict[str, Any]:
    """Handle download events from the dedicated download worker queue."""
    return handle_request(event.request_text)
