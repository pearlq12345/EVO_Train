#!/usr/bin/env python3
"""Training task business functions for the TCP server."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import threading
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
from train.workflows import build_training_plan, enrich_params_from_message, materialize_training_request, parse_params


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
    "下载损失",
    "AI配置训练",
    "查询状态",
    "查询下载目录",
    "请求用户日志",
    "GPU规格查询",
    "AutoDL镜像查询",
    "训练运行时匹配",
    "运行时匹配",
}

RESPONSE_SCHEMA_VERSION = "roboclaw.train.v1"
IDENTIFIER_PATTERN = re.compile(r"^[\w.@:-]{1,128}$", re.ASCII)
PATH_FRAGMENT_FORBIDDEN = {"/", "\\", "\x00"}


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


def _request_id(request: dict[str, Any]) -> str:
    return _request_string(request, "requestId") or _request_string(request, "traceId")


def _invalid_identifier(value: str, field_name: str) -> str | None:
    if not value:
        return None
    if any(fragment in value for fragment in PATH_FRAGMENT_FORBIDDEN) or ".." in value:
        return f"invalid {field_name}: path fragments are not allowed"
    if not IDENTIFIER_PATTERN.fullmatch(value):
        return f"invalid {field_name}: use 1-128 ASCII letters, digits, '_', '-', '.', ':', or '@'"
    return None


def _validate_roboclaw_request(request: dict[str, Any], action: str, username: str, task_name: str) -> dict[str, Any] | None:
    if not action:
        return _response("invalid request", error_code="INVALID_ACTION", request=request)
    if action not in ADMIN_ACTIONS and action not in USER_ACTIONS and action != "价格查询":
        return _response("invalid action", error_code="INVALID_ACTION", request=request)
    for field_name, value in (("username", username), ("taskName", task_name)):
        error = _invalid_identifier(value, field_name)
        if error:
            return _response(error, error_code="INVALID_IDENTIFIER", request=request, tasks=[])
    return None


def _response(
    message: str,
    *,
    ok: bool | None = None,
    error_code: str | None = None,
    request: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": ok if ok is not None else error_code is None,
        "errorCode": error_code,
        "message": message,
        "schemaVersion": RESPONSE_SCHEMA_VERSION,
    }
    if request is not None:
        request_id = _request_id(request)
        if request_id:
            response["requestId"] = request_id
    response.update(extra)
    return response


def _error_code_from_message(message: str) -> str | None:
    lowered = message.lower()
    if not lowered:
        return None
    if "unauthorized" in lowered:
        return "UNAUTHORIZED"
    if "invalid json" in lowered:
        return "INVALID_JSON"
    if "invalid request" in lowered:
        return "INVALID_REQUEST"
    if "invalid action" in lowered:
        return "INVALID_ACTION"
    if "unsupported provider" in lowered:
        return "UNSUPPORTED_PROVIDER"
    if "insufficient balance" in lowered:
        return "INSUFFICIENT_BALANCE"
    if "missing required field" in lowered or "missing workflow fields" in lowered:
        return "MISSING_REQUIRED_FIELD"
    if "disabled" in lowered:
        return "DISABLED"
    if "failed" in lowered or "error" in lowered:
        return "INTERNAL_ERROR"
    return None


def _finalize_response(response: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, Any]:
    message = str(response.get("message") or "")
    error_code = response.get("errorCode")
    if error_code is None:
        error_code = _error_code_from_message(message)
    finalized = dict(response)
    finalized.setdefault("schemaVersion", RESPONSE_SCHEMA_VERSION)
    finalized["errorCode"] = error_code
    finalized["ok"] = bool(finalized.get("ok", error_code is None))
    if request is not None and "requestId" not in finalized:
        request_id = _request_id(request)
        if request_id:
            finalized["requestId"] = request_id
    return finalized


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

    default_data_centers = os.environ.get("AUTODL_DATA_CENTER_LIST", "")
    for sku in skus:
        sku.setdefault("provider", "autodl")
        sku.setdefault("enabled", True)
        if default_data_centers and not sku.get("autodlDataCenters"):
            sku["autodlDataCenters"] = default_data_centers
        cost = int(sku.get("costHourlyCents") or 0)
        if cost > 0 and not sku.get("hourlyPriceCents"):
            sku["hourlyPriceCents"] = _sale_price_from_cost(cost)
    return skus


def _load_autodl_images() -> list[dict[str, Any]]:
    raw = os.environ.get("AUTODL_IMAGES_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid AUTODL_IMAGES_JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("AUTODL_IMAGES_JSON must be a JSON array")
        images = [dict(item) for item in parsed if isinstance(item, dict)]
    else:
        images = []
        default_image_uuid = os.environ.get("AUTODL_IMAGE_UUID", "")
        default_cuda_v_from = os.environ.get("AUTODL_CUDA_V_FROM", "")
        if default_image_uuid and default_cuda_v_from:
            images.append(
                {
                    "imageId": "default",
                    "displayName": os.environ.get("AUTODL_IMAGE_NAME", "Default AutoDL image"),
                    "autodlImageUuid": default_image_uuid,
                    "cudaVFrom": int(default_cuda_v_from),
                    "enabled": True,
                }
            )

    for image in images:
        image.setdefault("enabled", True)
    return images


def _public_sku(sku: dict[str, Any]) -> dict[str, Any]:
    cost = int(sku.get("costHourlyCents") or 0)
    sale = int(sku.get("hourlyPriceCents") or _sale_price_from_cost(cost) or 0)
    ready_to_start = _sku_ready_to_start(sku)
    response: dict[str, Any] = {
        "skuId": str(sku.get("skuId") or ""),
        "provider": str(sku.get("provider") or "autodl"),
        "displayName": str(sku.get("displayName") or sku.get("gpuSpec") or sku.get("skuId") or ""),
        "gpuSpec": str(sku.get("gpuSpec") or sku.get("skuId") or ""),
        "gpuCount": str(sku.get("gpuCount") or 1),
        "hourlyPriceCents": str(sale),
        "serviceFeeRate": str(_service_fee_rate()),
        "enabled": bool(sku.get("enabled", True)),
        "readyToStart": ready_to_start,
    }
    for source, target in (
        ("gpuMemoryGb", "gpuMemoryGb"),
        ("diskGb", "diskGb"),
        ("region", "region"),
    ):
        if sku.get(source) not in (None, ""):
            response[target] = str(sku.get(source))
    for key in (
        "supportedBackends",
        "supportedModels",
        "supportedBenchmarks",
        "supportedAlgorithms",
        "supportedTrainingModes",
        "capabilities",
        "simFrameworks",
        "datasetFormats",
        "gpuFamilies",
    ):
        if key in sku:
            response[key] = _string_list(sku.get(key))
    if cost > 0:
        response["costHourlyCents"] = str(cost)
    return response


def _sku_ready_to_start(sku: dict[str, Any]) -> bool:
    if not bool(sku.get("enabled", True)):
        return False
    if sku.get("autodlGpuSpecUuid") in (None, ""):
        return False
    cost = int(sku.get("costHourlyCents") or 0)
    sale = int(sku.get("hourlyPriceCents") or _sale_price_from_cost(cost) or 0)
    return sale > 0


def _gpu_sku_response(request: dict[str, Any]) -> dict[str, Any]:
    provider = normalize_provider(_request_optional_string(request, "provider") or "autodl")
    include_disabled = _request_bool(request, "includeDisabled")
    include_incomplete = _request_bool(request, "includeIncomplete")
    try:
        skus = [
            _public_sku(sku)
            for sku in _load_autodl_skus()
            if normalize_provider(str(sku.get("provider") or "autodl")) == provider
            and (include_disabled or bool(sku.get("enabled", True)))
            and (include_incomplete or _sku_ready_to_start(sku))
        ]
    except ValueError as exc:
        return {"message": f"gpu sku query failed: {exc}", "provider": provider, "skus": []}
    return {"message": "gpu sku query success", "provider": provider, "skus": skus}


def _public_image(image: dict[str, Any]) -> dict[str, Any]:
    response: dict[str, Any] = {
        "imageId": str(image.get("imageId") or ""),
        "displayName": str(image.get("displayName") or image.get("imageId") or ""),
        "autodlImageUuid": str(image.get("autodlImageUuid") or ""),
        "cudaVFrom": str(image.get("cudaVFrom") or ""),
        "enabled": bool(image.get("enabled", True)),
        "readyToStart": _image_ready_to_start(image),
    }
    for source, target in (
        ("python", "python"),
        ("torch", "torch"),
        ("minDiskGb", "minDiskGb"),
        ("healthcheck", "healthcheck"),
        ("setupProfile", "setupProfile"),
        ("status", "status"),
    ):
        if image.get(source) not in (None, ""):
            response[target] = str(image.get(source))
    for key in (
        "supportedBackends",
        "supportedModels",
        "supportedBenchmarks",
        "supportedAlgorithms",
        "supportedTrainingModes",
        "capabilities",
        "simFrameworks",
        "datasetFormats",
        "gpuFamilies",
        "frameworks",
    ):
        if key in image:
            response[key] = _string_list(image.get(key))
    return response


def _image_ready_to_start(image: dict[str, Any]) -> bool:
    if not bool(image.get("enabled", True)):
        return False
    return bool(image.get("autodlImageUuid")) and image.get("cudaVFrom") not in (None, "")


def _autodl_image_response(request: dict[str, Any]) -> dict[str, Any]:
    include_disabled = _request_bool(request, "includeDisabled")
    include_incomplete = _request_bool(request, "includeIncomplete")
    try:
        images = [
            _public_image(image)
            for image in _load_autodl_images()
            if (include_disabled or bool(image.get("enabled", True)))
            and (include_incomplete or _image_ready_to_start(image))
        ]
    except ValueError as exc:
        return {"message": f"autodl image query failed: {exc}", "images": []}
    return {"message": "autodl image query success", "images": images}


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _requirement_value(request: dict[str, Any], key: str) -> str:
    params = request.get("params")
    if isinstance(params, dict) and params.get(key) not in (None, ""):
        return str(params.get(key) or "").strip().lower()
    return _request_string(request, key).lower()


def _requirement_int(request: dict[str, Any], key: str) -> int:
    params = request.get("params")
    value: Any = None
    if isinstance(params, dict):
        value = params.get(key)
    if value in (None, ""):
        value = request.get(key)
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field: {key}") from exc


def _requirement_list(request: dict[str, Any], key: str) -> list[str]:
    params = request.get("params")
    if isinstance(params, dict) and params.get(key) not in (None, ""):
        return _string_list(params.get(key))
    return _string_list(request.get(key))


def _supports(value: str, supported: list[str]) -> bool:
    return not value or not supported or value in supported


def _runtime_match_response(request: dict[str, Any]) -> dict[str, Any]:
    provider = normalize_provider(_request_optional_string(request, "provider") or "autodl")
    if provider != "autodl":
        return {"message": "runtime match failed: unsupported provider", "provider": provider, "matches": []}
    req = {
        "backendKind": _requirement_value(request, "backendKind") or _requirement_value(request, "backend"),
        "modelFamily": _requirement_value(request, "modelFamily"),
        "benchmark": _requirement_value(request, "benchmark") or _requirement_value(request, "envType"),
        "algorithm": _requirement_value(request, "algorithm"),
        "trainingMode": _requirement_value(request, "trainingMode"),
        "minGpuMemoryGb": _requirement_int(request, "minGpuMemoryGb"),
        "minDiskGb": _requirement_int(request, "minDiskGb"),
        "requiredCapabilities": _requirement_list(request, "requiredCapabilities"),
    }
    requested_sku_id = _request_optional_string(request, "skuId")
    requested_image_id = _request_optional_string(request, "imageId")
    try:
        skus = [
            _public_sku(sku)
            for sku in _load_autodl_skus()
            if bool(sku.get("enabled", True))
            and (not requested_sku_id or str(sku.get("skuId") or "") == requested_sku_id)
        ]
        images = [
            _public_image(image)
            for image in _load_autodl_images()
            if bool(image.get("enabled", True))
            and (not requested_image_id or str(image.get("imageId") or "") == requested_image_id)
        ]
    except ValueError as exc:
        return {"message": f"runtime match failed: {exc}", "provider": provider, "requirements": req, "matches": []}
    matches = []
    for sku in skus:
        for image in images:
            result = _score_runtime_candidate(req, sku, image)
            matches.append(result)
    matches.sort(key=lambda item: (item["compatible"], item["score"]), reverse=True)
    return {
        "message": "runtime match success",
        "provider": provider,
        "requirements": req,
        "matches": matches,
        "readyToStart": any(item["compatible"] for item in matches),
    }


def _score_runtime_candidate(req: dict[str, Any], sku: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    blocking: list[str] = []
    risks: list[str] = []
    score = 100

    if not sku.get("readyToStart"):
        blocking.append("sku is incomplete or disabled")
        score -= 40
    if not image.get("readyToStart"):
        blocking.append("image is incomplete or disabled")
        score -= 40
    for field, label, sku_key, image_key in (
        ("backendKind", "backend", "supportedBackends", "supportedBackends"),
        ("modelFamily", "model", "supportedModels", "supportedModels"),
        ("benchmark", "benchmark", "supportedBenchmarks", "supportedBenchmarks"),
        ("algorithm", "algorithm", "supportedAlgorithms", "supportedAlgorithms"),
        ("trainingMode", "training mode", "supportedTrainingModes", "supportedTrainingModes"),
    ):
        value = str(req.get(field) or "")
        sku_supported = _string_list(sku.get(sku_key))
        image_supported = _string_list(image.get(image_key))
        if not _supports(value, sku_supported):
            blocking.append(f"sku does not support {label}: {value}")
            score -= 20
        elif value:
            reasons.append(f"sku supports {label}: {value}")
        if not _supports(value, image_supported):
            blocking.append(f"image does not support {label}: {value}")
            score -= 20
        elif value:
            reasons.append(f"image supports {label}: {value}")
    min_gpu = int(req.get("minGpuMemoryGb") or 0)
    gpu_memory = int(str(sku.get("gpuMemoryGb") or "0") or 0)
    if min_gpu > 0 and gpu_memory > 0 and gpu_memory < min_gpu:
        blocking.append(f"gpu memory too small: {gpu_memory}GB < {min_gpu}GB")
        score -= 25
    elif min_gpu > 0 and gpu_memory > 0:
        reasons.append(f"gpu memory ok: {gpu_memory}GB")
    elif min_gpu > 0:
        risks.append("gpu memory is not declared for sku")
        score -= 5
    min_disk = int(req.get("minDiskGb") or 0)
    image_disk = int(str(image.get("minDiskGb") or "0") or 0)
    if min_disk > 0 and image_disk > 0 and image_disk > min_disk:
        risks.append(f"image recommends at least {image_disk}GB disk")
        score -= 5
    required_capabilities = [str(item) for item in req.get("requiredCapabilities") or [] if str(item)]
    if required_capabilities:
        sku_capabilities = _candidate_capabilities(sku)
        image_capabilities = _candidate_capabilities(image)
        runtime_capabilities = sku_capabilities | image_capabilities
        missing = [item for item in required_capabilities if item not in runtime_capabilities]
        if missing:
            blocking.append(f"missing capabilities: {', '.join(missing)}")
            score -= min(40, 10 * len(missing))
        else:
            reasons.append(f"capabilities ok: {', '.join(required_capabilities)}")
    if not reasons:
        reasons.append("candidate has no declared incompatibility")
    return {
        "sku": sku,
        "image": image,
        "compatible": not blocking,
        "score": max(0, min(100, score)),
        "reasons": reasons,
        "blockingReasons": blocking,
        "risks": risks,
    }


def _candidate_capabilities(candidate: dict[str, Any]) -> set[str]:
    keys = (
        "capabilities",
        "frameworks",
        "simFrameworks",
        "datasetFormats",
        "supportedBackends",
        "supportedModels",
        "supportedBenchmarks",
        "supportedAlgorithms",
        "supportedTrainingModes",
        "gpuFamilies",
    )
    values: set[str] = set()
    for key in keys:
        values.update(_string_list(candidate.get(key)))
    return values


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


def _find_autodl_image(request: dict[str, Any]) -> dict[str, Any] | None:
    image_id = _request_optional_string(request, "imageId")
    if not image_id:
        return None
    for image in _load_autodl_images():
        if not bool(image.get("enabled", True)):
            continue
        if str(image.get("imageId") or "") == image_id:
            return image
    return None


def _apply_autodl_sku(request: dict[str, Any]) -> dict[str, Any]:
    if normalize_provider(_request_optional_string(request, "provider") or "aliyun") != "autodl":
        return request
    sku = _find_autodl_sku(request)
    if sku is None:
        if _request_optional_string(request, "skuId"):
            raise ValueError(f"unknown or disabled AutoDL skuId: {_request_string(request, 'skuId')}")
        if _request_optional_string(request, "imageId"):
            return _apply_autodl_image(request)
        return request
    enriched = dict(request)
    enriched["gpuSpec"] = sku.get("gpuSpec") or enriched.get("gpuSpec")
    enriched["gpuCount"] = sku.get("gpuCount") or enriched.get("gpuCount") or 1
    enriched["autodlGpuSpecUuid"] = sku.get("autodlGpuSpecUuid") or enriched.get("autodlGpuSpecUuid")
    enriched["autodlDataCenters"] = sku.get("autodlDataCenters") or enriched.get("autodlDataCenters")
    enriched["autodlImageUuid"] = enriched.get("autodlImageUuid") or sku.get("autodlImageUuid")
    enriched["cudaVFrom"] = enriched.get("cudaVFrom") or sku.get("cudaVFrom")
    cost = int(sku.get("costHourlyCents") or 0)
    enriched["hourlyPriceCents"] = sku.get("hourlyPriceCents") or _sale_price_from_cost(cost) or enriched.get("hourlyPriceCents")
    if sku.get("requiresImage") is not False and not enriched.get("imageId") and not enriched.get("autodlImageUuid"):
        raise ValueError("missing required field: imageId")
    return _apply_autodl_image(enriched)


def _apply_autodl_image(request: dict[str, Any]) -> dict[str, Any]:
    if normalize_provider(_request_optional_string(request, "provider") or "aliyun") != "autodl":
        return request
    image = _find_autodl_image(request)
    if image is None:
        if _request_optional_string(request, "imageId"):
            raise ValueError(f"unknown or disabled AutoDL imageId: {_request_string(request, 'imageId')}")
        enriched = dict(request)
    else:
        enriched = dict(request)
        enriched["autodlImageUuid"] = image.get("autodlImageUuid") or enriched.get("autodlImageUuid")
        enriched["cudaVFrom"] = image.get("cudaVFrom") or enriched.get("cudaVFrom")
    missing = [
        field
        for field in ("autodlGpuSpecUuid", "autodlImageUuid", "cudaVFrom")
        if enriched.get(field) in (None, "")
    ]
    if missing and not _request_optional_string(enriched, "autodlInstanceUuid"):
        raise ValueError(f"AutoDL sku is missing required fields: {', '.join(missing)}")
    if int(enriched.get("hourlyPriceCents") or 0) <= 0:
        raise ValueError("AutoDL sku is missing hourly price")
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


def _submit_task_background(
    *,
    username: str,
    task_name: str,
    provider: str,
    region: str,
    job_config: dict[str, Any],
    hourly_price_cents: int,
) -> None:
    try:
        platform = get_platform(provider, region_id=region)
        remote_job_id = platform.submit(job_config)
        if not remote_job_id:
            raise RuntimeError("missing remote job id")
        sql_update_user_task(
            username,
            task_name,
            status="Submitted",
            remote_job_id=remote_job_id,
            last_error="",
        )
    except BaseException as exc:
        sql_refund_frozen_balance(username, task_name, hourly_price_cents, "background submit failed")
        sql_update_user_task(
            username,
            task_name,
            status="SubmitFailed",
            billing_status="submit_failed",
            last_error=str(exc),
            stopped_at=utc_now_text(),
        )


def _start_background_submit(
    *,
    username: str,
    task_name: str,
    provider: str,
    region: str,
    job_config: dict[str, Any],
    hourly_price_cents: int,
) -> None:
    thread = threading.Thread(
        target=_submit_task_background,
        kwargs={
            "username": username,
            "task_name": task_name,
            "provider": provider,
            "region": region,
            "job_config": job_config,
            "hourly_price_cents": hourly_price_cents,
        },
        name=f"evo-submit-{username}-{task_name}",
        daemon=True,
    )
    thread.start()


def _create_task(username: str, task_name: str, request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if sql_get_user_task(username, task_name) is not None:
        return "create task failed", sql_get_user_all_task(username)
    if request.get("command") and not request.get("workflow") and not _env_bool("EVO_TRAIN_ALLOW_RAW_COMMAND"):
        return (
            "raw command is disabled; submit workflow + params or set EVO_TRAIN_ALLOW_RAW_COMMAND=true",
            sql_get_user_all_task(username),
        )
    try:
        request = _enrich_workflow_params(request)
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
        if not _request_bool(request, "waitForSubmit") and not _request_bool(request, "syncSubmit"):
            if job_config["dry_run"]:
                sql_refund_frozen_balance(username, task_name, hourly_price_cents, "dry run")
                return "dry run only", sql_get_user_all_task(username)
            created = sql_add_user_task(
                username,
                task_name,
                status="Submitting",
                provider=provider,
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
            _start_background_submit(
                username=username,
                task_name=task_name,
                provider=provider,
                region=_request_optional_string(request, "region") or "cn-hangzhou",
                job_config=job_config,
                hourly_price_cents=hourly_price_cents,
            )
            return "training accepted", sql_get_user_all_task(username)

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


def _download_loss_artifact(username: str, task_name: str, request: dict[str, Any]) -> dict[str, Any]:
    task = sql_get_user_task(username, task_name)
    if task is None:
        return _with_wallet(username, "download loss failed", sql_get_user_all_task(username))
    loss_path = (
        _request_optional_string(request, "lossPath")
        or _request_optional_string(request, "artifactPath")
        or _request_optional_string(request, "downloadPath")
    )
    if not loss_path and task.get("checkpointPath"):
        loss_path = f"{task['checkpointPath'].rstrip('/')}/loss/loss.txt"
    if not loss_path:
        return _with_wallet(username, "download loss failed: missing lossPath", sql_get_user_all_task(username))
    artifact_request = dict(request)
    artifact_request["artifactPath"] = loss_path
    response = _download_task_artifact(username, task_name, artifact_request)
    response["message"] = response["message"].replace("download artifact", "download loss")
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
        request = _enrich_workflow_params(request)
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


def _enrich_workflow_params(request: dict[str, Any]) -> dict[str, Any]:
    if not request.get("workflow") and not request.get("message") and not request.get("prompt"):
        return request
    enriched = dict(request)
    enriched["params"] = enrich_params_from_message(request, parse_params(request.get("params")))
    return enriched


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
        return _finalize_response({"message": "invalid json", "tasks": []})
    if not isinstance(request, dict):
        return _finalize_response({"message": "invalid request", "tasks": []})

    username = _request_string(request, "username")
    task_name = _request_string(request, "taskName")
    action = _request_string(request, "action")
    validation_error = _validate_roboclaw_request(request, action, username, task_name)
    if validation_error is not None:
        return validation_error
    auth_error = _authorization_error(request, action)
    if auth_error is not None:
        return _finalize_response(auth_error, request)

    if action == "价格设置":
        return _finalize_response(_admin_set_price(request), request)
    if action == "价格查询":
        return _finalize_response(_price_response(request), request)
    if action == "平台余额查询":
        return _finalize_response(_platform_balance_response(request), request)
    if action == "GPU规格查询":
        return _finalize_response(_gpu_sku_response(request), request)
    if action == "AutoDL镜像查询":
        return _finalize_response(_autodl_image_response(request), request)
    if action in {"训练运行时匹配", "运行时匹配"}:
        return _finalize_response(_runtime_match_response(request), request)

    if action in {"余额查询", "账单查询", "管理员充值"} and not username:
        return _finalize_response({"message": "invalid request", "tasks": []}, request)

    if action == "余额查询":
        return _finalize_response(_wallet_response(username), request)
    if action == "账单查询":
        return _finalize_response(_billing_records_response(username), request)
    if action == "管理员充值":
        return _finalize_response(_admin_set_balance(username, request), request)
    if action == "AI配置训练":
        return _finalize_response(_ai_training_plan_response(username, request), request)

    if action == "任务同步":
        response = _with_wallet(username, "sync success", _refresh_user_tasks(username, request))
        response["datasetDir"] = _list_user_dataset_dirs(username)
        return _finalize_response(response, request)

    if not username or not task_name:
        return _finalize_response({"message": "invalid request", "tasks": sql_get_user_all_task(username)}, request)

    if action == "开始训练":
        message, tasks = _create_task(username, task_name, request)
    elif action == "结束训练":
        message, tasks = _stop_task(username, task_name, request)
    elif action == "结果下载":
        return _finalize_response(_download_task_artifact(username, task_name, request), request)
    elif action == "下载损失":
        return _finalize_response(_download_loss_artifact(username, task_name, request), request)
    elif action == "查询状态":
        return _finalize_response(_status_response(username, task_name, request), request)
    elif action == "查询下载目录":
        return _finalize_response(_download_directory_response(username, task_name), request)
    elif action == "请求用户日志":
        return _finalize_response(_user_logs_response(username, task_name), request)
    elif action == "删除任务":
        message = "delete task success" if sql_delete_user_task(username, task_name) else "delete task failed"
        tasks = sql_get_user_all_task(username)
    else:
        message = "invalid action"
        tasks = sql_get_user_all_task(username)

    return _finalize_response(_with_wallet(username, message, tasks), request)


def handle_request_text(text: str) -> str:
    """Handle one complete JSON request and return a JSON response string."""
    return json.dumps(handle_request(text), ensure_ascii=False)


def handle_download_task(event: Any) -> dict[str, Any]:
    """Handle download events from the dedicated download worker queue."""
    return handle_request(event.request_text)
