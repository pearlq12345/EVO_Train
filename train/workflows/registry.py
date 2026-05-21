from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable

from .base import WorkflowPlan
from . import custom_project, evf_libero, evf_metaworld, rlinf_vla
from .source_contract import normalize_training_sources, source_missing_fields, source_warnings


WorkflowBuilder = Callable[[dict[str, Any], dict[str, Any]], WorkflowPlan]

WORKFLOWS: dict[str, WorkflowBuilder] = {
    custom_project.WORKFLOW_NAME: custom_project.build_plan,
    evf_metaworld.WORKFLOW_NAME: evf_metaworld.build_plan,
    evf_libero.WORKFLOW_NAME: evf_libero.build_plan,
    rlinf_vla.WORKFLOW_NAME: rlinf_vla.build_plan,
    rlinf_vla.GENERIC_WORKFLOW_NAME: rlinf_vla.build_plan,
}


def parse_params(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        payload = value.strip()
        if not payload:
            return {}
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("params must be a JSON object")
        return parsed
    raise ValueError("params must be an object")


def infer_workflow(request: dict[str, Any], params: dict[str, Any]) -> str:
    workflow = str(request.get("workflow") or params.get("workflow") or "").strip()
    if workflow:
        return workflow
    message = str(request.get("message") or request.get("prompt") or "").lower()
    if (
        "rlinf" in message
        or "vla+rl" in message
        or "后训练" in message
        or "grpo" in message
        or "ppo" in message
        or "dexbotic" in message
        or "co-training" in message
        or "共训练" in message
        or "联合优化" in message
    ):
        return rlinf_vla.WORKFLOW_NAME
    if "repo" in message or "github" in message or "自定义" in message or "自己的项目" in message:
        return custom_project.WORKFLOW_NAME
    if "libero" in message:
        return evf_libero.WORKFLOW_NAME
    return evf_metaworld.WORKFLOW_NAME


def build_training_plan(request: dict[str, Any]) -> WorkflowPlan:
    params = normalize_training_sources(parse_params(request.get("params")))
    workflow = infer_workflow(request, params)
    builder = WORKFLOWS.get(workflow)
    if builder is None:
        raise ValueError(f"unsupported workflow: {workflow}")
    plan = builder(request, params)
    missing_fields = _merge_unique(plan.missing_fields, source_missing_fields(params))
    warnings = _merge_unique(plan.warnings, source_warnings(params))
    if missing_fields == plan.missing_fields and warnings == plan.warnings:
        return plan
    plan_params = dict(plan.params)
    for key in (
        "datasetSource",
        "modelSource",
        "sourceContract",
        "datasetSourceKind",
        "modelSourceKind",
        "datasetFormat",
        "checkpointFormat",
        "datasetAuthRef",
        "modelAuthRef",
        "resolvedModelSource",
    ):
        if key in params:
            plan_params[key] = params[key]
    return replace(plan, params=plan_params, missing_fields=missing_fields, warnings=warnings)


def materialize_training_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("command"):
        return request
    if not request.get("workflow"):
        return request
    plan = build_training_plan(request)
    if plan.missing_fields:
        raise ValueError(f"missing workflow fields: {', '.join(plan.missing_fields)}")
    materialized = dict(request)
    materialized["provider"] = plan.provider
    materialized["command"] = plan.command
    materialized["workdir"] = plan.workdir
    materialized["datasetPath"] = plan.dataset_path
    materialized["checkpointPath"] = plan.checkpoint_path
    materialized["checkpointFrequency"] = materialized.get("checkpointFrequency") or 1
    materialized["epochs"] = materialized.get("epochs") or plan.params.get("epochs") or 1
    materialized["gpuSpec"] = materialized.get("gpuSpec") or plan.gpu_spec
    materialized["hourlyPriceCents"] = materialized.get("hourlyPriceCents") or plan.hourly_price_cents
    if plan.provider == "autodl":
        materialized["autodlManaged"] = materialized.get("autodlManaged", True)
    return materialized


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged
