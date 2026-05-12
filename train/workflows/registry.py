from __future__ import annotations

import json
import re
from typing import Any, Callable

from .base import WorkflowPlan
from . import evf_libero, evf_metaworld


WorkflowBuilder = Callable[[dict[str, Any], dict[str, Any]], WorkflowPlan]

WORKFLOWS: dict[str, WorkflowBuilder] = {
    evf_metaworld.WORKFLOW_NAME: evf_metaworld.build_plan,
    evf_libero.WORKFLOW_NAME: evf_libero.build_plan,
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
    if "libero" in message:
        return evf_libero.WORKFLOW_NAME
    return evf_metaworld.WORKFLOW_NAME


def enrich_params_from_message(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(params)
    message = str(request.get("message") or request.get("prompt") or "")
    lowered = message.lower()
    epochs_match = re.search(r"(\d+)\s*(?:个)?\s*(?:epoch|epochs|轮)", lowered)
    if epochs_match and "epochs" not in enriched:
        enriched["epochs"] = int(epochs_match.group(1))
    eval_match = re.search(r"(\d+)\s*(?:个)?\s*(?:eval|评估|episode|episodes)", lowered)
    if eval_match and "evalEpisodes" not in enriched:
        enriched["evalEpisodes"] = int(eval_match.group(1))
    if "pick-place" in lowered and "envName" not in enriched:
        enriched["envName"] = "pick-place-v2"
    if "libero_object" in lowered and "suite" not in enriched:
        enriched["suite"] = "libero_object_task"
    elif "libero" in lowered and "suite" not in enriched:
        enriched["suite"] = "libero_object_task"
    return enriched


def build_training_plan(request: dict[str, Any]) -> WorkflowPlan:
    params = enrich_params_from_message(request, parse_params(request.get("params")))
    workflow = infer_workflow(request, params)
    builder = WORKFLOWS.get(workflow)
    if builder is None:
        raise ValueError(f"unsupported workflow: {workflow}")
    return builder(request, params)


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
    materialized["epochs"] = materialized.get("epochs") or plan.params.get("epochs")
    materialized["gpuSpec"] = materialized.get("gpuSpec") or plan.gpu_spec
    materialized["hourlyPriceCents"] = materialized.get("hourlyPriceCents") or plan.hourly_price_cents
    if plan.provider == "autodl":
        materialized["autodlManaged"] = materialized.get("autodlManaged", True)
    return materialized
