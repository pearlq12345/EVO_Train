from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowPlan:
    workflow: str
    provider: str
    params: dict[str, Any]
    command: str
    workdir: str
    checkpoint_path: str
    dataset_path: str
    gpu_spec: str
    hourly_price_cents: int
    summary: str
    needs_confirmation: bool = True

    def to_response(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "provider": self.provider,
            "params": self.params,
            "command": self.command,
            "workdir": self.workdir,
            "checkpointPath": self.checkpoint_path,
            "datasetPath": self.dataset_path,
            "gpuSpec": self.gpu_spec,
            "hourlyPriceCents": str(self.hourly_price_cents),
            "summary": self.summary,
            "needsConfirmation": self.needs_confirmation,
        }


def param_string(params: dict[str, Any], key: str, default: str = "") -> str:
    value = params.get(key)
    if value is None or value == "":
        return default
    return str(value).strip()


def param_int(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key)
    if value is None or value == "":
        return default
    return int(value)


def param_float(params: dict[str, Any], key: str, default: float) -> float:
    value = params.get(key)
    if value is None or value == "":
        return default
    return float(value)


def param_bool(params: dict[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def quote_args(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)
