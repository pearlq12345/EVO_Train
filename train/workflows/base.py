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
    missing_fields: list[str]
    warnings: list[str]
    estimated_hours: int
    needs_confirmation: bool = True

    def to_response(self) -> dict[str, Any]:
        estimated_minimum_cost = self.hourly_price_cents * max(1, self.estimated_hours)
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
            "estimatedHours": str(self.estimated_hours),
            "estimatedMinimumCostCents": str(estimated_minimum_cost),
            "missingFields": self.missing_fields,
            "warnings": self.warnings,
            "summary": self.summary,
            "needsConfirmation": self.needs_confirmation or bool(self.warnings),
            "readyToStart": not self.missing_fields,
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


def estimate_hours(epochs: int, eval_episodes: int) -> int:
    if epochs > 80 or eval_episodes > 40:
        return 3
    if epochs > 30 or eval_episodes > 20:
        return 2
    return 1


def training_warnings(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    eval_episodes: int,
    gpu_spec: str,
) -> list[str]:
    warnings: list[str] = []
    if epochs <= 0:
        warnings.append("epochs must be greater than 0")
    elif epochs < 5:
        warnings.append("epochs is very small; this is only suitable for a smoke test")
    elif epochs > 100:
        warnings.append("epochs is high; confirm budget and expected runtime before starting")
    if batch_size <= 0:
        warnings.append("batchSize must be greater than 0")
    elif batch_size > 128:
        warnings.append("batchSize is high and may cause GPU OOM")
    if learning_rate <= 0:
        warnings.append("learningRate must be greater than 0")
    elif learning_rate > 0.01:
        warnings.append("learningRate is high and may make training unstable")
    if eval_episodes <= 0:
        warnings.append("evalEpisodes must be greater than 0")
    if gpu_spec == "default":
        warnings.append("gpuSpec is default; RoboClaw should confirm the target GPU tier")
    return warnings
