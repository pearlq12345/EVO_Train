from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import os
import shlex
from typing import Any


class TrainPlatform(ABC):
    @abstractmethod
    def submit(self, job_config: dict[str, Any]) -> str:
        """Submit one training job and return its provider job id."""

    @abstractmethod
    def metadata(self, job_id: str) -> dict[str, str]:
        """Return status and error metadata for one provider job."""

    @abstractmethod
    def stop(self, job_id: str) -> None:
        """Stop one provider job."""

    def download_artifact_chunk(
        self,
        job_id: str,
        artifact_path: str,
        *,
        offset: int = 0,
        chunk_size: int = 1024 * 1024,
    ) -> dict[str, str | int | bool]:
        """Return one base64-encoded result chunk for a provider job."""
        raise NotImplementedError("artifact download is not supported by this provider")


def first_value(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def require_value(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing required value: {name}")
    return value


def require_int(value: Any, name: str) -> int:
    if value is None or value == "":
        raise SystemExit(f"Missing required value: {name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid integer value for {name}: {value}") from exc


def optional_string(job_config: Mapping[str, Any], key: str) -> str | None:
    value = job_config.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def config_bool(job_config: Mapping[str, Any], key: str) -> bool:
    value = job_config.get(key)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_train_command(job_config: Mapping[str, Any]) -> str:
    command = optional_string(job_config, "command")
    if command:
        return command

    template = first_value(
        optional_string(job_config, "command_template"),
        os.environ.get("DLC_TRAIN_COMMAND_TEMPLATE"),
        "python train.py --dataset-path {dataset_path} --epochs {epochs} "
        "--checkpoint-path {checkpoint_path} --checkpoint-frequency {checkpoint_frequency}",
    )
    dataset_path = require_value(optional_string(job_config, "dataset_path"), "dataset_path")
    checkpoint_path = require_value(optional_string(job_config, "checkpoint_path"), "checkpoint_path")
    epochs = require_int(job_config.get("epochs"), "epochs")
    checkpoint_frequency = require_int(job_config.get("checkpoint_frequency"), "checkpoint_frequency")
    gpu_count = require_int(job_config.get("gpu_count"), "gpu_count")
    values = {
        "dataset_path": shlex.quote(dataset_path),
        "epochs": epochs,
        "checkpoint_path": shlex.quote(checkpoint_path),
        "checkpoint_frequency": checkpoint_frequency,
        "gpu_count": gpu_count,
    }
    return template.format(**values)
