from __future__ import annotations

from collections.abc import Mapping
import json
import os
import sys
from typing import Any

from .base import (
    TrainPlatform,
    build_train_command,
    config_bool,
    first_value,
    optional_int,
    optional_string,
    require_int,
    require_value,
)


DLCClient: Any = None
dlc_models: Any = None
openapi_models: Any = None

DONE_STATUSES = {"Succeeded", "Succeed", "SUCCESS", "SUCCEEDED"}
FAILED_STATUSES = {"Failed", "FAILED", "Stopped", "STOPPED", "Deleted", "DELETED"}
DEFAULT_JOB_TYPE = "PyTorch"
DEFAULT_JOB_ROLE = "Worker"
DEFAULT_ACCESSIBILITY = "PRIVATE"
DEFAULT_POD_COUNT = 1


def log(message: str) -> None:
    print(f"[start_train] {message}", flush=True)


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_dlc_sdk() -> None:
    global DLCClient, dlc_models, openapi_models
    if DLCClient and dlc_models and openapi_models:
        return
    try:
        from alibabacloud_pai_dlc20201203.client import Client
        from alibabacloud_pai_dlc20201203 import models
        from alibabacloud_tea_openapi import models as tea_openapi_models
    except ImportError as exc:
        print(
            "Missing Alibaba Cloud SDK dependencies. Install them with:\n"
            "  python3 -m pip install --user "
            "alibabacloud_pai-dlc20201203 alibabacloud_tea_openapi paramiko",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    DLCClient = Client
    dlc_models = models
    openapi_models = tea_openapi_models


def create_client(region_id: str) -> Any:
    load_dlc_sdk()
    access_key_id = require_value(
        env_first("ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIBABACLOUD_ACCESS_KEY_ID", "OSS_ACCESS_KEY_ID"),
        "ALIBABA_CLOUD_ACCESS_KEY_ID or OSS_ACCESS_KEY_ID",
    )
    access_key_secret = require_value(
        env_first("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "ALIBABACLOUD_ACCESS_KEY_SECRET", "OSS_ACCESS_KEY_SECRET"),
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET or OSS_ACCESS_KEY_SECRET",
    )
    endpoint = os.environ.get("PAI_DLC_ENDPOINT", f"pai-dlc.{region_id}.aliyuncs.com")
    config = openapi_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        region_id=region_id,
        endpoint=endpoint,
    )
    return DLCClient(config)


def to_plain(value: Any) -> Any:
    if hasattr(value, "to_map"):
        return value.to_map()
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return value


def print_json(value: Any) -> None:
    print(json.dumps(to_plain(value), ensure_ascii=False, indent=2, sort_keys=True))


def _mount_specs(job_config: Mapping[str, Any]) -> list[str]:
    mounts = job_config.get("mount")
    if mounts is None or mounts == "":
        return []
    if isinstance(mounts, str):
        mount_items = [mounts]
    else:
        mount_items = [str(item) for item in mounts]
    return [item.strip() for item in mount_items if item.strip()]


def make_data_sources(job_config: Mapping[str, Any]) -> list[Any]:
    load_dlc_sdk()
    data_sources: list[Any] = []
    for spec in _mount_specs(job_config):
        try:
            uri, rest = spec.split("=", 1)
        except ValueError as exc:
            raise SystemExit(f"Invalid --mount value: {spec}. Expected URI=MOUNT_PATH[:RO|RW]") from exc
        mount_path = rest
        mount_access = "RW"
        if rest.endswith(":RO") or rest.endswith(":RW"):
            mount_path, mount_access = rest.rsplit(":", 1)
        data_sources.append(
            dlc_models.CreateJobRequestDataSources(
                uri=uri,
                mount_path=mount_path,
                mount_access=mount_access,
            )
        )
    return data_sources


def build_job_request(job_config: Mapping[str, Any]) -> Any:
    load_dlc_sdk()
    workspace_id = require_value(
        first_value(optional_string(job_config, "workspace_id"), os.environ.get("PAI_WORKSPACE_ID")),
        "--workspace-id or PAI_WORKSPACE_ID",
    )
    image = require_value(
        first_value(optional_string(job_config, "image"), os.environ.get("PAI_DLC_IMAGE")),
        "--image or PAI_DLC_IMAGE",
    )
    ecs_spec = require_value(
        first_value(optional_string(job_config, "ecs_spec"), os.environ.get("PAI_ECS_SPEC")),
        "--ecs-spec or PAI_ECS_SPEC",
    )
    resource_id = first_value(optional_string(job_config, "resource_id"), os.environ.get("PAI_RESOURCE_ID"))
    gpu_count = require_int(job_config.get("gpu_count"), "gpu_count")
    dataset_path = require_value(optional_string(job_config, "dataset_path"), "dataset_path")
    checkpoint_path = require_value(optional_string(job_config, "checkpoint_path"), "checkpoint_path")
    epochs = require_int(job_config.get("epochs"), "epochs")
    checkpoint_frequency = require_int(job_config.get("checkpoint_frequency"), "checkpoint_frequency")

    resource_config = dlc_models.ResourceConfig(
        gpu=gpu_count,
        cpu=optional_int(job_config, "cpu"),
        memory=optional_int(job_config, "memory"),
    )
    gpu_type = optional_string(job_config, "gpu_type")
    if gpu_type:
        resource_config.gputype = gpu_type

    job_spec = dlc_models.JobSpec(
        type=DEFAULT_JOB_ROLE,
        image=image,
        pod_count=DEFAULT_POD_COUNT,
        ecs_spec=ecs_spec,
        resource_config=resource_config,
    )

    return dlc_models.CreateJobRequest(
        display_name=optional_string(job_config, "job_name") or "evo-train",
        workspace_id=workspace_id,
        resource_id=resource_id,
        job_type=DEFAULT_JOB_TYPE,
        job_specs=[job_spec],
        data_sources=make_data_sources(job_config),
        user_command=build_train_command(job_config),
        accessibility=DEFAULT_ACCESSIBILITY,
        job_max_running_time_minutes=optional_int(job_config, "max_running_minutes"),
        description=optional_string(job_config, "description"),
        envs={
            "DATASET_PATH": dataset_path,
            "EPOCHS": str(epochs),
            "CHECKPOINT_PATH": checkpoint_path,
            "CHECKPOINT_FREQUENCY": str(checkpoint_frequency),
            "GPU_COUNT": str(gpu_count),
        },
    )


def fetch_job_body(client: Any, job_id: str, *, need_detail: bool = False) -> Any:
    load_dlc_sdk()
    response = client.get_job(job_id, dlc_models.GetJobRequest(need_detail=need_detail))
    return response.body


def print_job_body(body: Any, *, need_detail: bool = False) -> None:
    if need_detail:
        print_json(body)
        return
    print_json(
        {
            "job_id": body.job_id,
            "display_name": body.display_name,
            "status": body.status,
            "sub_status": body.sub_status,
            "reason_code": body.reason_code,
            "reason_message": body.reason_message,
            "workspace_id": body.workspace_id,
            "resource_id": body.resource_id,
            "user_command": body.user_command,
        }
    )


def stop_job_body(client: Any, job_id: str) -> Any:
    load_dlc_sdk()
    response = client.stop_job(job_id, dlc_models.StopJobRequest())
    return response.body


class AliyunDLCPlatform(TrainPlatform):
    def __init__(self, region_id: str | None = None, client: Any | None = None) -> None:
        self.region_id = region_id or os.getenv("ALIYUN_REGION", "cn-hangzhou")
        self._client = client

    def _client_instance(self) -> Any:
        if self._client is None:
            self._client = create_client(self.region_id)
        return self._client

    def submit(self, job_config: dict[str, Any]) -> str:
        request = build_job_request(job_config)
        if config_bool(job_config, "dry_run"):
            print_json(request)
            return ""
        log("submitting PAI DLC training job")
        response = self._client_instance().create_job(request)
        print_json(response.body)
        job_id = getattr(response.body, "job_id", None)
        if not job_id:
            raise SystemExit("CreateJob did not return a job_id")
        return str(job_id)

    def status(self, job_id: str) -> str:
        body = self.fetch_job_body(job_id, need_detail=False)
        return str(getattr(body, "status", "") or "")

    def metadata(self, job_id: str) -> dict[str, str]:
        body = self.fetch_job_body(job_id, need_detail=False)
        return {
            "status": str(getattr(body, "status", "") or ""),
            "last_error": str(getattr(body, "reason_message", "") or ""),
        }

    def stop(self, job_id: str) -> None:
        body = stop_job_body(self._client_instance(), job_id)
        print_json(body)

    def fetch_job_body(self, job_id: str, *, need_detail: bool = False) -> Any:
        return fetch_job_body(self._client_instance(), job_id, need_detail=need_detail)
