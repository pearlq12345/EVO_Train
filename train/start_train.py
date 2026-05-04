#!/usr/bin/env python3
"""Submit and manage Alibaba Cloud PAI DLC training jobs.

Typical usage:
  python3 start_train.py submit \
    --dataset-path /mnt/nas/libero_10_no_noops_1.0.0_lerobot \
    --epochs 10 \
    --checkpoint-path /mnt/nas/checkpoints/run-001 \
    --checkpoint-frequency 1 \
    --gpu-count 1

Required PAI settings can be passed as flags or environment variables:
  PAI_WORKSPACE_ID, PAI_DLC_IMAGE, PAI_RESOURCE_ID, PAI_ECS_SPEC

Parameter notes:
  Global arguments:
    --region
      Alibaba Cloud region used by the DLC API. Defaults to ALIYUN_REGION,
      then cn-hangzhou.
    --env-file
      Extra .env file to load credentials and PAI defaults from.

  submit training arguments:
    --dataset-path
      Dataset path visible inside the DLC container. This is passed to the
      training command as --dataset-path and exported as DATASET_PATH.
    --epochs
      Number of training epochs. This is passed to the training command and
      exported as EPOCHS.
    --checkpoint-path
      Directory for saving training checkpoints. This is passed to the training
      command and exported as CHECKPOINT_PATH.
    --checkpoint-frequency
      Checkpoint save interval, usually measured in epochs by the training
      script. This is passed to the training command and exported as
      CHECKPOINT_FREQUENCY.
    --command
      Full command to run inside the DLC container. When set, it overrides the
      generated training command and --command-template.
    --command-template
      Template used to build the training command. Available placeholders are
      {dataset_path}, {epochs}, {checkpoint_path}, {checkpoint_frequency}, and
      {gpu_count}. Defaults to DLC_TRAIN_COMMAND_TEMPLATE, then the built-in
      train.py command.

  submit PAI DLC job arguments:
    --job-name
      Display name of the DLC job in PAI.
    --workspace-id
      PAI workspace ID. Defaults to PAI_WORKSPACE_ID.
    --resource-id
      PAI resource group ID. Defaults to PAI_RESOURCE_ID.
    --image
      Training container image used by DLC. Defaults to PAI_DLC_IMAGE.
    --description
      Optional description shown on the DLC job.

  submit resource arguments:
    --ecs-spec
      DLC ECS instance specification. Defaults to PAI_ECS_SPEC.
    --gpu-count
      Number of GPUs requested for the worker. Also exported as GPU_COUNT.
    --gpu-type
      Optional GPU type hint for the resource config.
    --cpu
      Optional CPU count requested for the worker.
    --memory
      Optional memory requested for the worker, in GB.
    --max-running-minutes
      Optional maximum runtime for the DLC job. PAI stops the job after this
      limit.

  submit data arguments:
    --mount
      Data source mount specification in URI=MOUNT_PATH[:RO|RW] format. Can be
      passed multiple times, for example nas://xxx/=/mnt/nas:RW.

  submit execution arguments:
    --wait
      Keep polling the DLC job after submission until it succeeds, fails, or
      times out.
    --timeout
      Maximum seconds to wait when --wait is set. Defaults to 86400.
    --interval
      Polling interval in seconds when --wait is set. Defaults to 30.
    --dry-run
      Print the CreateJob request JSON without submitting the job.

  status arguments:
    --job-id
      DLC job ID to inspect.
    --detail
      Print detailed job information instead of the compact status view.

  stop arguments:
    --job-id
      DLC job ID to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

DLCClient: Any = None
dlc_models: Any = None
openapi_models: Any = None


DEFAULT_ENV_FILES = [
    Path.cwd() / ".env",
    Path.home() / "EVO_Train" / ".env",
    Path("/home/evomind/evo-data_backend/.env"),
]

DONE_STATUSES = {"Succeeded", "Succeed", "SUCCESS", "SUCCEEDED"}
FAILED_STATUSES = {"Failed", "FAILED", "Stopped", "STOPPED", "Deleted", "DELETED"}
DEFAULT_JOB_TYPE = "PyTorch"
DEFAULT_JOB_ROLE = "Worker"
DEFAULT_ACCESSIBILITY = "PRIVATE"
DEFAULT_POD_COUNT = 1


def log(message: str) -> None:
    print(f"[start_train] {message}", flush=True)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_env(extra_env: str | None) -> None:
    for path in DEFAULT_ENV_FILES:
        load_dotenv_file(path)
    if extra_env:
        load_dotenv_file(Path(extra_env).expanduser())


def first_value(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def require_value(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing required value: {name}")
    return value


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
            "alibabacloud_pai-dlc20201203 alibabacloud_tea_openapi python-dotenv",
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


def build_train_command(args: argparse.Namespace) -> str:
    if args.command:
        return args.command

    template = first_value(
        args.command_template,
        os.environ.get("DLC_TRAIN_COMMAND_TEMPLATE"),
        "python train.py --dataset-path {dataset_path} --epochs {epochs} "
        "--checkpoint-path {checkpoint_path} --checkpoint-frequency {checkpoint_frequency}",
    )
    values = {
        "dataset_path": shlex.quote(args.dataset_path),
        "epochs": args.epochs,
        "checkpoint_path": shlex.quote(args.checkpoint_path),
        "checkpoint_frequency": args.checkpoint_frequency,
        "gpu_count": args.gpu_count,
    }
    return template.format(**values)


def make_data_sources(args: argparse.Namespace) -> list[Any]:
    data_sources: list[Any] = []
    for spec in args.mount or []:
        # Format: URI=MOUNT_PATH[:RO|RW]
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


def build_job_request(args: argparse.Namespace) -> Any:
    workspace_id = require_value(first_value(args.workspace_id, os.environ.get("PAI_WORKSPACE_ID")), "--workspace-id or PAI_WORKSPACE_ID")
    image = require_value(first_value(args.image, os.environ.get("PAI_DLC_IMAGE")), "--image or PAI_DLC_IMAGE")
    ecs_spec = require_value(first_value(args.ecs_spec, os.environ.get("PAI_ECS_SPEC")), "--ecs-spec or PAI_ECS_SPEC")
    resource_id = first_value(args.resource_id, os.environ.get("PAI_RESOURCE_ID"))

    resource_config = dlc_models.ResourceConfig(
        gpu=args.gpu_count,
        cpu=args.cpu,
        memory=args.memory,
    )
    if args.gpu_type:
        resource_config.gputype = args.gpu_type

    job_spec = dlc_models.JobSpec(
        type=DEFAULT_JOB_ROLE,
        image=image,
        pod_count=DEFAULT_POD_COUNT,
        ecs_spec=ecs_spec,
        resource_config=resource_config,
    )

    return dlc_models.CreateJobRequest(
        display_name=args.job_name,
        workspace_id=workspace_id,
        resource_id=resource_id,
        job_type=DEFAULT_JOB_TYPE,
        job_specs=[job_spec],
        data_sources=make_data_sources(args),
        user_command=build_train_command(args),
        accessibility=DEFAULT_ACCESSIBILITY,
        job_max_running_time_minutes=args.max_running_minutes,
        description=args.description,
        envs={
            "DATASET_PATH": args.dataset_path,
            "EPOCHS": str(args.epochs),
            "CHECKPOINT_PATH": args.checkpoint_path,
            "CHECKPOINT_FREQUENCY": str(args.checkpoint_frequency),
            "GPU_COUNT": str(args.gpu_count),
        },
    )


def submit_job(client: DLCClient, args: argparse.Namespace) -> str:
    body = create_job(client, args)
    print_json(body)
    job_id = getattr(body, "job_id", None)
    if not job_id:
        raise SystemExit("CreateJob did not return a job_id")
    return job_id


def create_job(client: DLCClient, args: argparse.Namespace) -> Any:
    request = build_job_request(args)
    if args.dry_run:
        print_json(request)
        return None
    log("submitting PAI DLC training job")
    response = client.create_job(request)
    return response.body


def build_status_payload(body: Any) -> dict[str, Any]:
    return {
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


def fetch_job_body(client: DLCClient, job_id: str, need_detail: bool = False) -> Any:
    response = client.get_job(job_id, dlc_models.GetJobRequest(need_detail=need_detail))
    return response.body


def get_job(client: DLCClient, job_id: str, need_detail: bool = False) -> Any:
    body = fetch_job_body(client, job_id, need_detail=need_detail)
    if need_detail:
        print_json(body)
    else:
        print_json(build_status_payload(body))
    return body


def wait_job(client: DLCClient, job_id: str, timeout: int, interval: int) -> None:
    deadline = time.time() + timeout
    while True:
        body = get_job(client, job_id, need_detail=False)
        status = getattr(body, "status", None)
        if status in DONE_STATUSES:
            log(f"job finished successfully: {status}")
            return
        if status in FAILED_STATUSES:
            raise SystemExit(f"job reached terminal status: {status}")
        if time.time() >= deadline:
            raise SystemExit(f"timeout waiting for job; last status={status}")
        time.sleep(interval)


def stop_job(client: DLCClient, job_id: str) -> None:
    response = client.stop_job(job_id, dlc_models.StopJobRequest())
    print_json(response.body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit PAI DLC training jobs from ECS.")

    parser.add_argument("--region", default=os.getenv("ALIYUN_REGION", "cn-hangzhou"))
    parser.add_argument("--env-file", help="Optional .env file with Alibaba Cloud credentials and PAI defaults")
    sub = parser.add_subparsers(dest="command_name", required=True)

    submit = sub.add_parser("submit", help="Create a PAI DLC training job")

    train_args = submit.add_argument_group("training arguments")
    train_args.add_argument("--dataset-path", required=True, help="Path visible inside DLC container, for example /mnt/nas/dataset")
    train_args.add_argument("--epochs", required=True, type=int)
    train_args.add_argument("--checkpoint-path", required=True)
    train_args.add_argument("--checkpoint-frequency", required=True, type=int)
    train_args.add_argument("--command", help="Full training command. Overrides command template")
    train_args.add_argument("--command-template", help="Template using {dataset_path}, {epochs}, {checkpoint_path}, {checkpoint_frequency}, {gpu_count}")

    pai_args = submit.add_argument_group("PAI DLC job arguments")
    pai_args.add_argument("--job-name", default="evo-train")
    pai_args.add_argument("--workspace-id", help="Defaults to PAI_WORKSPACE_ID")
    pai_args.add_argument("--resource-id", help="Defaults to PAI_RESOURCE_ID")
    pai_args.add_argument("--image", help="Training image. Defaults to PAI_DLC_IMAGE")
    pai_args.add_argument("--description")

    resource_args = submit.add_argument_group("resource arguments")
    resource_args.add_argument("--ecs-spec", help="PAI DLC ECS spec. Defaults to PAI_ECS_SPEC")
    resource_args.add_argument("--gpu-count", required=True, type=int)
    resource_args.add_argument("--gpu-type")
    resource_args.add_argument("--cpu", type=int)
    resource_args.add_argument("--memory", type=int, help="Memory in GB")
    resource_args.add_argument("--max-running-minutes", type=int)

    data_args = submit.add_argument_group("data arguments")
    data_args.add_argument("--mount", action="append", help="Mount data source as URI=MOUNT_PATH[:RO|RW], for example nas://xxx/=/mnt/nas:RW")

    execution_args = submit.add_argument_group("execution arguments")
    execution_args.add_argument("--wait", action="store_true")
    execution_args.add_argument("--timeout", type=int, default=86400)
    execution_args.add_argument("--interval", type=int, default=30)
    execution_args.add_argument("--dry-run", action="store_true", help="Print CreateJob request without submitting")

    status = sub.add_parser("status", help="Show DLC job status")
    status.add_argument("--job-id", required=True)
    status.add_argument("--detail", action="store_true")

    stop = sub.add_parser("stop", help="Stop a DLC job")
    stop.add_argument("--job-id", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env_file)
    client = create_client(args.region)

    if args.command_name == "submit":
        job_id = submit_job(client, args)
        if job_id and args.wait:
            wait_job(client, job_id, args.timeout, args.interval)
    elif args.command_name == "status":
        get_job(client, args.job_id, args.detail)
    elif args.command_name == "stop":
        stop_job(client, args.job_id)
    else:
        raise SystemExit(f"unknown command: {args.command_name}")


if __name__ == "__main__":
    main()
