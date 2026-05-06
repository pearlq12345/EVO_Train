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
import os
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.platform.aliyun_dlc import (
    AliyunDLCPlatform,
    DONE_STATUSES,
    FAILED_STATUSES,
    create_client as create_dlc_client,
    fetch_job_body,
    print_job_body,
)


DEFAULT_ENV_FILES = [
    Path.cwd() / ".env",
    Path.home() / "EVO_Train" / ".env",
    Path("/home/evomind/evo-data_backend/.env"),
]


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


def create_client(region_id: str) -> Any:
    return create_dlc_client(region_id)


def submit_job(client: Any, args: argparse.Namespace) -> str:
    platform = AliyunDLCPlatform(region_id=args.region, client=client)
    return platform.submit(vars(args))


def get_job(client: Any, job_id: str, need_detail: bool = False) -> Any:
    body = fetch_job_body(client, job_id, need_detail=need_detail)
    print_job_body(body, need_detail=need_detail)
    return body


def wait_job(client: Any, job_id: str, timeout: int, interval: int) -> None:
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


def stop_job(client: Any, job_id: str) -> None:
    platform = AliyunDLCPlatform(client=client)
    platform.stop(job_id)


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
