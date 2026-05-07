#!/usr/bin/env python3
"""Submit and manage training jobs across supported platforms.

Typical usage:
  python3 start_train.py submit \
    --dataset-path /mnt/nas/libero_10_no_noops_1.0.0_lerobot \
    --epochs 10 \
    --checkpoint-path /mnt/nas/checkpoints/run-001 \
    --checkpoint-frequency 1 \
    --gpu-count 1

Required backend settings can be passed as flags or environment variables.
For Aliyun DLC:
  PAI_WORKSPACE_ID, PAI_DLC_IMAGE, PAI_RESOURCE_ID, PAI_ECS_SPEC
For AutoDL:
  AUTODL_HOST, AUTODL_PORT, AUTODL_USER, AUTODL_KEY_PATH, AUTODL_WORKDIR

Parameter notes:
  Global arguments:
    --platform
      Training backend name. Defaults to TRAIN_PLATFORM, then aliyun.
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
    --workdir
      Optional working directory used by SSH-based backends such as AutoDL.

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
from train.platform.base import TrainPlatform
from train.platform.factory import get_platform


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


def resolve_platform(args: argparse.Namespace) -> TrainPlatform:
    platform_name = (args.platform or "").strip().lower() or None
    return get_platform(platform_name, region_id=args.region)


def submit_job(args: argparse.Namespace, platform: TrainPlatform | None = None) -> str:
    platform = platform or resolve_platform(args)
    return platform.submit(vars(args))


def get_job(args: argparse.Namespace, platform: TrainPlatform | None = None) -> Any:
    platform = platform or resolve_platform(args)
    if isinstance(platform, AliyunDLCPlatform):
        body = platform.fetch_job_body(args.job_id, need_detail=args.detail)
        print_job_body(body, need_detail=args.detail)
        return body

    metadata = platform.metadata(args.job_id)
    payload = {
        "job_id": args.job_id,
        "platform": (args.platform or "aliyun").strip().lower(),
        **metadata,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def wait_job(platform: TrainPlatform, job_id: str, timeout: int, interval: int) -> None:
    deadline = time.time() + timeout
    while True:
        if isinstance(platform, AliyunDLCPlatform):
            body = platform.fetch_job_body(job_id, need_detail=False)
            print_job_body(body, need_detail=False)
            status = getattr(body, "status", None)
        else:
            status = platform.status(job_id)
            log(f"job status: {status}")
        if status in DONE_STATUSES:
            log(f"job finished successfully: {status}")
            return
        if status in FAILED_STATUSES:
            raise SystemExit(f"job reached terminal status: {status}")
        if time.time() >= deadline:
            raise SystemExit(f"timeout waiting for job; last status={status}")
        time.sleep(interval)


def stop_job(args: argparse.Namespace, platform: TrainPlatform | None = None) -> None:
    platform = platform or resolve_platform(args)
    platform.stop(args.job_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit training jobs through the selected backend.")

    parser.add_argument(
        "--platform",
        default=os.getenv("TRAIN_PLATFORM", "aliyun"),
        help="Training backend name. Defaults to TRAIN_PLATFORM, then aliyun",
    )
    parser.add_argument("--region", default=os.getenv("ALIYUN_REGION", "cn-hangzhou"))
    parser.add_argument("--env-file", help="Optional .env file with backend credentials and training defaults")
    sub = parser.add_subparsers(dest="command_name", required=True)

    submit = sub.add_parser("submit", help="Create a training job")

    train_args = submit.add_argument_group("training arguments")
    train_args.add_argument("--dataset-path", required=True, help="Path visible inside DLC container, for example /mnt/nas/dataset")
    train_args.add_argument("--epochs", required=True, type=int)
    train_args.add_argument("--checkpoint-path", required=True)
    train_args.add_argument("--checkpoint-frequency", required=True, type=int)
    train_args.add_argument("--command", help="Full training command. Overrides command template")
    train_args.add_argument("--command-template", help="Template using {dataset_path}, {epochs}, {checkpoint_path}, {checkpoint_frequency}, {gpu_count}")
    train_args.add_argument("--workdir", help="Working directory for SSH-based backends such as AutoDL")

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

    status = sub.add_parser("status", help="Show job status")
    status.add_argument("--job-id", required=True)
    status.add_argument("--detail", action="store_true")

    stop = sub.add_parser("stop", help="Stop a job")
    stop.add_argument("--job-id", required=True)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    load_env(args.env_file)

    if args.command_name == "submit":
        platform = resolve_platform(args)
        job_id = submit_job(args, platform)
        if job_id and args.wait:
            wait_job(platform, job_id, args.timeout, args.interval)
    elif args.command_name == "status":
        get_job(args)
    elif args.command_name == "stop":
        stop_job(args)
    else:
        raise SystemExit(f"unknown command: {args.command_name}")


if __name__ == "__main__":
    main()
