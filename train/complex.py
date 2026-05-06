#!/usr/bin/env python3
"""Submit a heavier Alibaba Cloud PAI DLC training job.

This script intentionally reads credentials and uncertain resource identifiers
from environment variables or command-line flags instead of hardcoding them.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, List, Optional

from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import (
    CreateJobRequest,
    CreateJobRequestCodeSource,
    CreateJobRequestDataSources,
    GetJobRequest,
    JobSpec,
    ResourceConfig,
)
from alibabacloud_tea_openapi.models import Config


DEFAULT_REGION = "cn-hangzhou"
DEFAULT_WORKSPACE_ID = "604723"
DEFAULT_RESOURCE_ID = "quota8xa9l64xn9c"
DEFAULT_JOB_NAME = "evo-lerobot-complex"
DEFAULT_JOB_TYPE = "PyTorchJob"
DEFAULT_ROLE = "Worker"
DEFAULT_CPU = "10"
DEFAULT_GPU = "1"
DEFAULT_MEMORY = "50Gi"
DEFAULT_DATASET_MOUNT_PATH = "/root/data"
DEFAULT_DATASET_MOUNT_ACCESS = "RW"
DEFAULT_CODE_MOUNT_PATH = "/root/code"
TERMINAL_STATUSES = {"Stopped", "Succeeded", "Failed"}

TRAIN_COMMAND = """lerobot-train \\
  --dataset.repo_id=local/libero_10_no_noops_1.0.0_lerobot \\
  --dataset.root=/root/data/libero_10_no_noops_1.0.0_lerobot \\
  --dataset.video_backend=pyav \\
  --policy.type=act \\
  --policy.push_to_hub=false \\
  --policy.repo_id=local/libero_10_no_noops_1.0.0_lerobot \\
  --output_dir=/root/data/checkpoint \\
  --steps=1000 \\
  --policy.device=cuda"""


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_client(region: str) -> Client:
    return Client(
        config=Config(
            access_key_id=require_env("ALIBABA_CLOUD_ACCESS_KEY_ID"),
            access_key_secret=require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
            region_id=region,
            endpoint=f"pai-dlc.{region}.aliyuncs.com",
        )
    )


def make_data_sources(args: argparse.Namespace) -> List[Any]:
    if not args.dataset_uri and not args.dataset_id:
        return []

    data_source = CreateJobRequestDataSources(
        data_source_id=args.dataset_id,
        uri=args.dataset_uri,
        mount_path=args.dataset_mount_path,
    )
    if hasattr(data_source, "mount_access"):
        data_source.mount_access = args.dataset_mount_access

    return [data_source]


def make_code_source(args: argparse.Namespace) -> Optional[CreateJobRequestCodeSource]:
    if not args.code_source_id:
        return None

    return CreateJobRequestCodeSource(
        code_source_id=args.code_source_id,
        branch=args.code_branch,
        commit=args.code_commit,
        mount_path=args.code_mount_path,
    )


def build_request(args: argparse.Namespace) -> CreateJobRequest:
    resource_config = ResourceConfig(
        cpu=args.cpu,
        gpu=args.gpu,
        memory=args.memory,
    )
    if args.gpu_type:
        resource_config.gputype = args.gpu_type

    job_spec = JobSpec(
        type=DEFAULT_ROLE,
        image=args.image,
        pod_count=1,
        resource_config=resource_config,
    )
    if args.ecs_spec:
        job_spec.ecs_spec = args.ecs_spec

    return CreateJobRequest(
        resource_id=args.resource_id,
        workspace_id=args.workspace_id,
        display_name=args.job_name,
        job_type=args.job_type,
        job_specs=[job_spec],
        code_source=make_code_source(args),
        data_sources=make_data_sources(args),
        user_command=TRAIN_COMMAND,
        envs={
            "DATASET_ROOT": "/root/data/libero_10_no_noops_1.0.0_lerobot",
            "OUTPUT_DIR": "/root/data/checkpoint",
        },
    )


def submit_job(client: Client, args: argparse.Namespace) -> str:
    response = client.create_job(build_request(args))
    job_id = response.body.job_id
    print(f"任务提交成功！Job ID: {job_id}")
    print("开始实时跟踪任务状态...\n")
    return job_id


def wait_job(client: Client, job_id: str, interval: int) -> None:
    while True:
        job = client.get_job(job_id, GetJobRequest()).body
        status = job.status
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务状态: {status}")
        if status in TERMINAL_STATUSES:
            print("\n===== 任务最终状态 =====")
            print(f"Job ID: {job_id}")
            print(f"状态: {job.status}")
            print(f"执行命令: {job.user_command}")
            return
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a complex PAI DLC lerobot training job.")
    parser.add_argument("--region", default=os.environ.get("ALIYUN_REGION", DEFAULT_REGION))
    parser.add_argument("--workspace-id", default=os.environ.get("PAI_WORKSPACE_ID", DEFAULT_WORKSPACE_ID))
    parser.add_argument("--resource-id", default=os.environ.get("PAI_RESOURCE_ID", DEFAULT_RESOURCE_ID))
    parser.add_argument("--job-name", default=os.environ.get("PAI_DLC_JOB_NAME", DEFAULT_JOB_NAME))
    parser.add_argument("--job-type", default=os.environ.get("PAI_DLC_JOB_TYPE", DEFAULT_JOB_TYPE))
    parser.add_argument(
        "--image",
        default=os.environ.get("PAI_DLC_IMAGE"),
        required=os.environ.get("PAI_DLC_IMAGE") is None,
    )
    parser.add_argument("--ecs-spec", default=os.environ.get("PAI_ECS_SPEC"))
    parser.add_argument("--cpu", default=os.environ.get("PAI_DLC_CPU", DEFAULT_CPU))
    parser.add_argument("--gpu", default=os.environ.get("PAI_DLC_GPU", DEFAULT_GPU))
    parser.add_argument("--gpu-type", default=os.environ.get("PAI_DLC_GPU_TYPE"))
    parser.add_argument("--memory", default=os.environ.get("PAI_DLC_MEMORY", DEFAULT_MEMORY))
    parser.add_argument("--dataset-id", default=os.environ.get("PAI_DLC_DATASET_ID"))
    parser.add_argument("--dataset-uri", default=os.environ.get("PAI_DLC_DATASET_URI"))
    parser.add_argument(
        "--dataset-mount-path",
        default=os.environ.get("PAI_DLC_DATASET_MOUNT_PATH", DEFAULT_DATASET_MOUNT_PATH),
    )
    parser.add_argument(
        "--dataset-mount-access",
        default=os.environ.get("PAI_DLC_DATASET_MOUNT_ACCESS", DEFAULT_DATASET_MOUNT_ACCESS),
    )
    parser.add_argument("--code-source-id", default=os.environ.get("PAI_DLC_CODE_SOURCE_ID"))
    parser.add_argument("--code-branch", default=os.environ.get("PAI_DLC_CODE_BRANCH"))
    parser.add_argument("--code-commit", default=os.environ.get("PAI_DLC_CODE_COMMIT"))
    parser.add_argument("--code-mount-path", default=os.environ.get("PAI_DLC_CODE_MOUNT_PATH", DEFAULT_CODE_MOUNT_PATH))
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--no-wait", action="store_true", help="Only create the job and print its ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_client(args.region)
    job_id = submit_job(client, args)
    if not args.no_wait:
        wait_job(client, job_id, args.interval)


if __name__ == "__main__":
    main()
