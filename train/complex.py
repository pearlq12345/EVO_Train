#!/usr/bin/env python3
"""Submit a PAI DLC PyTorchJob with cloned job defaults."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, List, Optional

from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import (
    CreateJobRequest,
    CreateJobRequestCodeSource,
    CreateJobRequestDataSources,
    CreateJobRequestUserVpc,
    GetJobRequest,
    JobSettings,
    JobSpec,
    ResourceConfig,
)
from alibabacloud_tea_openapi.models import Config


DEFAULT_REGION = "cn-hangzhou"
DEFAULT_WORKSPACE_ID = "604723"
DEFAULT_RESOURCE_ID = "quota8xa9l64xn9c"
DEFAULT_JOB_NAME = "test3_clone4"
DEFAULT_JOB_TYPE = "PyTorchJob"
DEFAULT_ROLE = "Worker"
DEFAULT_IMAGE = "evo-train-mirror-registry-vpc.cn-hangzhou.cr.aliyuncs.com/evo-mirror-namespace/evo-mirror:v1"
DEFAULT_CPU = "10"
DEFAULT_GPU = "1"
DEFAULT_MEMORY = "50Gi"
DEFAULT_SHARED_MEMORY = "50Gi"
DEFAULT_WORKERS = 1
DEFAULT_DATA_SOURCES = "d-jp359y1kyfksonvlvx:v1:/mnt/data/,d-r696y5jblhz39llv99:v1:/mnt/oss/"
DEFAULT_DATASET_ID = None
DEFAULT_DATASET_MOUNT_PATH = "/mnt/data/"
DEFAULT_DATASET_MOUNT_ACCESS = "RW"
DEFAULT_CODE_MOUNT_PATH = "/root/code"
DEFAULT_TAGS = "CloneFromJobID=dlchpho961hk8804"
DEFAULT_VPC_ID = "vpc-bp1txl55oqch56uhbpc43"
DEFAULT_SWITCH_ID = "vsw-bp14hh8tnognnao2f79n2"
DEFAULT_SECURITY_GROUP_ID = "sg-bp171lhsibv9dwe0uia7"
DEFAULT_PRIORITY = 9
DEFAULT_ROUTE = "eth1"
TERMINAL_STATUSES = {"Stopped", "Succeeded", "Failed"}

LEROBOT_COMMAND = """lerobot-train \\
  --dataset.repo_id=local/libero_10_no_noops_1.0.0_lerobot \\
  --dataset.root=/root/data/libero_10_no_noops_1.0.0_lerobot \\
  --dataset.video_backend=pyav \\
  --policy.type=act \\
  --policy.push_to_hub=false \\
  --policy.repo_id=local/libero_10_no_noops_1.0.0_lerobot \\
  --output_dir=/root/data/checkpoint \\
  --steps=1000 \\
  --policy.device=cuda"""
DEFAULT_COMMAND = "sleep 3000"


class DataSourceSpec:
    def __init__(
        self,
        data_source_id: str | None = None,
        data_source_version: str | None = None,
        mount_path: str | None = None,
        uri: str | None = None,
        mount_access: str | None = None,
    ):
        self.data_source_id = data_source_id
        self.data_source_version = data_source_version
        self.mount_path = mount_path
        self.uri = uri
        self.mount_access = mount_access

    def validate(self) -> None:
        return None

    def to_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.data_source_id:
            result["DataSourceId"] = self.data_source_id
        if self.data_source_version:
            result["DataSourceVersion"] = self.data_source_version
        if self.mount_path:
            result["MountPath"] = self.mount_path
        if self.uri:
            result["Uri"] = self.uri
        if self.mount_access:
            result["MountAccess"] = self.mount_access
        return result


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


def parse_tags(value: str | None) -> dict[str, str] | None:
    if not value:
        return None

    tags: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, separator, tag_value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid tag value: {item}. Expected KEY=VALUE.")
        tags[key.strip()] = tag_value.strip()
    return tags or None


def parse_data_source(value: str) -> DataSourceSpec:
    parts = value.split(":", 2)
    if len(parts) == 3:
        data_source_id, data_source_version, mount_path = parts
    elif len(parts) == 2:
        data_source_id, mount_path = parts
        data_source_version = None
    else:
        raise ValueError(f"Invalid data source value: {value}. Expected ID[:VERSION]:MOUNT_PATH.")

    return DataSourceSpec(
        data_source_id=data_source_id,
        data_source_version=data_source_version,
        mount_path=mount_path,
    )


def make_data_sources(args: argparse.Namespace) -> List[Any]:
    if args.data_sources:
        return [parse_data_source(item.strip()) for item in args.data_sources.split(",") if item.strip()]

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


def make_user_vpc(args: argparse.Namespace) -> Optional[CreateJobRequestUserVpc]:
    if not args.vpc_id and not args.switch_id and not args.security_group_id and not args.default_route:
        return None

    return CreateJobRequestUserVpc(
        vpc_id=args.vpc_id,
        switch_id=args.switch_id,
        security_group_id=args.security_group_id,
        default_route=args.default_route,
    )


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
    resource_config = None
    if not args.ecs_spec:
        resource_config = ResourceConfig(
            cpu=args.cpu,
            gpu=args.gpu,
            memory=args.memory,
            shared_memory=args.shared_memory,
        )
        if args.gpu_type:
            resource_config.gputype = args.gpu_type

    job_spec = JobSpec(
        type=DEFAULT_ROLE,
        image=args.image,
        pod_count=args.workers,
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
        user_command=args.command,
        priority=args.priority,
        settings=JobSettings(tags=parse_tags(args.tags)),
        user_vpc=make_user_vpc(args),
        envs={
            "DATASET_ROOT": "/root/data/libero_10_no_noops_1.0.0_lerobot",
            "OUTPUT_DIR": "/root/data/checkpoint",
        },
    )


def submit_job(client: Client, args: argparse.Namespace) -> str:
    request = build_request(args)
    if args.dry_run:
        print(json.dumps(request.to_map(), ensure_ascii=False, indent=2))
        return ""

    response = client.create_job(request)
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
    parser = argparse.ArgumentParser(description="Submit a PAI DLC dataset mount repro job.")
    parser.add_argument("--region", default=os.environ.get("ALIYUN_REGION", DEFAULT_REGION))
    parser.add_argument("--workspace-id", default=os.environ.get("PAI_WORKSPACE_ID", DEFAULT_WORKSPACE_ID))
    parser.add_argument("--resource-id", default=os.environ.get("PAI_RESOURCE_ID", DEFAULT_RESOURCE_ID))
    parser.add_argument("--job-name", default=os.environ.get("PAI_DLC_JOB_NAME", DEFAULT_JOB_NAME))
    parser.add_argument("--job-type", default=os.environ.get("PAI_DLC_JOB_TYPE", DEFAULT_JOB_TYPE))
    parser.add_argument(
        "--image",
        default=os.environ.get("PAI_DLC_IMAGE", DEFAULT_IMAGE),
    )
    parser.add_argument("--ecs-spec", default=os.environ.get("PAI_ECS_SPEC"))
    parser.add_argument("--cpu", default=os.environ.get("PAI_DLC_CPU", DEFAULT_CPU))
    parser.add_argument("--gpu", default=os.environ.get("PAI_DLC_GPU", DEFAULT_GPU))
    parser.add_argument("--gpu-type", default=os.environ.get("PAI_DLC_GPU_TYPE"))
    parser.add_argument("--memory", default=os.environ.get("PAI_DLC_MEMORY", DEFAULT_MEMORY))
    parser.add_argument("--shared-memory", default=os.environ.get("PAI_DLC_SHARED_MEMORY", DEFAULT_SHARED_MEMORY))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PAI_DLC_WORKERS", DEFAULT_WORKERS)))
    parser.add_argument("--data-sources", default=os.environ.get("PAI_DLC_DATA_SOURCES", DEFAULT_DATA_SOURCES))
    parser.add_argument("--dataset-id", default=os.environ.get("PAI_DLC_DATASET_ID", DEFAULT_DATASET_ID))
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
    parser.add_argument("--command", default=os.environ.get("PAI_DLC_COMMAND", DEFAULT_COMMAND))
    parser.add_argument("--lerobot-command", action="store_const", dest="command", const=LEROBOT_COMMAND)
    parser.add_argument("--tags", default=os.environ.get("PAI_DLC_TAGS", DEFAULT_TAGS))
    parser.add_argument("--priority", type=int, default=int(os.environ.get("PAI_DLC_PRIORITY", DEFAULT_PRIORITY)))
    parser.add_argument("--vpc-id", default=os.environ.get("PAI_DLC_VPC_ID", DEFAULT_VPC_ID))
    parser.add_argument("--switch-id", default=os.environ.get("PAI_DLC_SWITCH_ID", DEFAULT_SWITCH_ID))
    parser.add_argument("--security-group-id", default=os.environ.get("PAI_DLC_SECURITY_GROUP_ID", DEFAULT_SECURITY_GROUP_ID))
    parser.add_argument("--default-route", default=os.environ.get("PAI_DLC_DEFAULT_ROUTE", DEFAULT_ROUTE))
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Print the CreateJob request without submitting.")
    parser.add_argument("--no-wait", action="store_true", help="Only create the job and print its ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_client(args.region)
    job_id = submit_job(client, args)
    if job_id and not args.no_wait:
        wait_job(client, job_id, args.interval)


if __name__ == "__main__":
    main()
