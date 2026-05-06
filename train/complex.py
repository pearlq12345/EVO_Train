#!/usr/bin/env python3
"""Submit a PAI DLC PyTorchJob with cloned job defaults."""
from __future__ import annotations
import os
import time
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import CreateJobRequest, CreateJobRequestDataSources, CreateJobRequestUserVpc, GetJobRequest, JobSpec, ResourceConfig
from alibabacloud_tea_openapi.models import Config

DEFAULT_REGION = "cn-hangzhou"
DEFAULT_WORKSPACE_ID = "604723"
DEFAULT_RESOURCE_ID = "quota8xa9l64xn9c"
DEFAULT_JOB_NAME = "job_for_mirror"
DEFAULT_JOB_TYPE = "PyTorchJob"
DEFAULT_ROLE = "Worker"
DEFAULT_IMAGE = "evo-train-mirror-registry-vpc.cn-hangzhou.cr.aliyuncs.com/evo-mirror-namespace/evo-mirror:v1"
DEFAULT_CPU = "10"
DEFAULT_GPU = "1"
DEFAULT_MEMORY = "50Gi"
DEFAULT_SHARED_MEMORY = "50Gi"
DEFAULT_WORKERS = 1
DEFAULT_VPC_ID = "vpc-bp1txl55oqch56uhbpc43"
DEFAULT_SWITCH_ID = "vsw-bp14hh8tnognnao2f79n2"
DEFAULT_SECURITY_GROUP_ID = "sg-bp171lhsibv9dwe0uia7"
DEFAULT_PRIORITY = 9
DEFAULT_ROUTE = "eth1"
TERMINAL_STATUSES = {"Stopped", "Succeeded", "Failed"}
DEFAULT_COMMAND = "sleep 3000"
POLL_INTERVAL = 5


DEFAULT_DATA_SOURCES = [
    CreateJobRequestDataSources(data_source_id="d-xfobl8zdj3cqdrleqo", mount_path="/mnt/pai/data/"), # 数据集的id，以及希望数据集挂载在拉起的容器的什么路径
    CreateJobRequestDataSources(data_source_id="d-hzpwiw5qvtyy7887oe", mount_path="/mnt/code/"), # 由于容器无法使用clone，所以暂时将代码也用数据集的方式进行管理
]

def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_request() -> CreateJobRequest:
    job_spec = JobSpec(
        type=DEFAULT_ROLE,
        image=DEFAULT_IMAGE,
        pod_count=DEFAULT_WORKERS,
        resource_config=ResourceConfig(cpu=DEFAULT_CPU, gpu=DEFAULT_GPU,
            memory=DEFAULT_MEMORY, shared_memory=DEFAULT_SHARED_MEMORY),
    )

    return CreateJobRequest(
        resource_id=DEFAULT_RESOURCE_ID,
        workspace_id=DEFAULT_WORKSPACE_ID,
        display_name=DEFAULT_JOB_NAME,
        job_type=DEFAULT_JOB_TYPE,
        job_specs=[job_spec],
        data_sources=DEFAULT_DATA_SOURCES,
        user_command=DEFAULT_COMMAND,
        priority=DEFAULT_PRIORITY,
        user_vpc=CreateJobRequestUserVpc(
            vpc_id=DEFAULT_VPC_ID,
            switch_id=DEFAULT_SWITCH_ID,
            security_group_id=DEFAULT_SECURITY_GROUP_ID,
            default_route=DEFAULT_ROUTE,
        ),
    )


def submit_job(client: Client) -> str:
    response = client.create_job(build_request())
    job_id = response.body.job_id
    print(f"任务提交成功！Job ID: {job_id}")
    print("开始实时跟踪任务状态...\n")
    return job_id


def wait_job(client: Client, job_id: str) -> None:
    while True:
        job = client.get_job(job_id, GetJobRequest()).body
        status = job.status
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务状态: {status}")
        if status in TERMINAL_STATUSES:
            print(f"\n===== 任务最终状态 =====\nJob ID: {job_id}\n状态: {job.status}\n执行命令: {job.user_command}")
            return
        time.sleep(POLL_INTERVAL)


def main() -> None:
    client = Client(config=Config(
            access_key_id=require_env("ALIBABA_CLOUD_ACCESS_KEY_ID"),
            access_key_secret=require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
            region_id=DEFAULT_REGION,
            endpoint=f"pai-dlc.{DEFAULT_REGION}.aliyuncs.com",
        )
    )
    job_id = submit_job(client)
    wait_job(client, job_id)


if __name__ == "__main__":
    main()
