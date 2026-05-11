#!/usr/bin/env python3
"""Submit a PAI DLC PyTorchJob with cloned job defaults."""
from __future__ import annotations
import os
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import CreateJobRequest, CreateJobRequestDataSources, CreateJobRequestUserVpc, GetJobRequest, GetWebTerminalRequest, JobSpec, ResourceConfig
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

DEFAULT_DATA_SOURCES = [
    CreateJobRequestDataSources(data_source_id="d-xfobl8zdj3cqdrleqo", mount_path="/mnt/pai/data/"), # 数据集的id，以及希望数据集挂载在拉起的容器的什么路径
    CreateJobRequestDataSources(data_source_id="d-hzpwiw5qvtyy7887oe", mount_path="/mnt/code/"), # 由于容器无法使用clone，所以暂时将代码也用数据集的方式进行管理
    CreateJobRequestDataSources(uri="oss://evo-model-result.oss-cn-hangzhou-internal.aliyuncs.com/model-result/", mount_path="/mnt/usrresult/", mount_access="RW")
]
# status 的取值
# {
#     "Creating",
#     "Queuing",
#     "Bidding",          # 仅灵骏 Spot 作业
#     "EnvPreparing",
#     "SanityChecking",
#     "Running",
#     "Restarting",
#     "Stopping",
#     "SucceededReserving",
#     "FailedReserving",
#     "Succeeded",
#     "Failed",
#     "Stopped",
# }

class PaiRequest:
    def __init__(self, user_cmd: str, job_id: str | None = None) -> None:
        self.user_cmd = user_cmd
        self.status = ""
        self.client = Client(config=Config(
            access_key_id=self.require_env("ALIBABA_CLOUD_ACCESS_KEY_ID"),
            access_key_secret=self.require_env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
            region_id=DEFAULT_REGION,
            endpoint=f"pai-dlc.{DEFAULT_REGION}.aliyuncs.com",
            )
        )
        self.job_id = job_id

    @staticmethod
    def require_env(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    def build_create_request(self) -> CreateJobRequest:
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
            user_command=self.user_cmd,
            priority=DEFAULT_PRIORITY,
            user_vpc=CreateJobRequestUserVpc(
                vpc_id=DEFAULT_VPC_ID,
                switch_id=DEFAULT_SWITCH_ID,
                security_group_id=DEFAULT_SECURITY_GROUP_ID,
                default_route=DEFAULT_ROUTE,
            ),
        )
    def submit_job(self) -> None:
        response = self.client.create_job(self.build_create_request())
        self.job_id = response.body.job_id
        print(f"任务提交成功！Job ID: {self.job_id}")
        print("开始实时跟踪任务状态...\n")

    def stop_job(self) -> None:
        self.client.stop_job(self.job_id)
        print(f"任务停止请求已发送！Job ID: {self.job_id}")

    def query_job(self) -> str:
        job = self.client.get_job(self.job_id, GetJobRequest()).body
        self.status = job.status
        print(f"任务状态: {self.status}")
        if self.status != "Running":
            return self.status            
        pod_id = job.pods[0].pod_id # 如果容器没有起来，id不可靠
        response = self.client.get_web_terminal(
            self.job_id,
            pod_id,
            GetWebTerminalRequest(is_shared=True),
        )
        return f"{self.status}, Link: {response.body.web_terminal_url}"
