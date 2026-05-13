#!/usr/bin/env python3
"""Submit a PAI DLC PyTorchJob with cloned job defaults."""
from __future__ import annotations
import os
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import CreateJobRequest, CreateJobRequestDataSources, CreateJobRequestUserVpc, GetJobRequest, GetPodLogsRequest, GetWebTerminalRequest, JobSpec, ResourceConfig, StopJobRequest
from alibabacloud_tea_openapi.models import Config
from param.env_param import *
DEFAULT_DATA_SOURCES = [
    CreateJobRequestDataSources(data_source_id=OSS_CODE_DATA_SOURCE_ID, mount_path=PAI_MOUNT_PATH_OF_CODE),
    CreateJobRequestDataSources(uri=NAS_MODEL_RESULT_URI, mount_path=PAI_MOUNT_PATH_OF_MODEL_RESULT, mount_access="RW"),
]
# DEFAULT_DATA_SOURCES = [
#     CreateJobRequestDataSources(data_source_id="d-xfobl8zdj3cqdrleqo", mount_path="/mnt/pai/data/"), # 数据集的id，以及希望数据集挂载在拉起的容器的什么路径
#     CreateJobRequestDataSources(data_source_id="d-hzpwiw5qvtyy7887oe", mount_path="/mnt/code/"), # 由于容器无法使用clone，所以暂时将代码也用数据集的方式进行管理
#     CreateJobRequestDataSources(uri="nas://001vtgf4opoobb8u5gh-vfp55.cn-hangzhou.nas.aliyuncs.com/", mount_path="/usrresult/", mount_access="RW")
# ]
#    如果想要将oss挂载到pai容器中，则执行以下命令
#    CreateJobRequestDataSources(uri="oss://evo-model-result.oss-cn-hangzhou-internal.aliyuncs.com/model-result/", mount_path="/mnt/usrresult/", mount_access="RW")
#    采用数据集方式挂载nas ，发现只读，无法写
#    CreateJobRequestDataSources(data_source_id="d-0dj8202laff1rt7s00", mount_path="/usrresult/", mount_access="RW"),  
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

    def build_create_request(self, usrname: str, datasetname: str = "") -> CreateJobRequest:
        job_spec = JobSpec(
            type=DEFAULT_ROLE,
            image=DEFAULT_IMAGE,
            pod_count=DEFAULT_WORKERS,
            resource_config=ResourceConfig(cpu=DEFAULT_CPU, gpu=DEFAULT_GPU,
                memory=DEFAULT_MEMORY, shared_memory=DEFAULT_SHARED_MEMORY),
        )
        # dataset_oss_url = "oss://evo-model-result.oss-cn-hangzhou-internal.aliyuncs.com/all_usr_dataset"
        data_sources = list(DEFAULT_DATA_SOURCES)
        # 如果用户有指定自己的数据集，则将指定的oss路径挂载pai容器的/mnt/pai/data路径下，否则挂载一个默认的数据集到该路径下
        # 对于后端，则是把evo-data/all_usr_dataset/都给挂载到了/home/evomind/usrdata/目录下。
        if datasetname != "":
            user_dataset_oss_url = (OSS_USR_DATASET_URI % (usrname, datasetname)) + "/"
            data_sources.append(CreateJobRequestDataSources(uri=user_dataset_oss_url, mount_path=PAI_MOUNT_PATH_OF_USR_DATASET))
            print(user_dataset_oss_url)
        else:
            data_sources.append(CreateJobRequestDataSources(data_source_id=DEFAULT_USR_DATASET_ID,  mount_path=PAI_MOUNT_PATH_OF_USR_DATASET))
                 
    # 如果用户没有指定数据集，就用默认的数据密
    # 数据集的id，以及希望数据集挂载在拉起的容器的什么路径

        return CreateJobRequest(
            resource_id=DEFAULT_RESOURCE_ID,
            workspace_id=DEFAULT_WORKSPACE_ID,
            display_name=DEFAULT_JOB_NAME,
            job_type=DEFAULT_JOB_TYPE,
            job_specs=[job_spec],
            data_sources=data_sources,
            user_command=self.user_cmd,
            priority=DEFAULT_PRIORITY,
            user_vpc=CreateJobRequestUserVpc(
                vpc_id=DEFAULT_VPC_ID,
                switch_id=DEFAULT_SWITCH_ID,
                security_group_id=DEFAULT_SECURITY_GROUP_ID,
                extended_cidrs=DEFAULT_EXTENDED_CIDRS,
                default_route=DEFAULT_ROUTE,
            ),
        )
    def submit_job(self, usrname: str, datasetname: str) -> None:
        response = self.client.create_job(self.build_create_request(usrname, datasetname))
        self.job_id = response.body.job_id
        print(f"任务提交成功！Job ID: {self.job_id}")
        print("开始实时跟踪任务状态...\n")

    def stop_job(self) -> None:
        self.client.stop_job(self.job_id, StopJobRequest())
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

    def query_user_logs(self, max_lines: int = 200) -> list[str]:
        job = self.client.get_job(self.job_id, GetJobRequest()).body
        if not job.pods:
            return []
        pod_id = job.pods[0].pod_id
        response = self.client.get_pod_logs(
            self.job_id,
            pod_id,
            GetPodLogsRequest(max_lines=max_lines),
        )
        return list(response.body.logs or [])
