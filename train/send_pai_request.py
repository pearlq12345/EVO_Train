#!/usr/bin/env python3
"""Submit a PAI DLC PyTorchJob with cloned job defaults."""
from __future__ import annotations
import os
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import CreateJobRequest, CreateJobRequestDataSources, CreateJobRequestUserVpc, GetJobRequest, GetWebTerminalRequest, JobSpec, ResourceConfig, StopJobRequest
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
DEFAULT_EXTENDED_CIDRS = ["172.19.0.0/16"]
DEFAULT_PRIORITY = 9
DEFAULT_ROUTE = "eth1"

OSS_CODE_DATA_SOURCE_ID = "d-hzpwiw5qvtyy7887oe" # 存放代码的OSS数据集ID
PAI_MOUNT_PATH_OF_CODE = "/mnt/code/" # 代码OSS数据集会被mount在PAI容器的哪个位置
# 由于容器无法使用clone，所以暂时将代码也用数据集的方式进行管理，直接挂载到pai的/mnt/code路径下

NAS_MODEL_RESULT_URI = "nas://001vtgf4opoobb8u5gh-vfp55.cn-hangzhou.nas.aliyuncs.com/" # 存放用户训练结果
PAI_MOUNT_PATH_OF_MODEL_RESULT = "/usrresult/" # pai容器下，用户训练结果存放的位置。也是nas挂载的位置。格式 /usrresult/usrname/taskname/
ECS_MOUNT_PATH_OF_MODEL_RESULT = "/mnt/usrresult" # NAS挂载在ECS的哪个路径下。
# pai 的/usrresult/ 和 ecs 的/mnt/usrresult 挂载的是同一个nas，在pai的/usrresult/usrname/taskname/下生成文件，
# 对应就在 ecs 的"/mnt/usrresult/%s/%s/checkpoint下生成文件
# 当时考虑用nas，是因为nas才支持像真正文件系统一样来读写，而oss虽然也能实现读写，但是无法进行链接操作（lerobot依赖链接操作生成训练结果。）
OSS_USR_DATASET_URI = "oss://evo-data.oss-cn-hangzhou-internal.aliyuncs.com/all_usr_dataset/%s/%s" 
PAI_MOUNT_PATH_OF_USR_DATASET = "/mnt/pai/data/" # 用户指定（或者默认）数据集在pai容器下的挂载路径
ECS_MOUNT_PATH_OF_USR_DATASET = "/home/evomind/usrdata/" # 用户的数据集挂载在ECS的哪个路径下，具体是格式 /home/evomind/usrdata/usrname/datasetname/
DEFAULT_USR_DATASET_ID = "d-xfobl8zdj3cqdrleqo" # 如果用户没有指定数据集，则使用默认数据集进行训练
# evo-data这个bucket下面的/all_usr_dataset/usrname/datasetname 是用户自己的数据集，所以在拉起pai训练任务的时候，需要将桶的某个路径同时挂载到pai 和 ecs
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
