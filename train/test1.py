from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203.models import (
    CreateJobRequest,
    JobSpec,
    ResourceConfig, GetJobRequest
)
import time  # 用于轮询等待

from aliyun_dlc_client import create_dlc_client

# 初始化客户端
region = 'cn-hangzhou'
client = create_dlc_client(region)
# 任务资源配置
spec = JobSpec(
    type='Worker',
    image=f'dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/pytorch:2.8-gpu-py312-cu130-ubuntu24.04-2b58613b-b2e26b6d',
    pod_count=1,
    resource_config=ResourceConfig(cpu='4', memory='5Gi')
)

# 创建任务请求
req = CreateJobRequest(
    resource_id='quota8xa9l64xn9c',
    workspace_id='604723',
    display_name='ru-dlc-job',
    job_type='TFJob',
    job_specs=[spec],
    user_command='nvcc -V',
)

# 提交任务
response = client.create_job(req)
job_id = response.body.job_id
print(f"任务提交成功！Job ID: {job_id}")
print("开始实时跟踪任务状态...\n")

# ====================== 核心：自动轮询查看状态 ======================
while True:
    job = client.get_job(job_id, GetJobRequest()).body
    status = job.status
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务状态: {status}")

    # 任务结束状态（停止/成功/失败）
    if status in ["Stopped", "Succeeded", "Failed"]:
        break

    time.sleep(5)  # 每5秒查一次

# 最终结果
print("\n===== 任务最终状态 =====")
print(f"Job ID: {job_id}")
print(f"状态: {job.status}")
print(f"执行命令: {job.user_command}")
