#!/usr/bin/env python3
"""
Author: Ru-hulu
Date: 2026-05-03

Handle each user request and return the user's current task list.
"""
from __future__ import annotations

import json
from typing import Any

from sql_lite.sql_pack import sql_add_user_task, sql_delete_user_task, sql_get_user_all_task, sql_get_user_jobid
from train.send_pai_request import PaiRequest
from train.user_param import UserTrainCmd

# 这里设计的时候先考虑用阻塞的方案来解决：每次任务起来以后，只有等到任务创建成功/失败 任务删除成功/失败的时候才会返回。
# 这样设计一方面是考虑当前并发-资源的关系很协调，另一方面是考虑用户体验，可以持续的看到当前创建任务过程的进展。
# 任务创建的过程大概会持续60s左右，主要是镜像比较大。
def handle_request(text: str) -> dict[str, Any]:
    try:
        request = json.loads(text)
    except json.JSONDecodeError:
        return {"message": "invalid json", "tasks": []}
    # TODO：检查资源，如果资源不足立即返回，这样运行中的线程就都是在处理相关业务。
    username = str(request.get("username") or "").strip()
    task_name = str(request.get("taskName") or "").strip()
    action = str(request.get("action") or "").strip()
    tasks = sql_get_user_all_task(username)
    if action == "任务同步":
        return {"message": "sync success", "tasks": tasks}
    if not username or not task_name:
        return {"message": "invalid request", "tasks": tasks}
    if action == "开始训练":
        # TODO：应该先检查是否有同名任务存在，如果已经存在就返回错误
        user_cmd = UserTrainCmd(request).create_train_cmd()
        this_req = PaiRequest(user_cmd)
        this_req.submit_job() ## TODO:没有做兜底逻辑
        if sql_add_user_task(username, task_name, this_req.job_id):
            message = "create task success"
            tasks = sql_get_user_all_task(username)
        else:
            message = "create task failed"
    elif action == "查询状态":
        # TODO: 很多兜底逻辑没有写，如果用户恶意访问，给空的taskname会出问题
        job_id = sql_get_user_jobid(username, task_name)
        if job_id:
            this_req = PaiRequest("", job_id)
            message = f"{task_name}: {this_req.query_job()}"
        else:
            message = f"{task_name}: query status failed, job id dose not exist."
    elif action == "结束训练":
        job_id = sql_get_user_jobid(username, task_name)
        if job_id:
            this_req = PaiRequest("", job_id)
            this_req.stop_job()
            message = f"{task_name}: stop success."
            print(f"[成功查询job_id] {message}")
        else:
            message = f"{task_name}: stop failed, job id does not exist."
            print(f"[job_id不存在] {message}")
        if sql_delete_user_task(username, task_name):
            message = f"{message} Update sql success"
        else:
            message = f"{message} Update sql failed"
        tasks = sql_get_user_all_task(username)
        # TODO 结束停止训练任务，但是不要杀掉容器
    elif action == "删除任务":
        message = "delete action"
        # TODO 需要删除任务所有的相关存储空间，这里需要对用户的存储空间进行约定。        
    else:
        message = "invalid action"
    return {"message": message, "tasks": tasks}


def handle_request_text(text: str) -> str:
    """Handle one complete JSON request and return a JSON response string."""
    return json.dumps(handle_request(text), ensure_ascii=False)
