#!/usr/bin/env python3
"""
Author: Ru-hulu
Date: 2026-05-03

Handle each user request and return the user's current task list.
"""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from typing import Any

from sql_lite.sql_pack import sql_add_user_task, sql_delete_user_task, sql_get_user_all_task, sql_get_user_jobid
from train.send_pai_request import PaiRequest
from train.user_param import UserTrainCmd
TASK_OUTPUT_DIR = "/mnt/usrresult/%s/%s" ## username task_name
CHECKPOINT_OUTPUT_DIR = TASK_OUTPUT_DIR + "/checkpoint" ## username task_name
DOWNLOAD_CHUNK_SIZE = 64 * 1024


def _start_training(request: dict[str, Any], username: str, task_name: str, tasks: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    # TODO：应该先检查是否有同名任务存在，如果已经存在就返回错误
    user_cmd = UserTrainCmd(request).create_train_cmd(username, task_name)
    this_req = PaiRequest(user_cmd)
    this_req.submit_job() ## TODO:没有做兜底逻辑
    if sql_add_user_task(username, task_name, this_req.job_id):
        return "create task success", sql_get_user_all_task(username)
    return "create task failed", tasks


def _query_status(username: str, task_name: str) -> str:
    # TODO: 很多兜底逻辑没有写，如果用户恶意访问，给空的taskname会出问题
    job_id = sql_get_user_jobid(username, task_name)
    if job_id:
        this_req = PaiRequest("", job_id)
        return f"{task_name}: {this_req.query_job()}"
    return f"{task_name}: query status failed, job id dose not exist."


def _stop_training(username: str, task_name: str) -> tuple[str, list[dict[str, str]]]:
    job_id = sql_get_user_jobid(username, task_name)
    if job_id:
        this_req = PaiRequest("", job_id)
        if this_req.query_job() not in {"Succeeded", "Failed", "Stopped"}:            
            this_req.stop_job()
            message = f"{task_name}: stop success."
            print(f"[任务结束成功] {message}")
        else:
            message = f"{task_name}: is finished, no need stop."
            print(f"[任务已经结束，无需停止] {message}")
    else:
        message = f"{task_name}: stop failed, job id does not exist."
        print(f"[job_id不存在] {message}")
    return message, sql_get_user_all_task(username)


def _delete_task(username: str, task_name: str) -> tuple[str, list[dict[str, str]]]:
    job_id = sql_get_user_jobid(username, task_name)
    if job_id:
        this_req = PaiRequest("", job_id)
        this_req.query_job()
        if this_req.status in {"Succeeded", "Failed", "Stopped"}:
            message = f"{task_name}: results are deleted."
            print(f"[任务删除成功] {message}")
            shutil.rmtree(TASK_OUTPUT_DIR % (username, task_name), ignore_errors=True)
            if sql_delete_user_task(username, task_name):
                message = f"{message} Update sql success"
            else:
                message = f"{message} Update sql failed"
        else:
            message = f"{task_name}: is running, please stop it first."
            print(f"[任务运行中，无法删除] {message}")
    else:
        message = f"{task_name}: delete failed, job id does not exist."
        print(f"[job_id不存在] {message}")
    return message, sql_get_user_all_task(username)


def _download_result(username: str, task_name: str) -> dict[str, Any]:
    job_id = sql_get_user_jobid(username, task_name)
    if not job_id:
        message = f"{task_name}: download failed, job id does not exist."
        print(f"[job_id不存在] {message}")
        return {"message": message, "tasks": sql_get_user_all_task(username)}

    this_req = PaiRequest("", job_id)
    this_req.query_job()
    if this_req.status not in {"Succeeded", "Failed", "Stopped"}:
        message = f"{task_name}: is running, please wait until it finishes."
        print(f"[任务运行中，无法下载] {message}")
        return {"message": message, "tasks": sql_get_user_all_task(username)}

    checkpoint_dir = CHECKPOINT_OUTPUT_DIR % (username, task_name)
    if not os.path.isdir(checkpoint_dir):
        message = f"{task_name}: download failed, checkpoint does not exist."
        print(f"[checkpoint不存在] {message}")
        return {"message": message, "tasks": sql_get_user_all_task(username)}

    if not any(filenames for _, _, filenames in os.walk(checkpoint_dir)):
        message = f"{task_name}: download failed, checkpoint is empty."
        print(f"[checkpoint为空] {message}")
        return {"message": message, "tasks": sql_get_user_all_task(username)}

    message = f"{task_name}: download task queued."
    print(f"[结果下载入队] {message}")
    return {
        "message": message,
        "tasks": sql_get_user_all_task(username),
        "_downloadPath": checkpoint_dir,
    }

# 这里设计的时候先考虑用阻塞的方案来解决：每次任务起来以后，只有等到任务创建成功/失败 任务删除成功/失败的时候才会返回。
# 这样设计一方面是考虑当前并发-资源的关系很协调，另一方面是考虑用户体验，可以持续的看到当前创建任务过程的进展。
# 任务创建的过程大概会持续60s左右，主要是镜像比较大。
# 约定：每个用户，不允许有两个相同名字的task_name, 即便一个已经运行结束也不行。只有删除了这个任务，才允许创建相同的task_name
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
        message, tasks = _start_training(request, username, task_name, tasks)
    elif action == "查询状态":
        message = _query_status(username, task_name)
    elif action == "结束训练":
        message, tasks = _stop_training(username, task_name)
    elif action == "删除任务":
        message, tasks = _delete_task(username, task_name)
    elif action == "结果下载":
        return _download_result(username, task_name)
    else:
        message = "invalid action"
    return {"message": message, "tasks": tasks}


def handle_request_text(text: str) -> str:
    """Handle one complete JSON request and return a JSON response string."""
    response = handle_request(text)
    response.pop("_downloadPath", None)
    return json.dumps(response, ensure_ascii=False)


def handle_download_task(event: Any) -> None:
    """Handle one download task event."""
    print(f"[结果下载任务] {event.client_id}: {event.download_path}")
    sock = event.client_socket
    old_timeout = sock.gettimeout()
    try:
        sock.setblocking(True)
        with sock.makefile("wb", buffering=DOWNLOAD_CHUNK_SIZE) as writer, tarfile.open(fileobj=writer, mode="w|") as tar:
            for root, _, filenames in os.walk(event.download_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, event.download_path)
                    try:
                        tar.add(file_path, arcname=arcname, recursive=False)
                    except FileNotFoundError:
                        print(f"[结果下载跳过] file disappeared: {file_path}")
        print(f"[结果下载完成] {event.client_id}: {event.download_path}")
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            pass
