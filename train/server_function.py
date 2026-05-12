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
import subprocess
import tarfile
import time
from typing import Any, TYPE_CHECKING

from sql_lite.sql_pack import sql_add_user_task, sql_delete_user_task, sql_get_user_all_task, sql_get_user_jobid
from train.send_pai_request import PaiRequest
from train.user_param import UserTrainCmd

if TYPE_CHECKING:
    from thread_pool.thread_pool import TaskEvent

TASK_OUTPUT_DIR = "/mnt/usrresult/%s/%s" ## username task_name
CHECKPOINT_OUTPUT_DIR = TASK_OUTPUT_DIR + "/checkpoint" ## username task_name
DOWNLOAD_CHUNK_SIZE = 64 * 1024
DOWNLOAD_TIMER_REFRESH_SECONDS = 60
TAR_BLOCK_SIZE = 512
TAR_RECORD_SIZE = TAR_BLOCK_SIZE * 20


def _run_debug_command(command: list[str]) -> None:
    print(f"[checkpoint不存在-debug] $ {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())


def _debug_missing_checkpoint_path(path: str) -> None:
    parent_paths = [
        "/mnt",
        "/mnt/usrresult",
        os.path.dirname(os.path.dirname(path)),
        os.path.dirname(path),
        path,
    ]
    print(f"[checkpoint不存在-debug] pid={os.getpid()} cwd={os.getcwd()}")
    for command in [
        ["hostname"],
        ["id"],
        ["readlink", "/proc/self/ns/mnt"],
        ["cat", "/proc/self/cgroup"],
        ["df", "-hT", "/mnt/usrresult"],
    ]:
        _run_debug_command(command)
    for parent_path in dict.fromkeys(parent_paths):
        _run_debug_command(["ls", "-lah", parent_path])
    _run_debug_command(["mount"])


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


def get_download_path(username: str, task_name: str) -> str:
    # job_id = sql_get_user_jobid(username, task_name)
    # if not job_id:
    #     print("job_id不存在")
    #     return "job_id does not exist." + "|" + ""

    # this_req = PaiRequest("", job_id)
    # this_req.query_job()
    # if this_req.status != "Running":
    #     message = f"{task_name}: is not running, please wait until it finishes."
    #     print(f"[任务停止，容器不存在，无法下载] {message}")
    #     return f"{message}" + "|" + ""
    checkpoint_dir = CHECKPOINT_OUTPUT_DIR % (username, task_name)
    if not os.path.isdir(checkpoint_dir):
        message = f"{task_name}: download failed, checkpoint does not exist."
        print(f"[checkpoint不存在] {checkpoint_dir}")
        _debug_missing_checkpoint_path(checkpoint_dir)
        return f"{message}" + "|" + ""

    if not any(filenames for _, _, filenames in os.walk(checkpoint_dir)):
        message = f"{task_name}: download failed, checkpoint is empty."
        print(f"[checkpoint为空] {message}")
        return f"{message}" + "|" + ""

    message = f"{task_name}: download task queued."
    print(f"[结果下载入队] {message}")
    return f"{message}" + "|" + checkpoint_dir


def _tar_stream_size(path: str) -> int:
    tar_size = 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                size = os.path.getsize(file_path)
            except FileNotFoundError:
                continue
            tar_size += TAR_BLOCK_SIZE
            tar_size += ((size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE
    tar_size += TAR_BLOCK_SIZE * 2
    tar_size += (TAR_RECORD_SIZE - (tar_size % TAR_RECORD_SIZE)) % TAR_RECORD_SIZE
    return tar_size


def _checkpoint_id(name: str) -> str:
    if name.isdigit():
        return str(int(name))
    return name


def _list_checkpoints(checkpoint_dir: str) -> list[dict[str, Any]]:
    checkpoints_dir = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return []
    checkpoints: list[dict[str, Any]] = []
    for name in sorted(os.listdir(checkpoints_dir)):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(path):
            continue
        checkpoints.append({
            "id": _checkpoint_id(name),
            "name": name,
            "downloadSize": _tar_stream_size(path),
        })
    return checkpoints


def _parse_download_list(value: Any) -> set[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value or "").split(",")
    return {str(item).strip() for item in raw_items if str(item).strip()}


def _download_all_requested(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _selected_download_roots(download_path: str, request: dict[str, Any]) -> tuple[str, list[str]]:
    if _download_all_requested(request.get("downloadAll", True)):
        return "download all", [download_path]

    requested_ids = _parse_download_list(request.get("downloadList"))
    if not requested_ids:
        return "download failed, no checkpoint selected.", []

    selected_roots: list[str] = []
    checkpoints_dir = os.path.join(download_path, "checkpoints")
    for checkpoint in _list_checkpoints(download_path):
        if str(checkpoint["id"]) not in requested_ids:
            continue
        path = os.path.join(checkpoints_dir, str(checkpoint["name"]))
        if os.path.isdir(path):
            selected_roots.append(path)
    if not selected_roots:
        return "download failed, selected checkpoint does not exist.", []
    return "download selected", selected_roots


def get_download_directory(username: str, task_name: str) -> dict[str, Any]:
    response = get_download_path(username, task_name)
    message, _, download_path = response.partition("|")
    if not download_path:
        return {
            "message": message or "download path does not exist.",
            "downloadSize": 0,
            "checkpoints": [],
        }
    return {
        "message": "download directory ready",
        "downloadSize": _tar_stream_size(download_path),
        "checkpoints": _list_checkpoints(download_path),
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
    elif action == "查询下载目录":
        return get_download_directory(username, task_name)
    else:
        message = "invalid action"
    return {"message": message, "tasks": tasks}


def handle_download_task(event: "TaskEvent") -> dict[str, str]:
    try:
        request = json.loads(event.request_text)
    except json.JSONDecodeError:
        return {"message": "invalid json"}

    username = str(request.get("username") or "").strip()
    task_name = str(request.get("taskName") or "").strip()
    if not username or not task_name:
        print(f"[结果下载失败] invalid request: {event.request_text}")
        return {"message": "invalid request"}

    response = get_download_path(username, task_name)
    message, _, download_path = response.partition("|")
    if not download_path:
        print(f"[结果下载失败] {message or 'download path does not exist.'}")
        return {"message": message or "download path does not exist."}
    select_message, selected_roots = _selected_download_roots(download_path, request)
    if not selected_roots:
        print(f"[结果下载失败] {select_message}")
        return {"message": select_message}

    print(f"[结果下载开始] {event.client_id}: {download_path}, {select_message}: {selected_roots}")
    sock = event.client.socket
    old_timeout = sock.gettimeout()
    event.refresh_client_expire_time()
    last_refresh_time = time.monotonic()
    try:
        sock.setblocking(True)
        with sock.makefile("wb", buffering=DOWNLOAD_CHUNK_SIZE) as writer, tarfile.open(fileobj=writer, mode="w|") as tar:
            for selected_root in selected_roots:
                for root, _, filenames in os.walk(selected_root):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        arcname = os.path.relpath(file_path, download_path)
                        try:
                            tar.add(file_path, arcname=arcname, recursive=False)
                        except FileNotFoundError:
                            print(f"[结果下载跳过] file disappeared: {file_path}")
                        delta_T = time.monotonic() - last_refresh_time
                        if delta_T >= DOWNLOAD_TIMER_REFRESH_SECONDS:
                            event.refresh_client_expire_time()
                            last_refresh_time = time.monotonic()
        print(f"[结果下载完成] {event.client_id}: {download_path}")
        return {"message": message}
    finally:
        event.refresh_client_expire_time()
        try:
            sock.settimeout(old_timeout)
        except OSError:
            pass
