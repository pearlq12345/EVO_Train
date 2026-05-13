#!/usr/bin/env python3
"""
Author: Ru-hulu
Date: 2026-05-03

This module implements a thread pool with 4 or 8 worker threads
to consume and process the queued request from users safely and efficiently.
It supports task submission, concurrent execution, and graceful shutdown.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


STOP_EVENT = object()
QUEUE_LIMIT_FACTOR = 3


@dataclass
class TaskEvent:
    client_id: str
    request_text: str
    response_callback: Callable[[str], None] | None = None
    client: Any | None = None
    timer_heap: Any | None = None

    def refresh_client_expire_time(self) -> None:
        self.timer_heap.refresh_client_expire_time(self.client)


TrainTaskEvent = TaskEvent


def log(message: str) -> None:
    """Print one thread-pool debug message."""
    print(f"[thread_pool] {message}", flush=True)


def fake_lite_task_handler(request_text: str) -> str:
    """Return a temporary debug response for one request."""
    time.sleep(0.2)
    return f"debug response for: {request_text}"


def fake_download_task_handler(event: TaskEvent) -> None:
    """Temporary debug handler for one download event."""


def server_busy_response(queue_size: int) -> str:
    """Return a JSON response when the corresponding worker queue is too long."""
    return json.dumps(
        {"message": f"当前服务器忙，有{queue_size}个任务在前面，请稍后重试", "tasks": []},
        ensure_ascii=False,
    )


class ThreadPool:
    def __init__(
        self,
        workers: int,
        task_handler: Callable[[str], dict[str, Any] | str] | None = None,
        lite_task_handler: Callable[[str], dict[str, Any] | str] = fake_lite_task_handler,
        download_task_handler: Callable[[TaskEvent], None] = fake_download_task_handler,
        download_workers: int = 2,
    ) -> None:
        """Create a 4-thread or 8-thread worker pool."""
        if workers not in (4, 8):
            raise ValueError("workers must be 4 or 8")
        if download_workers < 1:
            raise ValueError("download_workers must be at least 1")
        self.workers = workers
        self.download_workers = download_workers
        self.lite_task_handler = task_handler or lite_task_handler
        self.download_task_handler = download_task_handler
        self.train_task_queue: queue.Queue[TaskEvent | object] = queue.Queue()
        self.lite_threads: list[threading.Thread] = [] # 用来处理业务请求，比如任务的管理等
        self.download_task_queue: queue.Queue[TaskEvent | object] = queue.Queue()
        self.download_threads: list[threading.Thread] = [] # 用来处理IO密集的请求，占用时间较长。

    def start_lite(self) -> None:
        """Start all lite worker threads."""
        for index in range(self.workers):
            thread = threading.Thread(target=self._lite_worker_loop, args=(index,), daemon=True)
            thread.start()
            self.lite_threads.append(thread)
        log(f"started {self.workers} lite worker threads")

    def start_download(self) -> None:
        """Start all download worker threads."""
        for index in range(self.download_workers):
            thread = threading.Thread(target=self._download_worker_loop, args=(index,), daemon=True)
            thread.start()
            self.download_threads.append(thread)
        log(f"started {self.download_workers} download worker threads")

    def submit_lite(self, event: TaskEvent) -> None:
        """Push one training event into the worker queue."""
        queue_size = self.train_task_queue.qsize()
        if queue_size >= self.workers * QUEUE_LIMIT_FACTOR:
            if event.response_callback is not None:
                event.response_callback(server_busy_response(queue_size))
            return
        self.train_task_queue.put(event)
        log(f"queued event from {event.client_id}: {event.request_text}")

    def submit_download(self, event: TaskEvent) -> None:
        """Push one download event into the download queue."""
        queue_size = self.download_task_queue.qsize()
        if queue_size >= self.download_workers * QUEUE_LIMIT_FACTOR:
            if event.response_callback is not None:
                event.response_callback(server_busy_response(queue_size))
            return
        self.download_task_queue.put(event)

    def stop_lite(self) -> None:
        """Stop all lite worker threads after queued work is done."""
        for _ in self.lite_threads:
            self.train_task_queue.put(STOP_EVENT)
        for thread in self.lite_threads:
            thread.join()
        log("all lite worker threads stopped")

    def stop_download(self) -> None:
        """Stop all download worker threads after queued work is done."""
        for _ in self.download_threads:
            self.download_task_queue.put(STOP_EVENT)
        for thread in self.download_threads:
            thread.join()
        log("all download worker threads stopped")

    # 业务线程的死循环
    def _lite_worker_loop(self, worker_id: int) -> None:
        """Continuously consume queued events in one worker thread."""
        log(f"lite-worker-{worker_id} ready")
        while True:
            event = self.train_task_queue.get() # thread safe no need lock
            try:
                if event is STOP_EVENT:
                    log(f"lite-worker-{worker_id} stopping")
                    return
                self._lite_handle_event(worker_id, event)
            finally:
                self.train_task_queue.task_done()

    # 下载线程的死循环
    def _download_worker_loop(self, worker_id: int) -> None:
        """Continuously consume queued download events in one worker thread."""
        log(f"download-worker-{worker_id} ready")
        while True:
            event = self.download_task_queue.get() # thread safe no need lock
            try:
                if event is STOP_EVENT: # 当前线程从下载任务队列中拿一个任务，处理这个任务。
                    log(f"download-worker-{worker_id} stopping")
                    return
                self._download_handle_event(worker_id, event)
            finally:
                self.download_task_queue.task_done()

    # 具体的业务函数在这里
    def _handle_event(self, worker_id: int, event: TaskEvent) -> None:
        """Backward-compatible alias for the lite handler."""
        self._lite_handle_event(worker_id, event)

    def _lite_handle_event(self, worker_id: int, event: TaskEvent) -> None:
        """Run the task handler and optionally return its response."""
        log(f"worker-{worker_id} handling {event.client_id}: {event.request_text}")
        try:
            response = self.lite_task_handler(event.request_text)
        except BaseException as exc:
            response = {"message": f"internal error: {exc}", "tasks": []}
            log(f"worker-{worker_id} failed {event.client_id}: {exc}")
        response_text = self._prepare_response(response)
        if event.response_callback is not None:
            event.response_callback(response_text)
        log(f"worker-{worker_id} finished {event.client_id}")

    def _prepare_response(self, response: dict[str, Any] | str) -> str:
        """Serialize a lite response."""
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    # 具体的下载业务函数在这里
    def _download_handle_event(self, worker_id: int, event: TaskEvent) -> None:
        try:
            response = self.download_task_handler(event)
        except BaseException as exc:
            response = {"message": f"internal error: {exc}"}
            log(f"download-worker-{worker_id} failed {event.client_id}: {exc}")
        response_text = self._prepare_response(response)
        if event.response_callback is not None:
            event.response_callback(response_text)
        log(f"download-worker-{worker_id} finished {event.client_id}")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for thread-pool debugging."""
    parser = argparse.ArgumentParser(description="Debug training task thread pool.")
    parser.add_argument(
        "--workers",
        type=int,
        choices=(4, 8),
        default=4,
        help="Worker thread count, default: 4",
    )
    return parser
# def main() -> int:
#     """Run a standalone thread-pool debug demo."""
#     args = build_parser().parse_args()
#     pool = ThreadPool(args.workers)
#     pool.start()

#     for index in range(10):
#         pool.submit_lite(
#             TaskEvent(
#                 client_id=f"client-{index}",
#                 request_text=f'{{"username":"user-{index % 3}","action":"任务同步"}}',
#             )
#         )

#     pool.train_task_queue.join()
#     pool.stop()
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
