#!/usr/bin/env python3
"""Socket connection layer wired to the training task thread pool."""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import selectors
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from thread_pool.thread_pool import TaskEvent, ThreadPool
from train.server_function import handle_download_task, handle_request


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_RECV_BYTES = 4096
DEFAULT_IDLE_TIMEOUT = 120.0


def log(message: str) -> None:
    """Print one connection-layer debug message."""
    print(f"[server_connection] {message}", flush=True)


def is_download_task(request_text: str) -> bool:
    """Return whether this request is a result-download task."""
    try:
        request = json.loads(request_text)
    except json.JSONDecodeError:
        return False
    return str(request.get("action") or "").strip() == "结果下载"


@dataclass
class Client:
    socket: socket.socket
    address: tuple[str, int]
    last_active: float # 最后一次业务触发的活跃时间
    idle_deadline: float = 0.0 # 由最后一次业务活跃时间计算得到的死亡时间，死亡时间到达以后主线程会杀Client
    read_buffer: str = ""
    closed: bool = False

    @property
    def id(self) -> str:
        """Return a readable client identifier."""
        return self.format_address(self.address)

    @staticmethod
    def format_address(address: tuple[str, int]) -> str:
        """Convert a client address tuple into a readable string."""
        host, port = address
        return f"{host}:{port}"

    def close(self, reason: str) -> None:
        """Close this client socket and print the close reason."""
        if self.closed:
            return
        self.closed = True
        try:
            self.socket.close()
        finally:
            log(f"closed {self.id}: {reason}")


class TimerHeap:
    def __init__(self, idle_timeout: float) -> None:
        self.idle_timeout = idle_timeout
        self._heap: list[tuple[float, int, Client]] = []
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def refresh_client_expire_time(self, client: Client) -> None:
        """Refresh one client's idle deadline and push it into the timer heap."""
        with self._lock:
            client.last_active = time.monotonic()
            client.idle_deadline = client.last_active + self.idle_timeout
            heapq.heappush(self._heap, (client.idle_deadline, next(self._counter), client))

    def get_first_expire_time(self) -> float | None:
        """Return how long selector.select should wait before the next timeout."""
        with self._lock:
            self._drop_stale_entries()
            if not self._heap:
                return None
            deadline, _, _ = self._heap[0]
            return max(0.0, deadline - time.monotonic())

    def pop_expired_clients(self) -> list[Client]:
        """Pop and return clients whose active idle deadlines have expired."""
        expired_clients: list[Client] = []
        now = time.monotonic()
        with self._lock:
            while self._heap:
                deadline, _, client = self._heap[0]
                if deadline > now:
                    break
                heapq.heappop(self._heap)
                if client.closed or deadline != client.idle_deadline:
                    continue
                expired_clients.append(client)
        return expired_clients

    def _drop_stale_entries(self) -> None:
        """Drop heap records that no longer match the client's active deadline."""
        while self._heap:
            deadline, _, client = self._heap[0]
            if not client.closed and deadline == client.idle_deadline:
                return
            heapq.heappop(self._heap)


def make_server_socket(host: str, port: int, backlog: int) -> socket.socket:
    """Create and configure the listening server socket."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(backlog)
    server.setblocking(False)
    return server


def unregister_and_close(selector: selectors.BaseSelector, client: Client, reason: str) -> None:
    """Remove one client socket from the selector and close it."""
    unregister_socket(selector, client.socket)
    client.close(reason)


def unregister_socket(selector: selectors.BaseSelector, sock: socket.socket) -> None:
    """Remove one socket from the selector if it is registered."""
    try:
        selector.unregister(sock)
    except (KeyError, ValueError):
        pass


# 用户的业务被处理完了，现在将处理结果返回给用户
def make_response_callback(
    selector: selectors.BaseSelector,
    client: Client,
    encoding: str,
) -> Callable[[str], None]:
    """Build a callback that sends a worker response and keeps the connection open."""

    def send_response(response_text: str) -> None:
        try:
            if not response_text.endswith("\n"):
                response_text += "\n"
            client.socket.sendall(response_text.encode(encoding))
            log(f"response sent to {client.id}")
        except OSError as exc:
            unregister_and_close(selector, client, f"send error: {exc}")
    return send_response


def accept_clients(
    selector: selectors.BaseSelector,
    server: socket.socket,
    max_connections: int,
    timer_heap: TimerHeap,
) -> None:
    """Accept all pending client connections and register them for reads."""
    while True:
        try:
            client_socket, address = server.accept()
        except BlockingIOError:
            return

        active_connections = len(selector.get_map()) - 1
        if active_connections >= max_connections:
            log(f"rejecting {Client.format_address(address)}: max connections reached")
            client_socket.close()
            continue

        client_socket.setblocking(False)
        now = time.monotonic()
        client = Client(socket=client_socket, address=address, last_active=now)
        timer_heap.refresh_client_expire_time(client)
        selector.register(client_socket, selectors.EVENT_READ, data=client)
        log(f"accepted {client.id} ({active_connections + 1}/{max_connections})")


def read_client(
    selector: selectors.BaseSelector,
    client: Client,
    recv_bytes: int,
    encoding: str,
    timer_heap: TimerHeap,
) -> TaskEvent | None:
    """Read one client request and return a task event."""
    try:
        data = client.socket.recv(recv_bytes)
        # 之所以把数据的读放在主线程，有原因：
        # 当前已经发现sokcet可读，且用户发送的数据非常精简，读起来很快
        # 如果放到子线程中处理，下次epoll检查的时候这个socket可能还是可读，会引起混乱
    except ConnectionResetError:
        unregister_and_close(selector, client, "connection reset")
        return
    except OSError as exc:
        unregister_and_close(selector, client, f"read error: {exc}")
        return

    if not data:
        unregister_and_close(selector, client, "peer closed")
        return

    timer_heap.refresh_client_expire_time(client)
    client.read_buffer += data.decode(encoding, errors="replace")
    if "\n" not in client.read_buffer:
        return None
        # Todo: 如果这里的数据不完整，或许也需要向用户发送结果？
    request_text, client.read_buffer = client.read_buffer.split("\n", 1)
    request_text = request_text.strip()
    if not request_text:
        return None

    log(f"read event from {client.id}: {request_text}")
    return TaskEvent(
        client_id=client.id,
        request_text=request_text,
        response_callback=None,
        client=client,
        timer_heap=timer_heap,
    )


def close_registered_sockets(selector: selectors.BaseSelector) -> None:
    """Close all sockets that are still registered in the selector."""
    for key in list(selector.get_map().values()):
        sock = key.fileobj
        client = key.data
        unregister_socket(selector, sock)
        if client is None:
            sock.close()
            # 此处关闭的是服务端监听socket，就是监听9000端口
        else:
            client.close("server stopping")
            # 这里关闭的是客户端监听socket

# 先不考虑下载业务的情况。假如lite类业务并发非常高，read_client 120s以后任务都没有被处理完。
# 这时server函数就会判断连接已经过期，把连接断开。那么线程池在处理的时候，就会发现连接已经断开了。
# 上述情况是可能存在的，但是我们不考虑。因为任务在队列中120s未被处理，不会有这么高的并发。
def serve(args: argparse.Namespace, pool: ThreadPool) -> None:
    """Run the selector loop and hand read events to the worker pool."""
    selector = selectors.DefaultSelector()
    timer_heap = TimerHeap(args.idle_timeout)
    server = make_server_socket(args.host, args.port, args.max_connections)
    selector.register(server, selectors.EVENT_READ, data=None)
    pool.start_lite()
    pool.start_download()
    log(f"listening on {args.host}:{args.port}, workers={args.workers}, idle_timeout={args.idle_timeout}s")

    try:
        while True:
            # socket 有事件 -> 立刻醒
            # socket 没事件 -> 最多等到最近的 idle timeout
            for key, _ in selector.select(timeout=timer_heap.get_first_expire_time()):
                if key.data is None:
                    accept_clients(selector, server, args.max_connections, timer_heap)
                else:
                    event = read_client(selector, key.data, args.recv_bytes, args.encoding, timer_heap) 
                    # 这里刷新连接的过期时间，所以不会出现连接业务在处理的时候连接被杀掉的情况。
                    if event is not None:
                        event.response_callback = make_response_callback(selector, key.data, args.encoding)
                        if is_download_task(event.request_text):
                            pool.submit_download(event)
                        else:
                            pool.submit_lite(event)
            for expired_client in timer_heap.pop_expired_clients(): ## 这里处理过期链接
                unregister_and_close(selector, expired_client, "idle timeout")
    except KeyboardInterrupt:
        log("stopping")
    finally:
        close_registered_sockets(selector)
        selector.close()
        pool.train_task_queue.join()
        pool.stop_lite()
        pool.stop_download()


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the connection-layer debug server."""
    parser = argparse.ArgumentParser(description="Debug socket reactor with a worker thread pool.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host, default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port, default: {DEFAULT_PORT}")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=DEFAULT_MAX_CONNECTIONS,
        help=f"Maximum active client sockets, default: {DEFAULT_MAX_CONNECTIONS}",
    )
    parser.add_argument(
        "--recv-bytes",
        type=int,
        default=DEFAULT_RECV_BYTES,
        help=f"Bytes to read per socket event, default: {DEFAULT_RECV_BYTES}",
    )
    parser.add_argument("--encoding", default="utf-8", help="Socket text encoding, default: utf-8")
    parser.add_argument("--workers", type=int, choices=(4, 8), default=4, help="Worker thread count, default: 4")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help=f"Close client connections idle for this many seconds, default: {DEFAULT_IDLE_TIMEOUT:.0f}",
    )
    return parser


def main() -> int:
    """Validate arguments and start the connection-layer debug server."""
    args = build_parser().parse_args()
    if args.max_connections < 1:
        print("--max-connections must be at least 1", file=sys.stderr)
        return 2
    if args.recv_bytes < 1:
        print("--recv-bytes must be at least 1", file=sys.stderr)
        return 2
    if args.idle_timeout <= 0:
        print("--idle-timeout must be greater than 0", file=sys.stderr)
        return 2

    pool = ThreadPool(
        args.workers,
        lite_task_handler=handle_request,
        download_task_handler=handle_download_task,
    )
    serve(args, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
