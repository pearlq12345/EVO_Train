#!/usr/bin/env python3
"""Socket connection layer wired to the training task thread pool."""
from __future__ import annotations

import argparse
import heapq
import itertools
import selectors
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from thread_pool.thread_pool import ThreadPool, TrainTaskEvent
from train.server_function import handle_request_text


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_RECV_BYTES = 4096
DEFAULT_IDLE_TIMEOUT = 120.0


@dataclass
class Client:
    socket: socket.socket
    address: tuple[str, int]
    last_active: float
    idle_deadline: float = 0.0
    read_buffer: str = ""
    closed: bool = False

    @property
    def id(self) -> str:
        """Return a readable client identifier."""
        return format_address(self.address)


TimerHeap = list[tuple[float, int, Client]]


def log(message: str) -> None:
    """Print one connection-layer debug message."""
    print(f"[server_connection] {message}", flush=True)


def format_address(address: tuple[str, int]) -> str:
    """Convert a client address tuple into a readable string."""
    host, port = address
    return f"{host}:{port}"


def make_server_socket(host: str, port: int, backlog: int) -> socket.socket:
    """Create and configure the listening server socket."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(backlog)
    server.setblocking(False)
    return server


def close_client(client: Client, reason: str) -> None:
    """Close one client socket and print the close reason."""
    if client.closed:
        return
    client.closed = True
    try:
        client.socket.close()
    finally:
        log(f"closed {client.id}: {reason}")


def unregister_and_close(selector: selectors.BaseSelector, client: Client, reason: str) -> None:
    """Remove one client socket from the selector and close it."""
    unregister_socket(selector, client.socket)
    close_client(client, reason)


def unregister_socket(selector: selectors.BaseSelector, sock: socket.socket) -> None:
    """Remove one socket from the selector if it is registered."""
    try:
        selector.unregister(sock)
    except (KeyError, ValueError):
        pass


def schedule_idle_timeout(
    timer_heap: TimerHeap,
    timer_counter: itertools.count,
    client: Client,
    idle_timeout: float,
) -> None:
    """Schedule or refresh one client's idle timeout."""
    client.last_active = time.monotonic()
    client.idle_deadline = client.last_active + idle_timeout
    heapq.heappush(timer_heap, (client.idle_deadline, next(timer_counter), client))
    # next(timer_counter)作为备用key进行排序

def get_select_timeout(timer_heap: TimerHeap) -> float | None:
    """Return how long selector.select should wait before the next timeout."""
    while timer_heap:
        deadline, _, client = timer_heap[0]
        if client.closed or deadline != client.idle_deadline:
            heapq.heappop(timer_heap)
            continue
        return max(0.0, deadline - time.monotonic())
    return None

# 维护的是用户长链接，如果时间太长了，用户的链接会被我们主动关闭。
def process_idle_timeouts(selector: selectors.BaseSelector, timer_heap: TimerHeap) -> None:
    """Close all clients whose idle timers have expired."""
    now = time.monotonic()
    while timer_heap:
        deadline, _, client = timer_heap[0]
        if deadline > now:
            return
        heapq.heappop(timer_heap)
        if client.closed or deadline != client.idle_deadline:
            continue
        unregister_and_close(selector, client, "idle timeout")

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
    timer_counter: itertools.count,
    idle_timeout: float,
) -> None:
    """Accept all pending client connections and register them for reads."""
    while True:
        try:
            client_socket, address = server.accept()
        except BlockingIOError:
            return

        active_connections = len(selector.get_map()) - 1
        if active_connections >= max_connections:
            log(f"rejecting {format_address(address)}: max connections reached")
            client_socket.close()
            continue

        client_socket.setblocking(False)
        now = time.monotonic()
        client = Client(socket=client_socket, address=address, last_active=now)
        schedule_idle_timeout(timer_heap, timer_counter, client, idle_timeout)
        selector.register(client_socket, selectors.EVENT_READ, data=client)
        log(f"accepted {client.id} ({active_connections + 1}/{max_connections})")


def read_client(
    selector: selectors.BaseSelector,
    client: Client,
    recv_bytes: int,
    encoding: str,
    timer_heap: TimerHeap,
    timer_counter: itertools.count,
    idle_timeout: float,
) -> TrainTaskEvent | None:
    """Read one client request and return a train task event."""
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

    schedule_idle_timeout(timer_heap, timer_counter, client, idle_timeout)
    client.read_buffer += data.decode(encoding, errors="replace")
    if "\n" not in client.read_buffer:
        return None
        # Todo: 如果这里的数据不完整，或许也需要向用户发送结果？
    request_text, client.read_buffer = client.read_buffer.split("\n", 1)
    request_text = request_text.strip()
    if not request_text:
        return None

    log(f"read event from {client.id}: {request_text}")
    return TrainTaskEvent(
        client_id=client.id,
        request_text=request_text,
        response_callback=make_response_callback(selector, client, encoding),
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
            close_client(client, "server stopping")
            # 这里关闭的是客户端监听socket

def serve(args: argparse.Namespace, pool: ThreadPool) -> None:
    """Run the selector loop and hand read events to the worker pool."""
    selector = selectors.DefaultSelector()
    timer_heap: TimerHeap = []
    timer_counter = itertools.count()
    server = make_server_socket(args.host, args.port, args.max_connections)
    selector.register(server, selectors.EVENT_READ, data=None)
    pool.start()
    log(f"listening on {args.host}:{args.port}, workers={args.workers}, idle_timeout={args.idle_timeout}s")

    try:
        while True:
            # socket 有事件 -> 立刻醒
            # socket 没事件 -> 最多等到最近的 idle timeout
            for key, _ in selector.select(timeout=get_select_timeout(timer_heap)):
                if key.data is None:
                    accept_clients(
                        selector,
                        server,
                        args.max_connections,
                        timer_heap,
                        timer_counter,
                        args.idle_timeout,
                    )
                else:
                    event = read_client(
                        selector,
                        key.data,
                        args.recv_bytes,
                        args.encoding,
                        timer_heap,
                        timer_counter,
                        args.idle_timeout,
                    )
                    if event is not None:
                        pool.submit(event)
            process_idle_timeouts(selector, timer_heap)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        close_registered_sockets(selector)
        selector.close()
        pool.train_task_queue.join()
        pool.stop()


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

    pool = ThreadPool(args.workers, task_handler=handle_request_text)
    serve(args, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
