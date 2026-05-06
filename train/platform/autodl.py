from __future__ import annotations

from datetime import datetime
import json
import os
import shlex
from pathlib import Path
from typing import Any

from .base import TrainPlatform, build_train_command, config_bool, require_value


def _load_paramiko() -> Any:
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit(
            "Missing AutoDL SSH dependency. Install it with:\n"
            "  python3 -m pip install --user paramiko"
        ) from exc
    return paramiko


class AutoDLPlatform(TrainPlatform):
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        key_path: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path

    def _connection_settings(self) -> tuple[str, int, str, str]:
        host = require_value(self.host or os.environ.get("AUTODL_HOST"), "AUTODL_HOST")
        port = self.port if self.port is not None else int(os.environ.get("AUTODL_PORT", "22"))
        user = self.user or os.environ.get("AUTODL_USER", "root")
        key_path = require_value(self.key_path or os.environ.get("AUTODL_KEY_PATH"), "AUTODL_KEY_PATH")
        return host, port, user, key_path

    def _connect(self) -> Any:
        host, port, user, key_path = self._connection_settings()
        paramiko = _load_paramiko()
        client = paramiko.SSHClient()
        # AutoDL instances are ephemeral; host key changes between rents, so we
        # accept on first connect rather than maintain a known_hosts file.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            port=port,
            username=user,
            key_filename=str(Path(key_path).expanduser()),
            timeout=10,
        )
        return client

    def _exec(self, command: str) -> tuple[int, str, str]:
        client = self._connect()
        try:
            _, stdout, stderr = client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            return (
                exit_status,
                stdout.read().decode("utf-8", errors="replace").strip(),
                stderr.read().decode("utf-8", errors="replace").strip(),
            )
        finally:
            client.close()

    def _job_marker(self, job_id: str) -> str:
        return f"evo_train_{job_id}"

    def submit(self, job_config: dict[str, Any]) -> str:
        command = build_train_command(job_config)
        job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        marker = self._job_marker(job_id)

        if config_bool(job_config, "dry_run"):
            host, port, user, _ = self._connection_settings()
            print(
                json.dumps(
                    {
                        "host": host,
                        "port": port,
                        "user": user,
                        "job_id": job_id,
                        "command": command,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return ""

        launch_script = f"exec -a {shlex.quote(marker)} bash -lc {shlex.quote(command)}"
        remote_command = (
            f"nohup bash -lc {shlex.quote(launch_script)} "
            f">/tmp/{marker}.log 2>&1 < /dev/null & echo {shlex.quote(job_id)}"
        )
        exit_status, stdout, stderr = self._exec(remote_command)
        if exit_status != 0:
            raise RuntimeError(f"AutoDL submit failed: {stderr or stdout or 'unknown error'}")
        return stdout or job_id

    def status(self, job_id: str) -> str:
        marker = self._job_marker(job_id)
        _, stdout, _ = self._exec(
            f"ps aux | grep -F -- {shlex.quote(marker)} | grep -v grep || true"
        )
        return "Running" if stdout else "Unknown"

    def stop(self, job_id: str) -> None:
        marker = self._job_marker(job_id)
        self._exec(
            f"ps aux | grep -F -- {shlex.quote(marker)} | grep -v grep "
            "| awk '{print $2}' | xargs -r kill"
        )
