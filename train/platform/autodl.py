from __future__ import annotations

from datetime import UTC, datetime
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

    def _job_marker(self, job_id: str) -> str:
        return f"evo_train_{job_id}"

    def _job_pid_file(self, job_id: str) -> str:
        return f"/tmp/{self._job_marker(job_id)}.pid"

    def _job_log_file(self, job_id: str) -> str:
        return f"/tmp/{self._job_marker(job_id)}.log"

    def _job_exit_file(self, job_id: str) -> str:
        return f"/tmp/{self._job_marker(job_id)}.exit"

    def _job_stop_file(self, job_id: str) -> str:
        return f"/tmp/{self._job_marker(job_id)}.stopped"

    def _working_directory(self, job_config: dict[str, Any]) -> str | None:
        workdir = job_config.get("workdir") or os.environ.get("AUTODL_WORKDIR")
        if workdir is None:
            return None
        text = str(workdir).strip()
        return text or None

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

    def _build_remote_command(self, job_id: str, command: str, workdir: str | None) -> str:
        pid_file = self._job_pid_file(job_id)
        log_file = self._job_log_file(job_id)
        exit_file = self._job_exit_file(job_id)
        stop_file = self._job_stop_file(job_id)
        remote_parts = [
            "status=0",
            f"echo $$ > {shlex.quote(pid_file)}",
            f"rm -f {shlex.quote(exit_file)} {shlex.quote(stop_file)}",
        ]
        if workdir:
            remote_parts.append(f"cd {shlex.quote(workdir)} || status=$?")
        remote_parts.extend(
            [
                'if [ "$status" -eq 0 ]; then export PYTHONUNBUFFERED=1; fi',
                f'if [ "$status" -eq 0 ]; then bash -lc {shlex.quote(command)}; status=$?; fi',
                f"echo $status > {shlex.quote(exit_file)}",
                f"rm -f {shlex.quote(pid_file)}",
            ]
        )
        launch_script = "; ".join(remote_parts)
        return (
            f"nohup bash -lc {shlex.quote(launch_script)} "
            f">{shlex.quote(log_file)} 2>&1 < /dev/null & "
            f"echo {shlex.quote(job_id)}"
        )

    def submit(self, job_config: dict[str, Any]) -> str:
        command = build_train_command(job_config)
        job_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        workdir = self._working_directory(job_config)

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
                        "workdir": workdir or "",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return ""

        remote_command = self._build_remote_command(job_id, command, workdir)
        exit_status, stdout, stderr = self._exec(remote_command)
        if exit_status != 0:
            raise RuntimeError(f"AutoDL submit failed: {stderr or stdout or 'unknown error'}")
        return stdout or job_id

    def status(self, job_id: str) -> str:
        return self.metadata(job_id)["status"]

    def metadata(self, job_id: str) -> dict[str, str]:
        pid_file = self._job_pid_file(job_id)
        exit_file = self._job_exit_file(job_id)
        stop_file = self._job_stop_file(job_id)
        log_file = self._job_log_file(job_id)
        exit_status, stdout, _ = self._exec(
            f"if [ -f {shlex.quote(pid_file)} ] && kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null; then "
            "status=Running; "
            f"elif [ -f {shlex.quote(stop_file)} ]; then status=STOPPED; "
            f"elif [ -f {shlex.quote(exit_file)} ]; then "
            f"code=$(cat {shlex.quote(exit_file)}); "
            'if [ "$code" = "0" ]; then status=Succeeded; else status=Failed; fi; '
            "else status=Unknown; fi; "
            'printf "__STATUS__=%s\\n" "$status"; '
            f'if [ "$status" = "Failed" ] && [ -f {shlex.quote(log_file)} ]; then '
            f"tail -n 20 {shlex.quote(log_file)}; "
            "fi"
        )
        if exit_status != 0:
            return {"status": "Unknown", "last_error": ""}

        lines = stdout.splitlines()
        status_line = lines[0] if lines else "__STATUS__=Unknown"
        status = status_line.split("=", 1)[1] if "=" in status_line else "Unknown"
        last_error = "\n".join(line for line in lines[1:] if line.strip())
        return {"status": status or "Unknown", "last_error": last_error}

    def stop(self, job_id: str) -> None:
        pid_file = self._job_pid_file(job_id)
        stop_file = self._job_stop_file(job_id)
        self._exec(
            f"if [ -f {shlex.quote(pid_file)} ]; then "
            f"kill $(cat {shlex.quote(pid_file)}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(pid_file)}; "
            "fi; "
            f"echo STOPPED > {shlex.quote(stop_file)}"
        )
