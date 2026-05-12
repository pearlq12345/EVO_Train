from __future__ import annotations

from datetime import UTC, datetime
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from .base import TrainPlatform, build_train_command, config_bool, optional_string, require_value


DEFAULT_API_BASE_URL = "https://api.autodl.com"
DEFAULT_JOB_ROOT = "/root/autodl-tmp/evo_train/jobs"
JOB_REF_SEPARATOR = "::"


def _load_paramiko() -> Any:
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("Missing AutoDL SSH dependency. Install it with: python3 -m pip install paramiko") from exc
    return paramiko


class AutoDLApiClient:
    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = require_value(token or os.environ.get("AUTODL_TOKEN"), "AUTODL_TOKEN")
        self.base_url = (base_url or os.environ.get("AUTODL_API_BASE") or DEFAULT_API_BASE_URL).rstrip("/")

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        method = method.upper()
        url = f"{self.base_url}{path}"
        payload = None
        headers = {"Authorization": self.token}
        if method == "GET" and body:
            url = f"{url}?{urlparse.urlencode(body)}"
        elif method == "POST" and body is None:
            payload = b""
        elif body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urlrequest.Request(
            url,
            data=payload,
            method=method,
            headers=headers,
        )
        try:
            with urlrequest.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AutoDL API {method} {path} failed: HTTP {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"AutoDL API {method} {path} failed: {exc.reason}") from exc
        try:
            payload_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AutoDL API {method} {path} returned invalid JSON: {raw}") from exc
        if payload_obj.get("code") != "Success":
            raise RuntimeError(payload_obj.get("msg") or f"AutoDL API returned {payload_obj.get('code')}")
        return payload_obj

    def wallet_balance(self) -> dict[str, str]:
        payload = self._request("POST", "/api/v1/dev/wallet/balance")
        data = payload.get("data") or {}
        return {
            "assets": str(data.get("assets") or 0),
            "accumulate": str(data.get("accumulate") or 0),
            "voucherBalance": str(data.get("voucher_balance") or 0),
        }

    def create_instance(self, job_config: dict[str, Any]) -> str:
        body: dict[str, Any] = {
            "req_gpu_amount": int(job_config.get("gpu_count") or os.environ.get("AUTODL_GPU_COUNT", "1")),
            "gpu_spec_uuid": require_value(
                optional_string(job_config, "autodl_gpu_spec_uuid") or os.environ.get("AUTODL_GPU_SPEC_UUID"),
                "AUTODL_GPU_SPEC_UUID",
            ),
            "image_uuid": require_value(
                optional_string(job_config, "autodl_image_uuid") or os.environ.get("AUTODL_IMAGE_UUID"),
                "AUTODL_IMAGE_UUID",
            ),
            "expand_system_disk_by_gb": int(
                job_config.get("expand_system_disk_by_gb") or os.environ.get("AUTODL_EXPAND_SYSTEM_DISK_GB", "0")
            ),
        }
        data_centers = optional_string(job_config, "autodl_data_centers") or os.environ.get("AUTODL_DATA_CENTER_LIST")
        if data_centers:
            body["data_center_list"] = [item.strip() for item in data_centers.split(",") if item.strip()]
        payload = self._request("POST", "/api/v1/dev/instance/pro/create", body)
        data = payload.get("data") or {}
        instance_uuid = data.get("instance_uuid") or data.get("uuid")
        if not instance_uuid:
            raise RuntimeError(f"AutoDL create response did not include instance uuid: {data}")
        return str(instance_uuid)

    def status(self, instance_uuid: str) -> str:
        payload = self._request("GET", "/api/v1/dev/instance/pro/status", {"instance_uuid": instance_uuid})
        return str(payload.get("data") or "unknown")

    def power_on(self, instance_uuid: str, start_command: str | None = None) -> None:
        body = {"instance_uuid": instance_uuid, "payload": "gpu"}
        if start_command:
            body["start_command"] = start_command
        self._request("POST", "/api/v1/dev/instance/pro/power_on", body)

    def power_off(self, instance_uuid: str) -> None:
        self._request("POST", "/api/v1/dev/instance/pro/power_off", {"instance_uuid": instance_uuid})

    def release(self, instance_uuid: str) -> None:
        self._request("POST", "/api/v1/dev/instance/pro/release", {"instance_uuid": instance_uuid})


class AutoDLPlatform(TrainPlatform):
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        key_path: str | None = None,
        api_client: AutoDLApiClient | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path
        self.api_client = api_client

    def _connection_settings(self) -> tuple[str, int, str, str]:
        host = require_value(self.host or os.environ.get("AUTODL_HOST"), "AUTODL_HOST")
        port = self.port if self.port is not None else int(os.environ.get("AUTODL_PORT", "22"))
        user = self.user or os.environ.get("AUTODL_USER", "root")
        key_path = require_value(self.key_path or os.environ.get("AUTODL_KEY_PATH"), "AUTODL_KEY_PATH")
        return host, port, user, key_path

    def _api(self) -> AutoDLApiClient:
        if self.api_client is None:
            self.api_client = AutoDLApiClient()
        return self.api_client

    def _job_ref(self, instance_uuid: str, runner_job_id: str) -> str:
        return f"{instance_uuid}{JOB_REF_SEPARATOR}{runner_job_id}"

    def _split_job_ref(self, job_id: str) -> tuple[str | None, str]:
        if JOB_REF_SEPARATOR not in job_id:
            return None, job_id
        instance_uuid, runner_job_id = job_id.split(JOB_REF_SEPARATOR, 1)
        return instance_uuid or None, runner_job_id

    def _managed_enabled(self, job_config: dict[str, Any]) -> bool:
        return config_bool(job_config, "autodl_managed") or bool(os.environ.get("AUTODL_TOKEN"))

    def _ensure_platform_balance(self, job_config: dict[str, Any]) -> None:
        if not self._managed_enabled(job_config):
            return
        minimum_assets = int(os.environ.get("AUTODL_MIN_ASSETS", "0"))
        if minimum_assets <= 0:
            return
        balance = self._api().wallet_balance()
        assets = int(balance["assets"])
        if assets < minimum_assets:
            raise RuntimeError(f"AutoDL platform balance is low: assets={assets}, minimum={minimum_assets}")

    def _ensure_instance_running(self, job_config: dict[str, Any]) -> str:
        instance_uuid = optional_string(job_config, "autodl_instance_uuid") or os.environ.get("AUTODL_INSTANCE_UUID")
        if not instance_uuid:
            instance_uuid = self._api().create_instance(job_config)
        status = self._api().status(instance_uuid)
        if status == "running":
            return instance_uuid
        self._api().power_on(instance_uuid, optional_string(job_config, "autodl_start_command"))
        timeout = int(job_config.get("timeout") or os.environ.get("AUTODL_POWER_ON_TIMEOUT", "600"))
        interval = int(job_config.get("interval") or os.environ.get("AUTODL_POWER_ON_INTERVAL", "10"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._api().status(instance_uuid)
            if status == "running":
                return instance_uuid
            time.sleep(interval)
        raise RuntimeError(f"AutoDL instance {instance_uuid} did not become running, last status: {status}")

    def _job_marker(self, job_id: str) -> str:
        return f"evo_train_{job_id}"

    def _job_dir(self, job_id: str) -> str:
        root = os.environ.get("AUTODL_JOB_ROOT", DEFAULT_JOB_ROOT)
        return f"{root.rstrip('/')}/{self._job_marker(job_id)}"

    def _job_pid_file(self, job_id: str) -> str:
        return f"{self._job_dir(job_id)}/pid"

    def _job_log_file(self, job_id: str) -> str:
        return f"{self._job_dir(job_id)}/run.log"

    def _job_exit_file(self, job_id: str) -> str:
        return f"{self._job_dir(job_id)}/exit_code"

    def _job_stop_file(self, job_id: str) -> str:
        return f"{self._job_dir(job_id)}/stopped"

    def _job_artifact_file(self, job_id: str) -> str:
        return f"{self._job_dir(job_id)}/artifact.tar.gz"

    def _working_directory(self, job_config: dict[str, Any]) -> str | None:
        workdir = job_config.get("workdir") or os.environ.get("AUTODL_WORKDIR")
        if workdir is None:
            return None
        text = str(workdir).strip()
        return text or None

    def _connect(self) -> Any:
        host, port, user, key_path = self._connection_settings()
        paramiko = _load_paramiko()
        client = paramiko.SSHClient()
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
        job_dir = self._job_dir(job_id)
        pid_file = self._job_pid_file(job_id)
        log_file = self._job_log_file(job_id)
        exit_file = self._job_exit_file(job_id)
        stop_file = self._job_stop_file(job_id)
        remote_parts = [
            "status=0",
            f"mkdir -p {shlex.quote(job_dir)}",
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
        self._ensure_platform_balance(job_config)
        if config_bool(job_config, "dry_run"):
            host, port, user, _ = self._connection_settings()
            print(json.dumps({"host": host, "port": port, "user": user, "job_id": job_id, "command": command}))
            return ""
        instance_uuid = ""
        if self._managed_enabled(job_config):
            instance_uuid = self._ensure_instance_running(job_config)
        exit_status, stdout, stderr = self._exec(self._build_remote_command(job_id, command, workdir))
        if exit_status != 0:
            raise RuntimeError(f"AutoDL submit failed: {stderr or stdout or 'unknown error'}")
        runner_job_id = stdout or job_id
        if instance_uuid:
            return self._job_ref(instance_uuid, runner_job_id)
        return runner_job_id

    def metadata(self, job_id: str) -> dict[str, str]:
        instance_uuid, runner_job_id = self._split_job_ref(job_id)
        if instance_uuid:
            instance_status = self._api().status(instance_uuid)
            if instance_status != "running":
                return {"status": f"Instance:{instance_status}", "last_error": ""}
        pid_file = self._job_pid_file(runner_job_id)
        exit_file = self._job_exit_file(runner_job_id)
        stop_file = self._job_stop_file(runner_job_id)
        log_file = self._job_log_file(runner_job_id)
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
        return {
            "status": status or "Unknown",
            "last_error": "\n".join(line for line in lines[1:] if line.strip()),
            "log_path": log_file,
        }

    def stop(self, job_id: str) -> None:
        instance_uuid, runner_job_id = self._split_job_ref(job_id)
        pid_file = self._job_pid_file(runner_job_id)
        stop_file = self._job_stop_file(runner_job_id)
        self._exec(
            f"if [ -f {shlex.quote(pid_file)} ]; then "
            f"kill $(cat {shlex.quote(pid_file)}) 2>/dev/null || true; "
            f"rm -f {shlex.quote(pid_file)}; "
            "fi; "
            f"echo STOPPED > {shlex.quote(stop_file)}"
        )
        if instance_uuid and os.environ.get("AUTODL_POWER_OFF_ON_STOP", "").strip().lower() in {"1", "true", "yes"}:
            self._api().power_off(instance_uuid)
        if instance_uuid and os.environ.get("AUTODL_RELEASE_ON_STOP", "").strip().lower() in {"1", "true", "yes"}:
            self._api().release(instance_uuid)

    def _prepare_artifact(self, runner_job_id: str, artifact_path: str) -> str:
        tar_file = self._job_artifact_file(runner_job_id)
        exit_status, _, stderr = self._exec(
            f"mkdir -p {shlex.quote(self._job_dir(runner_job_id))}; "
            f"if [ -d {shlex.quote(artifact_path)} ]; then "
            f"tar -C {shlex.quote(artifact_path)} -czf {shlex.quote(tar_file)} .; "
            f"elif [ -f {shlex.quote(artifact_path)} ]; then "
            f"tar -C {shlex.quote(str(Path(artifact_path).parent))} "
            f"-czf {shlex.quote(tar_file)} {shlex.quote(Path(artifact_path).name)}; "
            "else exit 44; fi"
        )
        if exit_status == 44:
            raise RuntimeError(f"AutoDL artifact path does not exist: {artifact_path}")
        if exit_status != 0:
            raise RuntimeError(f"AutoDL artifact tar failed: {stderr or 'unknown error'}")
        return tar_file

    def download_artifact_chunk(
        self,
        job_id: str,
        artifact_path: str,
        *,
        offset: int = 0,
        chunk_size: int = 1024 * 1024,
    ) -> dict[str, str | int | bool]:
        _, runner_job_id = self._split_job_ref(job_id)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if offset == 0:
            tar_file = self._prepare_artifact(runner_job_id, artifact_path)
        else:
            tar_file = self._job_artifact_file(runner_job_id)
            exit_status, _, _ = self._exec(f"test -f {shlex.quote(tar_file)}")
            if exit_status != 0:
                raise RuntimeError("AutoDL artifact archive is not prepared; retry with offset=0")
        exit_status, stdout, stderr = self._exec(
            f"size=$(wc -c < {shlex.quote(tar_file)} | tr -d ' '); "
            f"data=$(dd if={shlex.quote(tar_file)} bs=1 skip={offset} count={chunk_size} 2>/dev/null | base64 | tr -d '\\n'); "
            'printf "__SIZE__=%s\\n__DATA__=%s\\n" "$size" "$data"'
        )
        if exit_status != 0:
            raise RuntimeError(f"AutoDL artifact read failed: {stderr or stdout or 'unknown error'}")
        lines = stdout.splitlines()
        size_line = next((line for line in lines if line.startswith("__SIZE__=")), "__SIZE__=0")
        data_line = next((line for line in lines if line.startswith("__DATA__=")), "__DATA__=")
        total_bytes = int(size_line.split("=", 1)[1] or "0")
        data = data_line.split("=", 1)[1] if "=" in data_line else ""
        next_offset = min(total_bytes, offset + chunk_size)
        return {
            "artifactPath": artifact_path,
            "archivePath": tar_file,
            "offset": offset,
            "nextOffset": next_offset,
            "chunkSize": chunk_size,
            "totalBytes": total_bytes,
            "done": next_offset >= total_bytes,
            "dataBase64": data,
        }
