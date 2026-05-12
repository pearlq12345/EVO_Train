from __future__ import annotations

import argparse
from typing import Any

from train import start_train

from .base import TrainPlatform


class AliyunDLCPlatform(TrainPlatform):
    def __init__(self, *, region_id: str = "cn-hangzhou") -> None:
        self.region_id = region_id

    def _args(self, job_config: dict[str, Any]) -> argparse.Namespace:
        return argparse.Namespace(
            region=job_config.get("region") or self.region_id,
            env_file=job_config.get("env_file"),
            dataset_path=job_config.get("dataset_path") or "",
            epochs=job_config.get("epochs"),
            checkpoint_path=job_config.get("checkpoint_path") or "",
            checkpoint_frequency=job_config.get("checkpoint_frequency"),
            command=job_config.get("command"),
            command_template=job_config.get("command_template"),
            job_name=job_config.get("job_name") or "evo-train",
            workspace_id=job_config.get("workspace_id"),
            resource_id=job_config.get("resource_id"),
            image=job_config.get("image"),
            description=job_config.get("description"),
            ecs_spec=job_config.get("ecs_spec"),
            gpu_count=job_config.get("gpu_count"),
            gpu_type=job_config.get("gpu_type"),
            cpu=job_config.get("cpu"),
            memory=job_config.get("memory"),
            max_running_minutes=job_config.get("max_running_minutes"),
            mount=job_config.get("mount"),
            wait=job_config.get("wait", False),
            timeout=job_config.get("timeout", 86400),
            interval=job_config.get("interval", 30),
            dry_run=job_config.get("dry_run", False),
        )

    def submit(self, job_config: dict[str, Any]) -> str:
        args = self._args(job_config)
        start_train.load_env(args.env_file)
        client = start_train.create_client(args.region)
        body = start_train.create_job(client, args)
        if body is None:
            return ""
        return str(getattr(body, "job_id", "") or "")

    def metadata(self, job_id: str) -> dict[str, str]:
        start_train.load_env(None)
        client = start_train.create_client(self.region_id)
        body = start_train.fetch_job_body(client, job_id, need_detail=False)
        payload = start_train.build_status_payload(body)
        return {
            "status": str(payload.get("status") or ""),
            "last_error": str(payload.get("reason_message") or ""),
        }

    def stop(self, job_id: str) -> None:
        start_train.load_env(None)
        client = start_train.create_client(self.region_id)
        start_train.stop_job(client, job_id)
