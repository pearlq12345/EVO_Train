#!/usr/bin/env python3
"""Build user training commands."""
from __future__ import annotations

from typing import Any


class UserTrainCmd:
    COMMAND = "lerobot-train"
    DATASET_REPO_ID = "local/libero_10_no_noops_1.0.0_lerobot"
    DEFAULT_DATASET_ROOT = "/mnt/pai/data"
    DATASET_VIDEO_BACKEND = "pyav"
    DEFAULT_POLICY_TYPE = "act"
    POLICY_PUSH_TO_HUB = "false"
    POLICY_REPO_ID = "local/libero_10_no_noops_1.0.0_lerobot"
    OUTPUT_DIR = "/usrresult/%s/%s/checkpoint" ## username task_name
    POLICY_DEVICE = "cuda"
    POLICY_PRETRAINED_BACKBONE_WEIGHTS = "null"

    def __init__(self, request: dict[str, Any]) -> None:
        # self.dataset_path = str(request.get("datasetPath") or self.DEFAULT_DATASET_ROOT)
        self.dataset_path = self.DEFAULT_DATASET_ROOT
        # self.policy_type = str(request.get("policyType") or self.DEFAULT_POLICY_TYPE)
        self.policy_type = self.DEFAULT_POLICY_TYPE
        self.empty_docker = bool(request.get("emptyDocker", False)) # 创建一个空容器 不执行任何指令
        self.sleep_t = int(request.get("sleepT", 6000)) # 空容器存活时间
        self.steps = int(request.get("steps", 10000))
        self.save_freq = int(request.get("saveFreq", 1000))
        # self.gpu_count = int(request.get("gpuCount", 1))
        self.gpu_count = 1
        # self.gpu_type = str(request.get("gpuType") or "")
        self.gpu_type = ""
        self.batch_size = int(request.get("batchSize", 16))
        self.log_freq = int(request.get("logFreq", 100))

    def create_train_cmd(self, usrname: str, jobname: str) -> str:
        if self.empty_docker:
            return f"sleep {self.sleep_t}"
        output_dir = self.OUTPUT_DIR % (usrname, jobname)
        return (
            f"{self.COMMAND} "
            f"--dataset.repo_id={self.DATASET_REPO_ID} "
            f"--dataset.root={self.dataset_path} "
            f"--dataset.video_backend={self.DATASET_VIDEO_BACKEND} "
            f"--policy.type={self.policy_type} "
            f"--policy.push_to_hub={self.POLICY_PUSH_TO_HUB} "
            f"--policy.repo_id={self.POLICY_REPO_ID} "
            f"--output_dir={output_dir} "
            f"--steps={self.steps} "
            f"--save_freq={self.save_freq} "
            f"--batch_size={self.batch_size} "
            f"--log_freq={self.log_freq} "
            f"--policy.device={self.POLICY_DEVICE} "
            f"--policy.pretrained_backbone_weights={self.POLICY_PRETRAINED_BACKBONE_WEIGHTS}"
        )
