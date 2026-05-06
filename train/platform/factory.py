from __future__ import annotations

import os

from .aliyun_dlc import AliyunDLCPlatform
from .autodl import AutoDLPlatform
from .base import TrainPlatform


def get_platform(name: str | None = None) -> TrainPlatform:
    platform_name = name if name is not None else os.environ.get("TRAIN_PLATFORM")
    normalized = (platform_name or "").strip()
    if not normalized:
        return AliyunDLCPlatform()
    if normalized == "aliyun_dlc":
        return AliyunDLCPlatform()
    if normalized == "autodl":
        return AutoDLPlatform()
    raise ValueError(f"Unknown training platform: {normalized}")
