from __future__ import annotations

import os
from typing import Any

from .aliyun_dlc import AliyunDLCPlatform
from .autodl import AutoDLPlatform
from .base import TrainPlatform


def get_platform(name: str | None = None, **kwargs: Any) -> TrainPlatform:
    platform_name = name if name is not None else os.environ.get("TRAIN_PLATFORM")
    normalized = (platform_name or "").strip().lower()
    if not normalized:
        return AliyunDLCPlatform(region_id=kwargs.get("region_id"))
    if normalized in {"aliyun", "aliyun_dlc"}:
        return AliyunDLCPlatform(region_id=kwargs.get("region_id"))
    if normalized == "autodl":
        return AutoDLPlatform()
    raise ValueError(f"Unknown training platform: {normalized}")
