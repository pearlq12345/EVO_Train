from __future__ import annotations

import os
from typing import Any

from .aliyun_dlc import AliyunDLCPlatform
from .autodl import AutoDLPlatform
from .base import TrainPlatform


def normalize_provider(provider: str | None) -> str:
    normalized = (provider or "").strip().lower()
    if not normalized or normalized == "aliyun_dlc":
        return "aliyun"
    return normalized


def get_platform(name: str | None = None, **kwargs: Any) -> TrainPlatform:
    normalized = normalize_provider(name if name is not None else os.environ.get("TRAIN_PLATFORM"))
    if normalized == "aliyun":
        return AliyunDLCPlatform(region_id=kwargs.get("region_id", "cn-hangzhou"))
    if normalized == "autodl":
        return AutoDLPlatform()
    raise ValueError(f"Unknown training platform: {normalized}")
