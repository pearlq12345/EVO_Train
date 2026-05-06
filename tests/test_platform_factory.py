from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from train.platform.aliyun_dlc import AliyunDLCPlatform
from train.platform.autodl import AutoDLPlatform
from train.platform.factory import get_platform


class PlatformFactoryTests(unittest.TestCase):
    def test_default_returns_aliyun_platform(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            platform = get_platform()
        self.assertIsInstance(platform, AliyunDLCPlatform)

    def test_factory_returns_aliyun_platform(self) -> None:
        platform = get_platform("aliyun_dlc")
        self.assertIsInstance(platform, AliyunDLCPlatform)

    def test_factory_returns_autodl_platform(self) -> None:
        platform = get_platform("autodl")
        self.assertIsInstance(platform, AutoDLPlatform)

    def test_factory_uses_environment_variable(self) -> None:
        with patch.dict(os.environ, {"TRAIN_PLATFORM": "autodl"}, clear=False):
            platform = get_platform()
        self.assertIsInstance(platform, AutoDLPlatform)

    def test_unknown_platform_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_platform("unknown")


if __name__ == "__main__":
    unittest.main()
