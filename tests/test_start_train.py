from __future__ import annotations

from io import StringIO
import unittest
from unittest.mock import MagicMock, patch

from train import start_train


class StartTrainCliTests(unittest.TestCase):
    def test_parse_args_accepts_platform_and_workdir(self) -> None:
        args = start_train.parse_args(
            [
                "--platform",
                "autodl",
                "submit",
                "--dataset-path",
                "/data/demo",
                "--epochs",
                "2",
                "--checkpoint-path",
                "/ckpt/demo",
                "--checkpoint-frequency",
                "1",
                "--gpu-count",
                "1",
                "--workdir",
                "/root/EVO_Train",
            ]
        )

        self.assertEqual(args.platform, "autodl")
        self.assertEqual(args.workdir, "/root/EVO_Train")

    def test_submit_job_uses_selected_platform(self) -> None:
        args = start_train.parse_args(
            [
                "--platform",
                "autodl",
                "submit",
                "--dataset-path",
                "/data/demo",
                "--epochs",
                "2",
                "--checkpoint-path",
                "/ckpt/demo",
                "--checkpoint-frequency",
                "1",
                "--gpu-count",
                "1",
                "--workdir",
                "/root/EVO_Train",
            ]
        )
        platform = MagicMock()
        platform.submit.return_value = "job-1"

        with patch.object(start_train, "get_platform", return_value=platform) as mocked_factory:
            job_id = start_train.submit_job(args)

        self.assertEqual(job_id, "job-1")
        mocked_factory.assert_called_once_with("autodl", region_id="cn-hangzhou")
        platform.submit.assert_called_once_with(vars(args))

    def test_get_job_returns_json_metadata_for_autodl(self) -> None:
        args = start_train.parse_args(["--platform", "autodl", "status", "--job-id", "job-1"])
        platform = MagicMock()
        platform.metadata.return_value = {"status": "Running", "last_error": ""}

        with (
            patch.object(start_train, "get_platform", return_value=platform),
            patch("sys.stdout", new_callable=StringIO) as stdout,
        ):
            payload = start_train.get_job(args)

        self.assertEqual(payload["status"], "Running")
        self.assertIn('"platform": "autodl"', stdout.getvalue())
        self.assertIn('"job_id": "job-1"', stdout.getvalue())

    def test_wait_job_polls_generic_platform_until_success(self) -> None:
        platform = MagicMock()
        platform.status.side_effect = ["Running", "Succeeded"]

        with patch.object(start_train.time, "sleep"):
            start_train.wait_job(platform, "job-1", timeout=5, interval=0)

        self.assertEqual(platform.status.call_count, 2)


if __name__ == "__main__":
    unittest.main()
