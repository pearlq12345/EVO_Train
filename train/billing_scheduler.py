#!/usr/bin/env python3
"""Background billing reconciliation for training tasks."""

from __future__ import annotations

import threading
import time

from train.server_function import scan_billing_tasks


class BillingScheduler:
    def __init__(self, *, interval_seconds: float, env_file: str | None = None, region: str | None = None) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0")
        self.interval_seconds = interval_seconds
        self.env_file = env_file
        self.region = region
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="billing-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            stats = scan_billing_tasks(env_file=self.env_file, region=self.region)
            if stats["checked"] or stats["errors"]:
                print(f"[billing_scheduler] scan {stats}", flush=True)
