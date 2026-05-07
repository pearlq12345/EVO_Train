#!/usr/bin/env python3
"""
Author: Ru-hulu
Date: 2026-05-03

Load data from Alibaba Cloud OSS to a CPFS directory.
"""
# python3 scripts/load_oss_to_cpfs.py \
#   --oss 'oss://evo-data/user_uploads/0e00c623-5730-495b-b958-069eff663555/c2f4e378-bf8e-48b9-9c36-a7aff9a0677c/libero_10_no_noops_1.0.0_lerobot/' \
#   --cpfs /mnt/libero_10_no_noops_1.0.0_lerobot \
#   --endpoint https://oss-cn-hangzhou-internal.aliyuncs.com \
#   --jobs 32 \
#   --update
# user_uploads/0e00c623-5730-495b-b958-069eff663555/a42b1ce1-02ed-4023-b80b-fcfbbaf755dc/260407-szk-TestData
# -oss: the directory of source data
# -cpfs: the directory of destination
# -endpoint: Designate network access entry for OSS. It can be find in .env
# -job: the number of process to copy data
# -update: If the target directory already contains a file with the same name and it is up-to-date, skip repeated downloading.
from __future__ import annotations
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

def log(message: str) -> None:
    print(f"[load_oss_to_cpfs] {message}", flush=True)

def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)

def run(command: list[str]) -> None:
    quoted = " ".join(shlex.quote(part) for part in command)
    log(quoted)
    subprocess.run(command, check=True)

def find_executable(names: list[str]) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    fail(f"missing required command: one of {', '.join(names)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load data from Alibaba Cloud OSS to a CPFS directory.",
    )
    parser.add_argument("--oss", required=True, help="OSS source, for example oss://bucket/path/")
    parser.add_argument("--cpfs", required=True, help="CPFS destination directory")
    parser.add_argument("--endpoint", help="OSS endpoint, for example oss-cn-hangzhou-internal.aliyuncs.com")
    parser.add_argument("--jobs", type=int, default=int(os.getenv("OSS_LOAD_JOBS", "32")))
    parser.add_argument("--update", action="store_true", help="Skip files that are already up to date")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.oss.startswith("oss://"):
        fail("--oss must start with oss://")
    if args.jobs <= 0:
        fail("--jobs must be greater than 0")

    ossutil_bin = find_executable(["ossutil", "ossutil64"])
    cpfs_dst = Path(args.cpfs).expanduser().resolve()

    log(f"OSS source: {args.oss}")
    log(f"CPFS destination: {cpfs_dst}")
    log(f"ossutil: {ossutil_bin}")

    cpfs_dst.mkdir(parents=True, exist_ok=True)

    command = [ossutil_bin]
    if args.endpoint:
        command.extend(["--endpoint", args.endpoint])
    command.extend(["cp", "-r", args.oss, str(cpfs_dst)])
    if args.update:
        command.append("--update")
    command.extend(["--jobs", str(args.jobs)])

    run(command)
    log("load completed")

if __name__ == "__main__":
    main()
