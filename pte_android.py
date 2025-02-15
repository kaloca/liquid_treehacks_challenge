#!/usr/bin/env python3
"""
Profile PTE via Android.

## Local device:
    ```
    PYTHONPATH=profiling \
            --pte your_pte_file.pte \
            --runner_path runner/android-arm64-v8a/pte_runner
    ```
"""

import argparse
import logging
from pathlib import Path
import pprint

import torch

from android_profiler import AndroidProfiler


def main(args):
    # prepare inputs
    inputs = (torch.randn(128, dtype=torch.float32),)

    # instantiate profiler
    profiler = AndroidProfiler(
        runner_local_path=args.runner_path,
        device_id=args.device_id,
        device_work_dir=args.device_workdir or AndroidProfiler.device_work_dir,
        cpu_threads=args.cpu_threads,
        adb_port=args.adb_port,
        ignore_cpu_throttling=(
            args.ignore_cpu_throttling
            if args.ignore_cpu_throttling is not None
            else AndroidProfiler.ignore_cpu_throttling
        ),
        cpu_throttling_max_wait_secs=(
            args.cpu_throttling_max_wait_secs
            if args.cpu_throttling_max_wait_secs is not None
            else AndroidProfiler.cpu_throttling_max_wait_secs
        ),
        use_fixed_performance_mode=(
            args.use_fixed_performance_mode
            if args.use_fixed_performance_mode is not None
            else AndroidProfiler.use_fixed_performance_mode
        ),
        cpu_throttling_check_interval=(
            args.cpu_throttling_check_interval
            if args.cpu_throttling_check_interval is not None
            else AndroidProfiler.cpu_throttling_check_interval
        ),
        cpu_throttling_thermal_status_threshold=(
            args.cpu_throttling_thermal_status_threshold
            if args.cpu_throttling_thermal_status_threshold is not None
            else AndroidProfiler.cpu_throttling_thermal_status_threshold
        ),
    )

    # profile
    profiling_result = profiler.profile(args.pte, inputs, repeats=args.repeats)

    # print results
    pprint.pprint(profiling_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile existing pte via ssh")
    parser.add_argument("--pte", type=Path, help="Path to pte file", required=True)
    parser.add_argument(
        "--runner_path", type=Path, help="the pte runner local path", required=True
    )
    parser.add_argument("--device_id", type=str, help="adb device id")
    parser.add_argument(
        "--device_workdir",
        type=Path,
        help="working directory on remote",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        help="Concurrency level",
        default=AndroidProfiler.cpu_threads,
    )
    parser.add_argument("--adb_port", type=int, help="Adb port if non-default required")
    parser.add_argument(
        "--repeats", type=int, help="Number of times to invoke the method", default=10
    )
    parser.add_argument(
        "--ignore_cpu_throttling",
        action="store_true",
        help="Disable throttling detection and cool down.",
    )
    parser.add_argument(
        "--cpu_throttling_max_wait_secs",
        type=int,
        help="Max interval in seconds to wait for the device cool down.",
    )
    parser.add_argument(
        "--use_fixed_performance_mode",
        action="store_true",
        help="Enable the fix performance mode",
    )
    parser.add_argument(
        "--cpu_throttling_check_interval",
        type=int,
        help="Interval in seconds to check for the throttling conditions.",
    )
    parser.add_argument(
        "--cpu_throttling_thermal_status_threshold",
        type=int,
        help="The thermal status, reaching which is considered throttling, use values from "
        "https://source.android.com/docs/core/power/thermal-mitigation#codes .",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)