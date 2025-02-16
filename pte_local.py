"""
Local PTE Profiler

This script provides functionality to profile PyTorch exported models locally.
It measures execution time of forward method for the given PTE file.

Usage:
    python pte_local.py --pte <path_to_pte_file>

Example:
    python pte_local.py --pte model.pte
"""

import argparse
from pathlib import Path
import pprint

import torch

from quickstart import LocalPyProfiler


def main(args):
    # prepare inputs
    inputs = (torch.ones(1, 5, dtype=torch.int64), torch.zeros(1, dtype=torch.int64))

    # profile with LocalPyProfiler
    profiler = LocalPyProfiler(args.runner_path)
    profiling_result = profiler.profile(args.pte, inputs, repeats=4)
    pprint.pprint(profiling_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile existing pte locally")
    parser.add_argument("--pte", type=Path, help="Path to pte file", required=True)
    parser.add_argument(
        "--runner_path",
        type=Path,
        help="path to pte_runner (for default is 'runner/linux/pte_runner')",
        required=True,
    )
    args = parser.parse_args()

    main(args)
