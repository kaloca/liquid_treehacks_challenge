from pathlib import Path
from typing import Optional, Tuple, Union
import os
import tempfile
import logging
import subprocess
import contextlib
import dataclasses
import subprocess
import hashlib

from executorch.devtools import Inspector
import numpy as np
import torch

from profiler import Profiler, ProfilingResults, extract_stats

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def pte_as_file(pte: Union[bytes, Path]):
    if isinstance(pte, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            tmp_file.write(pte)
            tmp_file.flush()
            yield Path(tmp_file.name)
    elif isinstance(pte, Path):
        yield pte
    else:
        raise ValueError(f"Invalid pte type: {type(pte)}, expected bytes or Path")


@contextlib.contextmanager
def input_as_npz_file(inputs: Tuple[torch.Tensor, ...] | Path):
    if isinstance(inputs, Path):
        yield inputs
    else:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp_file:
            np.savez(tmp_file, **{str(i): t.numpy() for i, t in enumerate(inputs)})
            tmp_file.flush()
            yield Path(tmp_file.name)


@dataclasses.dataclass
class AdbShell:
    device_id: Optional[str] = None
    port: Optional[str] = None
    port: Optional[str] = None

    def cmd(self, *args):
        r = ["adb"]
        if self.device_id:
            r.extend(["-s", str(self.device_id)])
        if self.port:
            r.extend(["-P", str(self.port)])
        r = ["adb"]
        if self.device_id:
            r.extend(["-s", str(self.device_id)])
        if self.port:
            r.extend(["-P", str(self.port)])
        r.extend(args)
        return r

    def calculate_adb_md5(self, path: Path):
        try:
            subprocess.check_output(self.cmd("shell", "stat", str(path)))
        except subprocess.CalledProcessError:
            logger.info(f"file {path} is not accessable on remote")
            return None
        # ba73f6793ff93db50abb82d1a0d1752a  /data/local/tmp/foo
        out = subprocess.check_output(self.cmd("shell", "md5sum", str(path)))
        return out.split()[0].strip()

    def push_files(self, paths_mapping: dict[Path, Path]):
        for src, dst in paths_mapping.items():

            def do_copy():
                logger.info(f"pushing {src} -> {dst}")
                subprocess.check_call(self.cmd("shell", "mkdir", "-p", str(dst.parent)))
                subprocess.check_call(self.cmd("push", str(src), str(dst)))

            if src.stat().st_size < 1024**2:  # copy small file
                do_copy()
            elif (remote_hash := self.calculate_adb_md5(dst)) is not None:
                if calculate_local_md5(src) != remote_hash:
                    do_copy()
                else:
                    logger.info(f"skip pusing identical {src} -> {dst}")
            else:
                do_copy()

    def pull_files(self, paths_mapping: dict[Path, Path]):
        for src, dst in paths_mapping.items():
            logger.info(f"pulling {src} -> {dst}")
            os.makedirs(dst.parent, exist_ok=True)
            subprocess.check_call(self.cmd("pull", str(src), str(dst)))

    def delete_files(self, paths: list[Path]):
        for p in paths:
            logger.info(f"deleting {p}")
            subprocess.check_call(self.cmd("shell", "rm", str(p)))

    def make_executable(self, paths: list[Path]):
        for p in paths:
            logger.info(f"making executable {p}")
            subprocess.check_call(self.cmd("shell", "chmod", "+x", str(p)))

def calculate_local_md5(file_path: Path):
    h = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest().encode("utf-8")


@dataclasses.dataclass
class AndroidProfiler(Profiler):
    runner_local_path: Path
    device_id: Optional[str] = None
    device_work_dir: Path = Path("/data/local/tmp/profiling")
    cpu_threads: int = -1
    adb_port: Optional[int] = None
    ignore_cpu_throttling: bool = False
    cpu_throttling_max_wait_secs: int = 60 * 5
    use_fixed_performance_mode: bool = False
    cpu_throttling_check_interval: int = 1
    cpu_throttling_thermal_status_threshold: int = 1

    def profile(
        self,
        pte: Union[bytes, Path],
        inputs: Tuple[torch.Tensor, ...] | Path,
        repeats: int,
    ) -> ProfilingResults:
        assert self.device_work_dir

        adb = AdbShell(device_id=self.device_id, port=self.adb_port)
        # check the device connection
        subprocess.check_call(
            adb.cmd(
                "shell",
                " ".join(
                    f'echo "{k}=$(getprop {k})";'
                    for k in (
                        "ro.product.model",
                        "ro.boot.serialno",
                        "ro.vendor.build.fingerprint",
                    )
                ),
            ),
        )

        with (
            pte_as_file(pte) as pte_local_path,
            input_as_npz_file(inputs) as inputs_local_path,
            tempfile.NamedTemporaryFile() as etdump_local_file,
        ):
            runner_remote_path = self.device_work_dir / self.runner_local_path.name
            pte_remote_path = self.device_work_dir / pte_local_path.name
            inputs_remote_path = self.device_work_dir / inputs_local_path.name
            etdump_remote_path = self.device_work_dir / f"{pte_local_path.name}.etdump"
            etdump_local_path = Path(etdump_local_file.name)

            adb.push_files(
                {
                    self.runner_local_path: runner_remote_path,
                    pte_local_path: pte_remote_path,
                    inputs_local_path: inputs_remote_path,
                }
            )
            adb.make_executable([runner_remote_path])

            pte_runner_args = [
                str(runner_remote_path),
                "-model_path",
                str(pte_remote_path),
                "-etdump_path",
                str(etdump_remote_path),
                "-inputs_npz_path",
                str(inputs_remote_path),
                "-iter",
                str(repeats),
                "-cpu_threads",
                str(self.cpu_threads),
            ]
            if self.ignore_cpu_throttling:
                pte_runner_args.append("-ignore_cpu_throttling")
            if self.cpu_throttling_max_wait_secs is not None:
                pte_runner_args += [
                    "-cpu_throttling_max_wait_secs",
                    str(self.cpu_throttling_max_wait_secs),
                ]
            if self.use_fixed_performance_mode:
                pte_runner_args.append("-use_fixed_performance_mode")
            if self.cpu_throttling_check_interval is not None:
                pte_runner_args += [
                    "-cpu_throttling_check_interval",
                    str(self.cpu_throttling_check_interval),
                ]
            if self.cpu_throttling_thermal_status_threshold is not None:
                pte_runner_args += [
                    "-cpu_throttling_thermal_status_threshold",
                    str(self.cpu_throttling_thermal_status_threshold),
                ]

            profile_cmd = adb.cmd("shell", *pte_runner_args)
            logger.info("running: %s", " ".join(profile_cmd))
            subprocess.check_call(profile_cmd)
            logger.info("run complete")

            adb.pull_files({etdump_remote_path: etdump_local_path})
            inspector = Inspector(str(etdump_local_path), debug_buffer_path=os.devnull)

            # clean some annoying files, .pte maybe has a stable name that gets overriden
            adb.delete_files([etdump_remote_path, inputs_remote_path])

        return extract_stats(inspector)
