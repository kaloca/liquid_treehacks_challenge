"""
This provides a quickstart reference that shows how to export a module wrapping torch.sin to Edge Dialect, lower it to XNNPACK, and profile it using the local Python profiler.

PYTHONPATH=profiling python references/quickstart.py
"""
from pathlib import Path
from typing import Tuple, Union
import os
import subprocess
import tempfile

import torch 
import torch.nn as nn
import numpy as np
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge
from executorch.exir.backend.backend_api import LoweredBackendModule, to_backend
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir.backend.test.backend_with_compiler_demo import (  # noqa
    BackendWithCompilerDemo,
)
from executorch.devtools import Inspector
from profiler import Profiler, ProfilingResults, extract_stats


class LocalPyProfiler(Profiler):
    """
    Local Python profiler implementation for ExecutorTorch modules.
    """

    def __init__(
        self,
        runner_path: Path,
    ):
        """
        Initialize local profiler.

        Args:
            runner_path: Path to the profiler runner
        """

        # Save arguments
        self._runner_path = runner_path

    def _load_pte(self, temp_dir: Path, pte: Union[bytes, Path]):
        if isinstance(pte, bytes):
            pte_path = temp_dir / "model.pte"
            with open(pte_path, "wb") as f:
                f.write(pte)
            return pte_path

        if isinstance(pte, Path):
            return pte
        
        elif isinstance(pte, str):
            pte_path = Path(pte)
            return pte_path

        raise ValueError(f"Invalid pte type: {type(pte)}, expected bytes or Path")

    def profile(self, pte: Union[bytes, Path], inputs: Tuple[torch.Tensor, ...], repeats: int) -> ProfilingResults:
        """
        Invoke repeats times forward method of the pte and collect execution times.

        Args:
            pte: Compiled model in bytes or Path to the model file
            inputs: Tuple of input tensors for the model
            repeats: Number of times to repeat the profiling

        Returns:
            ProfilingResults containing statistical measures of the profiling run
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            pte_path = self._load_pte(temp_dir, pte)

            inputs_path = temp_dir / "inputs.npz"
            np.savez(inputs_path, **{str(i): t.numpy() for i, t in enumerate(inputs)})

            etdump_path = temp_dir / "etdump.bin"

            profile_cmd = [
                self._runner_path,
                "-model_path",
                pte_path,
                "-etdump_path",
                etdump_path,
                "-inputs_npz_path",
                inputs_path,
                "-iter",
                f"{repeats}",
            ]

            try:
                subprocess.run(profile_cmd, capture_output=True, check=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                print(f"Error output:\n{e.stderr}")
                raise

            inspector = Inspector(etdump_path, debug_buffer_path=os.devnull)

        return extract_stats(inspector)


class LowerableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


# Export and lower the module to Edge Dialect
example_args = (torch.randn(128, dtype=torch.float32),)

module = LowerableModule()
aten_dialect_program: ExportedProgram = export(module, example_args)

edge_config = get_xnnpack_edge_compile_config()
edge_program: EdgeProgramManager = to_edge(aten_dialect_program, compile_config=edge_config)

# Lower the module
edge_manager_to_backend: LoweredBackendModule = edge_program.to_backend(XnnpackPartitioner())
print(edge_manager_to_backend)
et_program = edge_manager_to_backend.to_executorch()
print(et_program)

# Serialize and save it to a file   
save_path = "delegate.pte"
with open(save_path, "wb") as f:
    f.write(et_program.buffer)


pte_runner_path = os.environ.get("PTE_RUNNER_PATH", 'runner/macos-arm64/pte_runner')
profiler = LocalPyProfiler(pte_runner_path)
profiling_result = profiler.profile(save_path, example_args, repeats=4)
print(profiling_result)
