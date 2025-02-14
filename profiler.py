from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from pathlib import Path

import torch
from executorch.devtools import Inspector


@dataclass(frozen=True)
class ProfilingResults:
    """
    Data class containing profiling results in milliseconds.
    """

    raw: List[float]
    p10: float
    p50: float
    p90: float
    min: float
    avg: float
    max: float


class Profiler(ABC):
    """
    Abstract base class for implementing profilers.
    """

    @abstractmethod
    def profile(self, pte: Union[bytes, Path], inputs: Tuple[torch.Tensor, ...], repeats: int) -> ProfilingResults:
        """
        Profile the execution of a model with given inputs.

        Args:
            pte: Compiled model in bytes or Path to the compiled model file.
            inputs: Tuple of input tensors for the model.
            repeats: Number of times to repeat the profiling.

        Returns:
            ProfilingResults containing statistical measures of the profiling run.
        """
        pass


def extract_stats(inspector: Inspector) -> ProfilingResults:
    df = inspector.to_dataframe(
        include_units=False,
        include_delegate_debug_data=False,
    )

    stats = df[df["event_name"] == "Method::execute"][["raw", "p10", "p50", "p90", "min", "avg", "max"]].to_dict(
        orient="records"
    )[0]

    if not stats:
        raise ValueError("No statistics found in inspector data")
    return ProfilingResults(**stats)
