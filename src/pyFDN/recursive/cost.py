"""Analytical cost model utilities for recursive DSP stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch


Shape3D = Tuple[int, int, int]


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch dtype."""
    return int(torch.tensor([], dtype=dtype).element_size())


def nbytes_for_shape(shape: Shape3D, dtype: torch.dtype) -> int:
    """Compute the number of bytes for a tensor with the given shape and dtype."""
    numel = int(shape[0]) * int(shape[1]) * int(shape[2])
    return numel * dtype_nbytes(dtype)


@dataclass(frozen=True)
class CostContext:
    """Context for estimating stage cost."""

    batch_size: int
    block_size: int
    dtype: torch.dtype
    x_shape: Optional[Shape3D]
    lines_in: Optional[Shape3D]
    lines_out: Optional[Shape3D]
    y_shape: Optional[Shape3D]
    delay_buffer_len: Optional[int] = None


@dataclass(frozen=True)
class StageCost:
    """Cost for a stage (DSP-level FLOPs and bytes + implementation bytes)."""

    flops: float = 0.0
    bytes_read_dsp: float = 0.0
    bytes_write_dsp: float = 0.0
    bytes_read_impl: float = 0.0
    bytes_write_impl: float = 0.0
    int_ops: float = 0.0

    @classmethod
    def zero(cls) -> "StageCost":
        return cls()

    def __add__(self, other: "StageCost") -> "StageCost":
        return StageCost(
            flops=self.flops + other.flops,
            bytes_read_dsp=self.bytes_read_dsp + other.bytes_read_dsp,
            bytes_write_dsp=self.bytes_write_dsp + other.bytes_write_dsp,
            bytes_read_impl=self.bytes_read_impl + other.bytes_read_impl,
            bytes_write_impl=self.bytes_write_impl + other.bytes_write_impl,
            int_ops=self.int_ops + other.int_ops,
        )

    def scale(self, factor: float) -> "StageCost":
        return StageCost(
            flops=self.flops * factor,
            bytes_read_dsp=self.bytes_read_dsp * factor,
            bytes_write_dsp=self.bytes_write_dsp * factor,
            bytes_read_impl=self.bytes_read_impl * factor,
            bytes_write_impl=self.bytes_write_impl * factor,
            int_ops=self.int_ops * factor,
        )

    def per_sample(self, samples: int) -> "StageCost":
        if samples <= 0:
            raise ValueError(f"samples must be positive, got {samples}")
        return self.scale(1.0 / float(samples))

    @property
    def bytes_total_dsp(self) -> float:
        return self.bytes_read_dsp + self.bytes_write_dsp

    @property
    def bytes_total_impl(self) -> float:
        return self.bytes_read_impl + self.bytes_write_impl


@dataclass(frozen=True)
class StageCostEntry:
    stage_name: str
    stage_type: str
    cost: StageCost


@dataclass(frozen=True)
class CostReport:
    per_stage: Tuple[StageCostEntry, ...]
    block_cost: StageCost
    per_sample_cost: StageCost
    total_cost: Optional[StageCost]
    num_blocks: Optional[int]
    block_size: int
    batch_size: int
    dtype: torch.dtype
    total_samples: Optional[int]


def _sum_costs(costs: Iterable[StageCost]) -> StageCost:
    total = StageCost.zero()
    for cost in costs:
        total = total + cost
    return total


def estimate_cost_from_shape(
    stages_or_core: Sequence[object],
    *,
    batch_size: int,
    num_inputs: int,
    dtype: torch.dtype = torch.float32,
    block_size: Optional[int] = None,
    total_samples: Optional[int] = None,
) -> CostReport:
    """Estimate cost using shape-based inference."""
    if hasattr(stages_or_core, "stages"):
        stages = stages_or_core.stages
        if block_size is None:
            block_size = int(getattr(stages_or_core, "block_size", 0))
    else:
        stages = stages_or_core

    if block_size is None or block_size <= 0:
        raise ValueError("block_size must be provided or available on RecursionCore")

    from .delay_lines import DelayRead, DelayWrite, DiagonalDelay
    from .feedback_mix import FeedbackMix
    from .input_tap import InputTap
    from .output_tap import OutputTap
    from .biquads import Biquads

    B = int(batch_size)
    T = int(block_size)
    x_shape: Shape3D = (B, int(num_inputs), T)

    lines_shape: Optional[Shape3D] = None
    delay_buffers: dict[str, int] = {}
    per_stage_entries: list[StageCostEntry] = []

    for stage in stages:
        lines_in = lines_shape
        lines_out: Optional[Shape3D] = None
        stage_y_shape: Optional[Shape3D] = None
        delay_buffer_len: Optional[int] = None

        if isinstance(stage, DelayRead):
            N = int(stage.num_lines)
            lines_out = (B, N, T)
            max_delay = int(stage.delay_lengths.max().item()) if N > 0 else 0
            delay_buffer_len = max_delay + T
            delay_buffers[stage.state_key] = delay_buffer_len
        elif isinstance(stage, DelayWrite):
            lines_out = lines_in
            delay_buffer_len = delay_buffers.get(stage.state_key)
        elif isinstance(stage, DiagonalDelay):
            N = int(stage.num_lines)
            lines_out = lines_in if lines_in is not None else (B, N, T)
            max_delay = int(stage.delay_lengths.max().item()) if N > 0 else 0
            delay_buffer_len = max_delay + T
            delay_buffers[stage.state_key] = delay_buffer_len
        elif isinstance(stage, FeedbackMix):
            N = int(stage.A.shape[0])
            lines_out = lines_in if lines_in is not None else (B, N, T)
        elif isinstance(stage, InputTap):
            N = int(stage.B.shape[0])
            lines_out = lines_in if lines_in is not None else (B, N, T)
        elif isinstance(stage, OutputTap):
            N_out = int(stage.C.shape[0])
            lines_out = lines_in
            stage_y_shape = (B, N_out, T)
        elif isinstance(stage, Biquads):
            N = int(stage.num_lines)
            lines_out = lines_in if lines_in is not None else (B, N, T)
        else:
            lines_out = lines_in

        ctx = CostContext(
            batch_size=B,
            block_size=T,
            dtype=dtype,
            x_shape=x_shape,
            lines_in=lines_in,
            lines_out=lines_out,
            y_shape=stage_y_shape,
            delay_buffer_len=delay_buffer_len,
        )
        cost = stage.estimate_cost(ctx)
        per_stage_entries.append(
            StageCostEntry(stage_name=str(stage), stage_type=stage.__class__.__name__, cost=cost)
        )

        if lines_out is not None:
            lines_shape = lines_out

    block_cost = _sum_costs(entry.cost for entry in per_stage_entries)
    per_sample_cost = block_cost.per_sample(T)

    num_blocks: Optional[int] = None
    total_cost: Optional[StageCost] = None
    if total_samples is not None:
        num_blocks = int((int(total_samples) + T - 1) // T)
        total_cost = block_cost.scale(num_blocks)

    return CostReport(
        per_stage=tuple(per_stage_entries),
        block_cost=block_cost,
        per_sample_cost=per_sample_cost,
        total_cost=total_cost,
        num_blocks=num_blocks,
        block_size=T,
        batch_size=B,
        dtype=dtype,
        total_samples=total_samples,
    )


def estimate_cost_from_input(
    stages_or_core: Sequence[object],
    input_signal: torch.Tensor,
    *,
    block_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    total_samples: Optional[int] = None,
) -> CostReport:
    """Estimate cost from a real input tensor (infers batch and input channels)."""
    if input_signal.dim() == 2:
        T_total, num_inputs = input_signal.shape
        batch_size = 1
    elif input_signal.dim() == 3:
        batch_size, num_inputs, T_total = input_signal.shape
    else:
        raise ValueError("input_signal must be 2D [T, N_in] or 3D [B, N_in, T]")

    if total_samples is None:
        total_samples = int(T_total)

    resolved_dtype = dtype if dtype is not None else input_signal.dtype

    return estimate_cost_from_shape(
        stages_or_core,
        batch_size=batch_size,
        num_inputs=num_inputs,
        dtype=resolved_dtype,
        block_size=block_size,
        total_samples=total_samples,
    )


def derive_metrics(cost: StageCost, runtime_s: float) -> dict[str, float]:
    """Derive GFLOP/s and GB/s metrics from cost and runtime."""
    if runtime_s <= 0:
        raise ValueError(f"runtime_s must be positive, got {runtime_s}")

    gflops = cost.flops / runtime_s / 1e9

    gb_read_dsp = cost.bytes_read_dsp / runtime_s / 1e9
    gb_write_dsp = cost.bytes_write_dsp / runtime_s / 1e9
    gb_total_dsp = cost.bytes_total_dsp / runtime_s / 1e9

    gb_read_impl = cost.bytes_read_impl / runtime_s / 1e9
    gb_write_impl = cost.bytes_write_impl / runtime_s / 1e9
    gb_total_impl = cost.bytes_total_impl / runtime_s / 1e9

    return {
        "gflops": gflops,
        "gb_read_dsp": gb_read_dsp,
        "gb_write_dsp": gb_write_dsp,
        "gb_total_dsp": gb_total_dsp,
        "gb_read_impl": gb_read_impl,
        "gb_write_impl": gb_write_impl,
        "gb_total_impl": gb_total_impl,
    }
