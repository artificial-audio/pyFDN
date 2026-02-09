"""Profiling report utilities for recursive stage processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.profiler import ProfilerActivity

from .cost import StageCost


@dataclass(frozen=True)
class ProcessProfileConfig:
    """Configuration for `RecursionCore.process(..., profile=True)`."""

    activities: Optional[Tuple[ProfilerActivity, ...]] = None
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = True
    include_analytical: bool = True
    enabled: bool = True

    def resolve_activities(self, device: torch.device) -> Tuple[ProfilerActivity, ...]:
        if self.activities is not None:
            return self.activities
        resolved = [ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            resolved.append(ProfilerActivity.CUDA)
        return tuple(resolved)


@dataclass(frozen=True)
class StageProfileBucket:
    stage_index: int
    stage_name: str
    stage_type: str
    calls: int
    time_cpu_total_us: float
    time_cpu_self_us: float
    time_cuda_total_us: float
    time_cuda_self_us: float
    op_calls: int
    op_unique: int
    profiled_flops: float
    cpu_mem_total_bytes: float
    cpu_mem_self_bytes: float
    cuda_mem_total_bytes: float
    cuda_mem_self_bytes: float
    analytical_cost: Optional[StageCost] = None


@dataclass(frozen=True)
class ProcessProfileTotals:
    calls: int
    time_cpu_total_us: float
    time_cpu_self_us: float
    time_cuda_total_us: float
    time_cuda_self_us: float
    op_calls: int
    op_unique: int
    profiled_flops: float
    cpu_mem_total_bytes: float
    cpu_mem_self_bytes: float
    cuda_mem_total_bytes: float
    cuda_mem_self_bytes: float


@dataclass(frozen=True)
class ProcessProfileReport:
    enabled: bool
    device: str
    dtype: str
    batch_size: int
    block_size: int
    num_blocks: int
    total_samples: int
    stage_buckets: Tuple[StageProfileBucket, ...]
    totals: ProcessProfileTotals
    analytical_block_cost: Optional[StageCost]
    analytical_total_cost: Optional[StageCost]
    raw_profiler_table: str
    notes: Tuple[str, ...] = ()


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_stage_label(label: str, prefix: str = "stage::") -> Optional[Tuple[int, str]]:
    if not label.startswith(prefix):
        return None
    payload = label[len(prefix):]
    first = payload.find(":")
    if first < 0:
        return None
    idx_text = payload[:first]
    stage_type = payload[first + 1 :]
    try:
        idx = int(idx_text)
    except ValueError:
        return None
    return idx, stage_type


def _iter_descendants(event: object) -> Iterable[object]:
    children = getattr(event, "cpu_children", None)
    if children is None:
        return ()

    stack = list(children)
    out: list[object] = []
    while stack:
        node = stack.pop()
        out.append(node)
        node_children = getattr(node, "cpu_children", None)
        if node_children:
            stack.extend(node_children)
    return out


def _collect_stage_events(events: Sequence[object], prefix: str = "stage::") -> list[object]:
    return [event for event in events if str(getattr(event, "name", "")).startswith(prefix)]


def _aggregate_totals(buckets: Sequence[StageProfileBucket]) -> ProcessProfileTotals:
    return ProcessProfileTotals(
        calls=int(sum(bucket.calls for bucket in buckets)),
        time_cpu_total_us=float(sum(bucket.time_cpu_total_us for bucket in buckets)),
        time_cpu_self_us=float(sum(bucket.time_cpu_self_us for bucket in buckets)),
        time_cuda_total_us=float(sum(bucket.time_cuda_total_us for bucket in buckets)),
        time_cuda_self_us=float(sum(bucket.time_cuda_self_us for bucket in buckets)),
        op_calls=int(sum(bucket.op_calls for bucket in buckets)),
        op_unique=int(sum(bucket.op_unique for bucket in buckets)),
        profiled_flops=float(sum(bucket.profiled_flops for bucket in buckets)),
        cpu_mem_total_bytes=float(sum(bucket.cpu_mem_total_bytes for bucket in buckets)),
        cpu_mem_self_bytes=float(sum(bucket.cpu_mem_self_bytes for bucket in buckets)),
        cuda_mem_total_bytes=float(sum(bucket.cuda_mem_total_bytes for bucket in buckets)),
        cuda_mem_self_bytes=float(sum(bucket.cuda_mem_self_bytes for bucket in buckets)),
    )


def build_process_profile_report(
    *,
    prof: object,
    stages: Sequence[object],
    batch_size: int,
    block_size: int,
    num_blocks: int,
    total_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    input_signal: torch.Tensor,
    config: ProcessProfileConfig,
) -> ProcessProfileReport:
    """Build a structured process profiling report from a torch profiler capture."""
    events = list(prof.events())
    stage_events = _collect_stage_events(events)
    notes: list[str] = []

    stage_name_map: Dict[int, str] = {idx: str(stage) for idx, stage in enumerate(stages)}
    stage_type_map: Dict[int, str] = {
        idx: stage.__class__.__name__ for idx, stage in enumerate(stages)
    }

    aggregates: Dict[int, dict[str, object]] = {}
    missing_child_link_count = 0

    for event in stage_events:
        name = str(getattr(event, "name", ""))
        parsed = _parse_stage_label(name)
        if parsed is None:
            continue
        stage_index, stage_type = parsed

        if stage_index not in aggregates:
            aggregates[stage_index] = {
                "stage_name": stage_name_map.get(stage_index, stage_type),
                "stage_type": stage_type_map.get(stage_index, stage_type),
                "calls": 0,
                "time_cpu_total_us": 0.0,
                "time_cpu_self_us": 0.0,
                "time_cuda_total_us": 0.0,
                "time_cuda_self_us": 0.0,
                "op_calls": 0,
                "op_names": set(),
                "profiled_flops": 0.0,
                "cpu_mem_total_bytes": 0.0,
                "cpu_mem_self_bytes": 0.0,
                "cuda_mem_total_bytes": 0.0,
                "cuda_mem_self_bytes": 0.0,
                "analytical_cost": None,
            }

        acc = aggregates[stage_index]
        acc["calls"] = int(acc["calls"]) + 1
        acc["time_cpu_total_us"] = float(acc["time_cpu_total_us"]) + _as_float(
            getattr(event, "cpu_time_total", 0.0)
        )
        acc["time_cpu_self_us"] = float(acc["time_cpu_self_us"]) + _as_float(
            getattr(event, "self_cpu_time_total", 0.0)
        )
        acc["time_cuda_total_us"] = float(acc["time_cuda_total_us"]) + _as_float(
            getattr(event, "cuda_time_total", 0.0)
        )
        acc["time_cuda_self_us"] = float(acc["time_cuda_self_us"]) + _as_float(
            getattr(event, "self_cuda_time_total", 0.0)
        )
        acc["cpu_mem_total_bytes"] = float(acc["cpu_mem_total_bytes"]) + _as_float(
            getattr(event, "cpu_memory_usage", 0.0)
        )
        acc["cpu_mem_self_bytes"] = float(acc["cpu_mem_self_bytes"]) + _as_float(
            getattr(event, "self_cpu_memory_usage", 0.0)
        )
        acc["cuda_mem_total_bytes"] = float(acc["cuda_mem_total_bytes"]) + _as_float(
            getattr(event, "cuda_memory_usage", 0.0)
        )
        acc["cuda_mem_self_bytes"] = float(acc["cuda_mem_self_bytes"]) + _as_float(
            getattr(event, "self_cuda_memory_usage", 0.0)
        )

        descendants = list(_iter_descendants(event))
        if len(descendants) == 0 and getattr(event, "cpu_children", None) is None:
            missing_child_link_count += 1

        op_names = acc["op_names"]
        for desc in descendants:
            desc_name = str(getattr(desc, "name", ""))
            if desc_name.startswith("stage::"):
                continue
            op_names.add(desc_name)
            desc_count = getattr(desc, "count", 1)
            try:
                op_count = int(desc_count)
            except (TypeError, ValueError):
                op_count = 1
            if op_count <= 0:
                op_count = 1
            acc["op_calls"] = int(acc["op_calls"]) + op_count
            acc["profiled_flops"] = float(acc["profiled_flops"]) + _as_float(
                getattr(desc, "flops", 0.0)
            )

    if missing_child_link_count > 0:
        notes.append(
            "Profiler event tree linkage was unavailable for some stage ranges; "
            "op_calls/profiled_flops may be incomplete."
        )

    analytical_block_cost: Optional[StageCost] = None
    analytical_total_cost: Optional[StageCost] = None
    if config.include_analytical:
        from .cost import estimate_cost_from_input

        analytical_report = estimate_cost_from_input(
            stages,
            input_signal,
            block_size=block_size,
            total_samples=total_samples,
        )
        analytical_block_cost = analytical_report.block_cost
        analytical_total_cost = analytical_report.total_cost

        for idx, entry in enumerate(analytical_report.per_stage):
            if idx in aggregates:
                aggregates[idx]["analytical_cost"] = entry.cost

    stage_buckets: list[StageProfileBucket] = []
    for stage_index in range(len(stages)):
        if stage_index in aggregates:
            acc = aggregates[stage_index]
            op_names = acc["op_names"]
            stage_buckets.append(
                StageProfileBucket(
                    stage_index=stage_index,
                    stage_name=str(acc["stage_name"]),
                    stage_type=str(acc["stage_type"]),
                    calls=int(acc["calls"]),
                    time_cpu_total_us=float(acc["time_cpu_total_us"]),
                    time_cpu_self_us=float(acc["time_cpu_self_us"]),
                    time_cuda_total_us=float(acc["time_cuda_total_us"]),
                    time_cuda_self_us=float(acc["time_cuda_self_us"]),
                    op_calls=int(acc["op_calls"]),
                    op_unique=len(op_names),
                    profiled_flops=float(acc["profiled_flops"]),
                    cpu_mem_total_bytes=float(acc["cpu_mem_total_bytes"]),
                    cpu_mem_self_bytes=float(acc["cpu_mem_self_bytes"]),
                    cuda_mem_total_bytes=float(acc["cuda_mem_total_bytes"]),
                    cuda_mem_self_bytes=float(acc["cuda_mem_self_bytes"]),
                    analytical_cost=acc["analytical_cost"],
                )
            )
        else:
            stage_buckets.append(
                StageProfileBucket(
                    stage_index=stage_index,
                    stage_name=stage_name_map[stage_index],
                    stage_type=stage_type_map[stage_index],
                    calls=0,
                    time_cpu_total_us=0.0,
                    time_cpu_self_us=0.0,
                    time_cuda_total_us=0.0,
                    time_cuda_self_us=0.0,
                    op_calls=0,
                    op_unique=0,
                    profiled_flops=0.0,
                    cpu_mem_total_bytes=0.0,
                    cpu_mem_self_bytes=0.0,
                    cuda_mem_total_bytes=0.0,
                    cuda_mem_self_bytes=0.0,
                    analytical_cost=None,
                )
            )

    totals = _aggregate_totals(stage_buckets)
    try:
        raw_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1)
    except Exception:
        raw_table = ""

    return ProcessProfileReport(
        enabled=True,
        device=str(device),
        dtype=str(dtype),
        batch_size=int(batch_size),
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        total_samples=int(total_samples),
        stage_buckets=tuple(stage_buckets),
        totals=totals,
        analytical_block_cost=analytical_block_cost,
        analytical_total_cost=analytical_total_cost,
        raw_profiler_table=raw_table,
        notes=tuple(notes),
    )
