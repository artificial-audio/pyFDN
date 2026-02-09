"""Tests for `RecursionCore.process` profiling integration."""

import torch

from pyFDN.recursive import (
    DelayRead,
    DelayWrite,
    FeedbackMix,
    InputTap,
    OutputTap,
    ProcessProfileConfig,
    RecursionCore,
)


def _build_profiled_core(block_size: int = 4) -> RecursionCore:
    stages = [
        DelayRead(delay_length=4, num_lines=2),
        InputTap(input_matrix=torch.ones(2, 1)),
        FeedbackMix(feedback_matrix=torch.eye(2)),
        OutputTap(output_matrix=torch.ones(1, 2)),
        DelayWrite(),
    ]
    return RecursionCore(stages, block_size=block_size)


def test_process_profile_flag_backward_compatible():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    y_default = core.process(x)
    y_flag = core.process(x, profile=False)

    assert isinstance(y_default, torch.Tensor)
    assert isinstance(y_flag, torch.Tensor)
    assert y_default.shape == y_flag.shape
    assert torch.allclose(y_default, y_flag)
    assert core.last_profile_report is None


def test_process_profile_stores_report():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    y = core.process(x, profile=True)

    assert isinstance(y, torch.Tensor)
    assert core.last_profile_report is not None
    assert len(core.last_profile_report.stage_buckets) == len(core.stages)


def test_process_profile_return_profile():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    y, report = core.process(x, profile=True, return_profile=True)

    assert isinstance(y, torch.Tensor)
    assert report is not None
    assert report is core.last_profile_report


def test_stage_bucket_has_time_ops_memory():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    report = core.process(x, profile=True, return_profile=True)[1]
    assert report is not None

    for bucket in report.stage_buckets:
        assert bucket.calls == report.num_blocks
        assert bucket.time_cpu_total_us >= 0.0
        assert bucket.time_cpu_self_us >= 0.0
        assert bucket.time_cuda_total_us >= 0.0
        assert bucket.time_cuda_self_us >= 0.0
        assert bucket.op_calls >= 0
        assert bucket.op_unique >= 0
        assert bucket.profiled_flops >= 0.0
        assert bucket.cpu_mem_total_bytes >= 0.0
        assert torch.isfinite(torch.tensor(bucket.cpu_mem_self_bytes))
        assert bucket.cuda_mem_total_bytes >= 0.0
        assert torch.isfinite(torch.tensor(bucket.cuda_mem_self_bytes))


def test_profile_totals_consistency():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    report = core.process(x, profile=True, return_profile=True)[1]
    assert report is not None

    assert report.totals.calls == sum(bucket.calls for bucket in report.stage_buckets)
    assert report.totals.op_calls == sum(bucket.op_calls for bucket in report.stage_buckets)
    assert report.totals.op_unique == sum(bucket.op_unique for bucket in report.stage_buckets)
    assert report.totals.profiled_flops == sum(
        bucket.profiled_flops for bucket in report.stage_buckets
    )
    assert report.totals.time_cpu_total_us == sum(
        bucket.time_cpu_total_us for bucket in report.stage_buckets
    )
    assert report.totals.cpu_mem_total_bytes == sum(
        bucket.cpu_mem_total_bytes for bucket in report.stage_buckets
    )


def test_profile_with_analytical_cost_attached():
    core = _build_profiled_core(block_size=4)
    x = torch.randn(12, 1)

    config = ProcessProfileConfig(include_analytical=True)
    report = core.process(x, profile=True, profile_config=config, return_profile=True)[1]
    assert report is not None

    assert report.analytical_block_cost is not None
    assert report.analytical_total_cost is not None
    for bucket in report.stage_buckets:
        assert bucket.analytical_cost is not None


def test_profile_2d_and_3d_input_shapes():
    core_2d = _build_profiled_core(block_size=4)
    x_2d = torch.randn(12, 1)
    y_2d, report_2d = core_2d.process(x_2d, profile=True, return_profile=True)
    assert y_2d.shape == (12, 1)
    assert report_2d is not None
    assert report_2d.batch_size == 1

    core_3d = _build_profiled_core(block_size=4)
    x_3d = torch.randn(2, 1, 12)
    y_3d, report_3d = core_3d.process(x_3d, profile=True, return_profile=True)
    assert y_3d.shape == (2, 1, 12)
    assert report_3d is not None
    assert report_3d.batch_size == 2
