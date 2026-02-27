"""Tests for the FLAMO-like time-domain prototype graph."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyFDN.auxiliary.flamo_time import dsp, system
from pyFDN.translate.dss_to_time import dss_to_time


def test_parallel_gain_sum() -> None:
    gain_a = dsp.Gain(size=(1, 1))
    gain_a.assign_value(torch.tensor([[1.0]]))
    gain_b = dsp.Gain(size=(1, 1))
    gain_b.assign_value(torch.tensor([[2.0]]))

    model = system.Shell(
        core=system.Parallel(brA=gain_a, brB=gain_b, sum_output=True),
        block_size=4,
    )

    x = torch.arange(0, 8, dtype=torch.float32).unsqueeze(1)
    y = model.process(x)

    assert y.shape == (8, 1)
    assert torch.allclose(y[:, 0], 3.0 * x[:, 0], atol=1e-6)


def test_parallel_delay_sample_domain() -> None:
    delay = dsp.parallelDelay(size=(1,), max_len=32, unit=0)
    delay.assign_value(torch.tensor([3.0]))  # 3 samples
    model = system.Shell(core=delay, block_size=4)

    x = torch.zeros(12, 1)
    x[0, 0] = 1.0
    y = model.process(x)

    assert y.shape == (12, 1)
    assert y[0, 0].item() == 0.0
    assert y[3, 0].item() == 1.0


def test_recursion_uses_one_block_feedback_delay() -> None:
    forward_gain = dsp.Gain(size=(1, 1))
    forward_gain.assign_value(torch.tensor([[1.0]]))

    feedback_gain = dsp.Gain(size=(1, 1))
    feedback_gain.assign_value(torch.tensor([[0.5]]))

    model = system.Shell(
        core=system.Recursion(fF=forward_gain, fB=feedback_gain, feedback_delay_blocks=1),
        block_size=4,
    )

    x = torch.zeros(16, 1)
    x[0, 0] = 1.0
    y = model.process(x)

    expected_peaks = {
        0: 1.0,
        4: 0.5,
        8: 0.25,
        12: 0.125,
    }
    for idx, amp in expected_peaks.items():
        assert y[idx, 0].item() == pytest.approx(amp, abs=1e-6)


def test_dss_to_time_builds_and_runs() -> None:
    A = np.array([[0.6]], dtype=np.float32)
    B = np.array([[1.0]], dtype=np.float32)
    C = np.array([[1.0]], dtype=np.float32)
    D = np.array([[0.0]], dtype=np.float32)
    m = np.array([2.0], dtype=np.float32)  # samples

    model = dss_to_time(A, B, C, D, m, Fs=48_000.0, block_size=4)
    x = torch.zeros(24, 1)
    x[0, 0] = 1.0
    y = model.process(x)

    assert y.shape == (24, 1)
    assert torch.all(torch.isfinite(y))
    assert y[2, 0].item() == pytest.approx(1.0, abs=1e-6)

