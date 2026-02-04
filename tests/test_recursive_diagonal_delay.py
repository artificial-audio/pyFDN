"""Tests for DiagonalDelay and namespaced delay banks."""

import torch

from pyFDN.recursive import DelayRead, DelayWrite, DiagonalDelay, InputTap, OutputTap, RecursionCore


def _reference_per_line_delay(x: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    """Sample-accurate per-line delay reference (Python loop)."""
    if x.ndim != 2:
        raise ValueError("x must be [T, N]")
    if delays.ndim != 1:
        raise ValueError("delays must be [N]")
    t_total, n = x.shape
    if int(delays.numel()) != n:
        raise ValueError("delays length must match x.shape[1]")
    max_delay = int(torch.max(delays).item()) if n > 0 else 0
    buf_len = max_delay + 1

    buf = torch.zeros((n, buf_len), dtype=x.dtype)
    ptr = 0
    y = torch.zeros_like(x)

    delays_list = [int(d) for d in delays.tolist()]
    for t in range(t_total):
        buf[:, ptr] = x[t, :]
        for ch, d in enumerate(delays_list):
            y[t, ch] = buf[ch, (ptr - d) % buf_len]
        ptr = (ptr + 1) % buf_len

    return y


class TestNamespacedDelayBanks:
    def test_state_isolation(self):
        """Multiple independent delay modules can coexist via state_key namespacing."""
        num_lines = 4
        stages = [
            DelayRead(delay_length=32, num_lines=num_lines, state_key="delay"),
            DiagonalDelay([0, 1, 2, 3], state_key="ffm_d1"),
            DiagonalDelay([4, 5, 6, 7], state_key="ffm_d2"),
            DelayWrite(state_key="delay"),
        ]
        core = RecursionCore(stages, block_size=16)
        state = core.init_state(batch_size=2)

        expected_keys = {
            "delay_buffers",
            "delay_pointer",
            "ffm_d1_buffers",
            "ffm_d1_pointer",
            "ffm_d2_buffers",
            "ffm_d2_pointer",
        }
        assert expected_keys.issubset(set(state.keys()))


class TestDiagonalDelay:
    def test_small_delay_correctness(self):
        """DiagonalDelay is sample-accurate when delay < block_size (within-block)."""
        torch.manual_seed(0)
        num_lines = 4
        delays = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        x = torch.randn(73, num_lines)  # non-multiple of block_size

        stages = [
            InputTap(input_matrix=torch.eye(num_lines)),
            DiagonalDelay(delays, state_key="d"),
            OutputTap(output_matrix=torch.eye(num_lines)),
        ]
        core = RecursionCore(stages, block_size=16)
        y = core.process(x)

        y_ref = _reference_per_line_delay(x, delays)
        assert torch.allclose(y, y_ref, atol=1e-6)

    def test_block_size_invariance(self):
        """DiagonalDelay output is invariant to block size (within tolerance)."""
        torch.manual_seed(0)
        num_lines = 4
        delays = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        x = torch.randn(97, num_lines)

        def run(block_size: int) -> torch.Tensor:
            stages = [
                InputTap(input_matrix=torch.eye(num_lines)),
                DiagonalDelay(delays, state_key="d"),
                OutputTap(output_matrix=torch.eye(num_lines)),
            ]
            core = RecursionCore(stages, block_size=block_size)
            return core.process(x)

        y4 = run(4)
        y8 = run(8)
        y16 = run(16)

        assert torch.allclose(y4, y8, atol=1e-6)
        assert torch.allclose(y8, y16, atol=1e-6)
