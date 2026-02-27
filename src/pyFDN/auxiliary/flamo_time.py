"""
Prototype FLAMO-like time-domain graph modules.

This module mirrors the FLAMO-style API (`dsp` and `system` namespaces) with
block-based time-domain processing:

- dsp.Gain
- dsp.parallelDelay
- dsp.parallelSOSFilter
- system.Series
- system.Parallel
- system.Recursion
- system.Shell

The recursion implementation is intentionally block-causal with an explicit
feedback delay of at least one block (`feedback_delay_blocks=1` by default).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .flamo_graph import flamo_model_to_nodes


def _channels_from_size(size: Sequence[int] | tuple[int, ...]) -> int:
    if len(size) != 1:
        raise ValueError(f"Expected size=(N,), got {tuple(size)}")
    channels = int(size[0])
    if channels <= 0:
        raise ValueError(f"Number of channels must be positive, got {channels}")
    return channels


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _extract_module_value(module: Any) -> torch.Tensor | None:
    for attr in ("value", "_value", "values", "coeffs"):
        candidate = getattr(module, attr, None)
        if candidate is not None:
            try:
                return _as_tensor(candidate)
            except Exception:
                pass

    getter = getattr(module, "get_value", None)
    if callable(getter):
        try:
            return _as_tensor(getter())
        except Exception:
            return None
    return None


class _BlockModule:
    input_channels: int
    output_channels: int

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del batch_size, block_size, device, dtype

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x_block: torch.Tensor) -> torch.Tensor:
        return self.forward_block(x_block)


class Gain(_BlockModule):
    """Block gain/matrix stage: y = G @ x."""

    def __init__(
        self,
        size: tuple[int, int],
        nfft: int | None = None,
        *,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: torch.device | str | None = None,
    ):
        del nfft, requires_grad, alias_decay_db
        n_out, n_in = int(size[0]), int(size[1])
        if n_out <= 0 or n_in <= 0:
            raise ValueError(f"Gain size must be positive, got {size}")
        self.size = (n_out, n_in)
        self.output_channels = n_out
        self.input_channels = n_in
        self.value = torch.zeros(n_out, n_in, dtype=torch.float32)
        if device is not None:
            self.value = self.value.to(torch.device(device))

    def assign_value(self, value: Any) -> None:
        matrix = _as_tensor(value)
        if matrix.ndim != 2:
            raise ValueError(f"Gain value must be 2D, got shape {tuple(matrix.shape)}")
        if tuple(matrix.shape) != self.size:
            raise ValueError(f"Expected shape {self.size}, got {tuple(matrix.shape)}")
        self.value = matrix

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del batch_size, block_size
        self.value = self.value.to(device=device, dtype=dtype)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        if x_block.ndim != 3:
            raise ValueError(f"Gain expects [B, N_in, T], got {tuple(x_block.shape)}")
        if x_block.shape[1] != self.input_channels:
            raise ValueError(
                f"Gain expects {self.input_channels} input channels, got {x_block.shape[1]}"
            )
        matrix = self.value.to(device=x_block.device, dtype=x_block.dtype)
        return torch.einsum("bnt,on->bot", x_block, matrix)


class ParallelDelay(_BlockModule):
    """Per-channel integer delay with circular buffers."""

    def __init__(
        self,
        size: tuple[int, ...] | Sequence[int],
        max_len: int,
        nfft: int | None = None,
        *,
        isint: bool = True,
        unit: int = 1,
        fs: float = 48_000.0,
        requires_grad: bool = False,
        alias_decay_db: float = 0.0,
        device: torch.device | str | None = None,
    ):
        del nfft, requires_grad, alias_decay_db
        self.input_channels = _channels_from_size(size)
        self.output_channels = self.input_channels
        self.size = (self.input_channels,)
        self.max_len = int(max_len)
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        self.isint = bool(isint)
        self.unit = int(unit)
        self.fs = float(fs)
        self.value = torch.zeros(self.input_channels, dtype=torch.float32)
        self.delay_samples = torch.zeros(self.input_channels, dtype=torch.long)
        if device is not None:
            dev = torch.device(device)
            self.value = self.value.to(dev)
            self.delay_samples = self.delay_samples.to(dev)

        self._delay_buffers: torch.Tensor | None = None
        self._delay_pointer: torch.Tensor | None = None
        self._block_size = 0

    @staticmethod
    def sample2s(samples: Any, fs: float = 48_000.0) -> torch.Tensor:
        return _as_tensor(samples) / float(fs)

    @staticmethod
    def s2sample(seconds: Any, fs: float = 48_000.0) -> torch.Tensor:
        return torch.round(_as_tensor(seconds) * float(fs))

    def assign_value(self, value: Any) -> None:
        vector = _as_tensor(value).reshape(-1)
        if vector.numel() != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} delay values, got {vector.numel()}"
            )
        self.value = vector
        if self.unit == 1:
            samples = torch.round(vector * self.fs)
        else:
            samples = torch.round(vector)
        if not self.isint:
            # Prototype supports integer delays only; retain nearest-integer samples.
            samples = torch.round(samples)
        samples = samples.clamp(min=0, max=self.max_len).to(dtype=torch.long)
        self.delay_samples = samples

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del dtype
        delay_samples = self.delay_samples.to(device=device, dtype=torch.long)
        max_delay = int(delay_samples.max().item()) if delay_samples.numel() else 0
        buffer_len = max(1, max_delay + int(block_size))
        self._delay_buffers = torch.zeros(
            batch_size,
            self.input_channels,
            buffer_len,
            device=device,
            dtype=torch.float32,
        )
        self._delay_pointer = torch.zeros(
            batch_size,
            self.input_channels,
            device=device,
            dtype=torch.long,
        )
        self.delay_samples = delay_samples
        self._block_size = int(block_size)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        if x_block.ndim != 3:
            raise ValueError(
                f"parallelDelay expects [B, N_in, T], got {tuple(x_block.shape)}"
            )
        if x_block.shape[1] != self.input_channels:
            raise ValueError(
                "parallelDelay channel mismatch: "
                f"expected {self.input_channels}, got {x_block.shape[1]}"
            )
        if self._delay_buffers is None or self._delay_pointer is None:
            self.initialize_state(
                batch_size=x_block.shape[0],
                block_size=x_block.shape[2],
                device=x_block.device,
                dtype=x_block.dtype,
            )
        if x_block.shape[2] != self._block_size:
            raise ValueError(
                "parallelDelay block size mismatch: "
                f"expected {self._block_size}, got {x_block.shape[2]}"
            )

        buffers = self._delay_buffers
        pointer = self._delay_pointer
        if buffers is None or pointer is None:
            raise RuntimeError("parallelDelay state not initialized")

        _, _, buffer_len = buffers.shape
        block_size = x_block.shape[2]
        delayed = torch.zeros_like(x_block)

        for sample_idx in range(block_size):
            read_indices = (
                pointer - self.delay_samples.view(1, self.input_channels)
            ) % buffer_len
            delayed[:, :, sample_idx] = torch.gather(
                buffers, 2, read_indices.unsqueeze(-1)
            ).squeeze(-1)

            write_indices = pointer.unsqueeze(-1)
            buffers.scatter_(
                2,
                write_indices,
                x_block[:, :, sample_idx].to(dtype=buffers.dtype).unsqueeze(-1),
            )
            pointer = (pointer + 1) % buffer_len

        self._delay_pointer = pointer
        self._delay_buffers = buffers

        return delayed.to(dtype=x_block.dtype)


class ParallelSOSFilter(_BlockModule):
    """Per-channel parallel SOS bank (DF2T), FLAMO-like coefficient layout."""

    def __init__(
        self,
        size: tuple[int, ...] | Sequence[int],
        n_sections: int,
        nfft: int | None = None,
        *,
        device: torch.device | str | None = None,
    ):
        del nfft
        self.input_channels = _channels_from_size(size)
        self.output_channels = self.input_channels
        self.size = (self.input_channels,)
        self.n_sections = int(n_sections)
        if self.n_sections <= 0:
            raise ValueError(f"n_sections must be positive, got {self.n_sections}")
        self.value = torch.zeros(
            self.n_sections, 6, self.input_channels, dtype=torch.float32
        )
        if device is not None:
            self.value = self.value.to(torch.device(device))

        self._state: torch.Tensor | None = None
        self._coeffs: torch.Tensor | None = None
        self._block_size = 0

    def assign_value(self, value: Any) -> None:
        sos = _as_tensor(value)
        if sos.ndim == 2 and sos.shape[0] == 6:
            sos = sos.unsqueeze(0)
        if sos.ndim != 3:
            raise ValueError(f"SOS must have 3 dims, got shape {tuple(sos.shape)}")
        if sos.shape[1] != 6:
            raise ValueError(
                f"SOS second dimension must be 6, got shape {tuple(sos.shape)}"
            )
        if sos.shape[2] != self.input_channels:
            raise ValueError(
                "SOS channel mismatch: "
                f"expected {self.input_channels}, got {sos.shape[2]}"
            )
        if sos.shape[0] != self.n_sections:
            raise ValueError(
                f"Expected {self.n_sections} sections, got {int(sos.shape[0])}"
            )
        self.value = sos

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._coeffs = self.value.to(device=device, dtype=dtype)
        self._state = torch.zeros(
            batch_size,
            self.input_channels,
            self.n_sections,
            2,
            device=device,
            dtype=dtype,
        )
        self._block_size = int(block_size)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        if x_block.ndim != 3:
            raise ValueError(
                f"parallelSOSFilter expects [B, N_in, T], got {tuple(x_block.shape)}"
            )
        if x_block.shape[1] != self.input_channels:
            raise ValueError(
                "parallelSOSFilter channel mismatch: "
                f"expected {self.input_channels}, got {x_block.shape[1]}"
            )
        if self._state is None or self._coeffs is None:
            self.initialize_state(
                batch_size=x_block.shape[0],
                block_size=x_block.shape[2],
                device=x_block.device,
                dtype=x_block.dtype,
            )
        if x_block.shape[2] != self._block_size:
            raise ValueError(
                "parallelSOSFilter block size mismatch: "
                f"expected {self._block_size}, got {x_block.shape[2]}"
            )

        coeffs = self._coeffs
        state = self._state
        if coeffs is None or state is None:
            raise RuntimeError("parallelSOSFilter state not initialized")

        y = x_block
        block_size = y.shape[2]
        for section_idx in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = coeffs[section_idx].unbind(dim=0)
            if torch.any(a0 == 0):
                raise ValueError("SOS has a0=0 in at least one channel")

            inv_a0 = 1.0 / a0
            b0n = b0 * inv_a0
            b1n = b1 * inv_a0
            b2n = b2 * inv_a0
            a1n = a1 * inv_a0
            a2n = a2 * inv_a0

            z1 = state[:, :, section_idx, 0]
            z2 = state[:, :, section_idx, 1]
            output = torch.zeros_like(y)

            for sample_idx in range(block_size):
                x_n = y[:, :, sample_idx]
                y_n = b0n.unsqueeze(0) * x_n + z1
                z1 = b1n.unsqueeze(0) * x_n - a1n.unsqueeze(0) * y_n + z2
                z2 = b2n.unsqueeze(0) * x_n - a2n.unsqueeze(0) * y_n
                output[:, :, sample_idx] = y_n

            state[:, :, section_idx, 0] = z1
            state[:, :, section_idx, 1] = z2
            y = output

        self._state = state
        return y


class Series(_BlockModule):
    """Sequential composition of block modules."""

    def __init__(self, modules: Mapping[str, _BlockModule] | Sequence[tuple[str, _BlockModule]]):
        if hasattr(modules, "items"):
            ordered = OrderedDict(modules.items())  # type: ignore[arg-type]
        else:
            ordered = OrderedDict(modules)
        if not ordered:
            raise ValueError("Series requires at least one submodule")
        self._modules = ordered

        children = list(self._modules.values())
        self.input_channels = children[0].input_channels
        self.output_channels = children[-1].output_channels

        for prev, nxt in zip(children, children[1:]):
            if prev.output_channels != nxt.input_channels:
                raise ValueError(
                    "Series channel mismatch: "
                    f"{prev.output_channels} -> {nxt.input_channels}"
                )

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for module in self._modules.values():
            module.initialize_state(batch_size, block_size, device, dtype)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        y_block = x_block
        for module in self._modules.values():
            y_block = module.forward_block(y_block)
        return y_block


class Parallel(_BlockModule):
    """Two-branch parallel composition with optional output summation."""

    def __init__(self, brA: _BlockModule, brB: _BlockModule, sum_output: bool = True):
        self.brA = brA
        self.brB = brB
        self.sum_output = bool(sum_output)

        if brA.input_channels != brB.input_channels:
            raise ValueError(
                "Parallel input mismatch: "
                f"{brA.input_channels} vs {brB.input_channels}"
            )
        self.input_channels = brA.input_channels

        if self.sum_output:
            if brA.output_channels != brB.output_channels:
                raise ValueError(
                    "Parallel summed output mismatch: "
                    f"{brA.output_channels} vs {brB.output_channels}"
                )
            self.output_channels = brA.output_channels
        else:
            self.output_channels = brA.output_channels + brB.output_channels

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.brA.initialize_state(batch_size, block_size, device, dtype)
        self.brB.initialize_state(batch_size, block_size, device, dtype)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        y_a = self.brA.forward_block(x_block)
        y_b = self.brB.forward_block(x_block)
        if self.sum_output:
            return y_a + y_b
        return torch.cat((y_a, y_b), dim=1)


class Recursion(_BlockModule):
    """Block recursion: y[k] = fF(x[k] + fB(y[k-d])) with d blocks delay."""

    def __init__(self, fF: _BlockModule, fB: _BlockModule, feedback_delay_blocks: int = 1):
        self.fF = fF
        self.fB = fB
        self.feedback_delay_blocks = int(feedback_delay_blocks)
        if self.feedback_delay_blocks < 1:
            raise ValueError(
                "feedback_delay_blocks must be >= 1 for block-causal recursion"
            )

        if self.fB.output_channels != self.fF.input_channels:
            raise ValueError(
                "Recursion mismatch: feedback output channels must match "
                f"forward input ({self.fB.output_channels} vs {self.fF.input_channels})"
            )

        self.input_channels = self.fF.input_channels
        self.output_channels = self.fF.output_channels
        self._feedback_blocks: list[torch.Tensor] = []
        self._block_size = 0

    def initialize_state(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.fF.initialize_state(batch_size, block_size, device, dtype)
        self.fB.initialize_state(batch_size, block_size, device, dtype)
        self._feedback_blocks = [
            torch.zeros(
                batch_size,
                self.fF.input_channels,
                block_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.feedback_delay_blocks)
        ]
        self._block_size = int(block_size)

    def forward_block(self, x_block: torch.Tensor) -> torch.Tensor:
        if x_block.ndim != 3:
            raise ValueError(
                f"Recursion expects [B, N_in, T], got {tuple(x_block.shape)}"
            )
        if x_block.shape[1] != self.input_channels:
            raise ValueError(
                f"Recursion expects {self.input_channels} input channels, "
                f"got {x_block.shape[1]}"
            )
        if not self._feedback_blocks:
            self.initialize_state(
                batch_size=x_block.shape[0],
                block_size=x_block.shape[2],
                device=x_block.device,
                dtype=x_block.dtype,
            )
        if x_block.shape[2] != self._block_size:
            raise ValueError(
                f"Recursion block size mismatch: expected {self._block_size}, "
                f"got {x_block.shape[2]}"
            )

        feedback = self._feedback_blocks.pop(0)
        ff_in = x_block + feedback
        y_block = self.fF.forward_block(ff_in)
        next_feedback = self.fB.forward_block(y_block)
        if next_feedback.shape != ff_in.shape:
            raise ValueError(
                "Recursion feedback shape mismatch: "
                f"expected {tuple(ff_in.shape)}, got {tuple(next_feedback.shape)}"
            )
        self._feedback_blocks.append(next_feedback)
        return y_block


class Shell:
    """Top-level block processor with optional input/output layers."""

    def __init__(
        self,
        core: _BlockModule,
        input_layer: _BlockModule | None = None,
        output_layer: _BlockModule | None = None,
        *,
        block_size: int = 512,
        device: torch.device | str | None = None,
    ):
        self.__core = core
        self.__input_layer = input_layer
        self.__output_layer = output_layer

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = int(block_size)
        self.device = torch.device(device) if device is not None else None

        input_channels = core.input_channels
        if input_layer is not None:
            if input_layer.output_channels != core.input_channels:
                raise ValueError(
                    "input_layer output channels must match core input channels "
                    f"({input_layer.output_channels} vs {core.input_channels})"
                )
            input_channels = input_layer.input_channels
        output_channels = core.output_channels
        if output_layer is not None:
            if output_layer.input_channels != core.output_channels:
                raise ValueError(
                    "output_layer input channels must match core output channels "
                    f"({output_layer.input_channels} vs {core.output_channels})"
                )
            output_channels = output_layer.output_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

    def get_core(self) -> _BlockModule:
        return self.__core

    def get_inputLayer(self) -> _BlockModule | None:
        return self.__input_layer

    def get_outputLayer(self) -> _BlockModule | None:
        return self.__output_layer

    def _initialize_all(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if self.__input_layer is not None:
            self.__input_layer.initialize_state(batch_size, self.block_size, device, dtype)
        self.__core.initialize_state(batch_size, self.block_size, device, dtype)
        if self.__output_layer is not None:
            self.__output_layer.initialize_state(batch_size, self.block_size, device, dtype)

    def process(self, input_signal: torch.Tensor) -> torch.Tensor:
        if input_signal.ndim == 2:
            input_signal = input_signal.T.unsqueeze(0)
            squeeze_output = True
        elif input_signal.ndim == 3:
            squeeze_output = False
        else:
            raise ValueError(
                f"Input must be [T, N_in] or [B, N_in, T], got {tuple(input_signal.shape)}"
            )

        run_device = self.device or input_signal.device
        if run_device.type == "cpu" and input_signal.device.type != "cpu":
            run_device = input_signal.device
        input_signal = input_signal.to(device=run_device)

        batch_size, n_in, total_samples = input_signal.shape
        if n_in != self.input_channels:
            raise ValueError(
                f"Shell expects {self.input_channels} input channels, got {n_in}"
            )

        self._initialize_all(batch_size, run_device, input_signal.dtype)

        output_blocks: list[torch.Tensor] = []
        block_size = self.block_size
        num_blocks = (total_samples + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, total_samples)
            current_len = end - start

            x_block = input_signal[:, :, start:end]
            if current_len < block_size:
                padding = torch.zeros(
                    batch_size,
                    n_in,
                    block_size - current_len,
                    device=run_device,
                    dtype=input_signal.dtype,
                )
                x_block = torch.cat((x_block, padding), dim=2)

            if self.__input_layer is not None:
                x_block = self.__input_layer.forward_block(x_block)
            y_block = self.__core.forward_block(x_block)
            if self.__output_layer is not None:
                y_block = self.__output_layer.forward_block(y_block)

            output_blocks.append(y_block[:, :, :current_len])

        output = torch.cat(output_blocks, dim=2)
        if squeeze_output:
            return output.squeeze(0).T
        return output

    def __call__(self, input_signal: torch.Tensor) -> torch.Tensor:
        return self.process(input_signal)


def gain_module(values: np.ndarray) -> Gain:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"gain_module expects 1D or 2D values, got shape {matrix.shape}")
    module = Gain(size=(int(matrix.shape[0]), int(matrix.shape[1])))
    module.assign_value(matrix)
    return module


def delay_module(lengths_seconds: np.ndarray, *, Fs: float) -> ParallelDelay:
    lengths = np.asarray(lengths_seconds, dtype=np.float32).reshape(-1)
    if lengths.size == 0:
        raise ValueError("delay_module requires at least one delay value")
    max_len = max(1, int(np.ceil(float(lengths.max()) * float(Fs))))
    module = ParallelDelay(size=(int(lengths.size),), max_len=max_len, unit=1, fs=float(Fs))
    module.assign_value(lengths)
    return module


def sos_filter_module(sos: np.ndarray) -> ParallelSOSFilter:
    sos_arr = np.asarray(sos, dtype=np.float32)
    if sos_arr.ndim != 3 or sos_arr.shape[1] != 6:
        raise ValueError("sos must have shape (n_sections, 6, n_channels)")
    n_sections, _, n_channels = sos_arr.shape
    module = ParallelSOSFilter(size=(int(n_channels),), n_sections=int(n_sections))
    module.assign_value(sos_arr)
    return module


def _default_leaf_factory(module: Any, fs: float) -> _BlockModule:
    if isinstance(module, _BlockModule):
        return module

    typename = type(module).__name__
    value = _extract_module_value(module)

    if typename == "Gain" and value is not None and value.ndim == 2:
        gain = Gain(size=(int(value.shape[0]), int(value.shape[1])))
        gain.assign_value(value)
        return gain

    if typename == "parallelDelay" and value is not None and value.ndim == 1:
        max_len = int(getattr(module, "max_len", int(torch.ceil(value * fs).max().item())))
        unit = int(getattr(module, "unit", 1))
        module_fs = float(getattr(module, "fs", fs))
        delay = ParallelDelay(
            size=(int(value.shape[0]),),
            max_len=max(1, max_len),
            unit=unit,
            fs=module_fs,
        )
        delay.assign_value(value)
        return delay

    if typename == "parallelSOSFilter" and value is not None and value.ndim == 3:
        sos = ParallelSOSFilter(size=(int(value.shape[2]),), n_sections=int(value.shape[0]))
        sos.assign_value(value)
        return sos

    raise ValueError(
        "Unsupported FLAMO leaf module for automatic conversion: "
        f"{typename}. Provide a custom leaf_factory."
    )


def flamo_structure_to_time(
    model: Any,
    *,
    block_size: int,
    leaf_factory: Callable[[Any, str], _BlockModule] | None = None,
    include_shell_io: bool = True,
    fs: float = 48_000.0,
) -> Shell:
    """
    Copy FLAMO graph topology into time-domain modules.

    The default converter supports FLAMO-like leaves for Gain, parallelDelay, and
    parallelSOSFilter if parameter tensors are introspectable. For unsupported leaf
    modules, provide a custom `leaf_factory(module, path)` callback.
    """

    root = flamo_model_to_nodes(model, name="root", include_shell_io=include_shell_io)

    def convert(node: dict[str, Any], path: str) -> _BlockModule | Shell:
        node_type = node.get("type")
        if node_type == "Series":
            children = OrderedDict(
                (child["name"], convert(child, f"{path}/{child['name']}"))
                for child in (node.get("children") or [])
            )
            return Series(children)  # type: ignore[arg-type]
        if node_type == "Parallel":
            branches = {child["name"]: convert(child, f"{path}/{child['name']}") for child in (node.get("children") or [])}
            if "brA" not in branches or "brB" not in branches:
                raise ValueError("Parallel conversion requires brA and brB branches")
            return Parallel(
                brA=branches["brA"],  # type: ignore[arg-type]
                brB=branches["brB"],  # type: ignore[arg-type]
                sum_output=True,
            )
        if node_type == "Recursion":
            f_f = node.get("fF")
            f_b = node.get("fB")
            if f_f is None or f_b is None:
                raise ValueError("Recursion conversion requires both fF and fB")
            return Recursion(
                fF=convert(f_f, f"{path}/fF"),  # type: ignore[arg-type]
                fB=convert(f_b, f"{path}/fB"),  # type: ignore[arg-type]
                feedback_delay_blocks=1,
            )
        if node_type == "Shell":
            core_nodes = node.get("children") or []
            if not core_nodes:
                raise ValueError("Shell conversion requires a core child")
            core = convert(core_nodes[0], f"{path}/core")
            input_layer = node.get("input_layer")
            output_layer = node.get("output_layer")
            input_mod = (
                convert(input_layer, f"{path}/input_layer")
                if input_layer is not None
                else None
            )
            output_mod = (
                convert(output_layer, f"{path}/output_layer")
                if output_layer is not None
                else None
            )
            return Shell(
                core=core,  # type: ignore[arg-type]
                input_layer=input_mod,  # type: ignore[arg-type]
                output_layer=output_mod,  # type: ignore[arg-type]
                block_size=block_size,
            )

        raw_module = node.get("module")
        if leaf_factory is not None:
            return leaf_factory(raw_module, path)
        return _default_leaf_factory(raw_module, fs=fs)

    converted = convert(root, "root")
    if isinstance(converted, Shell):
        return converted
    return Shell(core=converted, block_size=block_size)  # type: ignore[arg-type]


# FLAMO-like namespaces
dsp = SimpleNamespace(
    Gain=Gain,
    parallelDelay=ParallelDelay,
    parallelSOSFilter=ParallelSOSFilter,
)
system = SimpleNamespace(
    Series=Series,
    Parallel=Parallel,
    Recursion=Recursion,
    Shell=Shell,
)

