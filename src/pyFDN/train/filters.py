"""FLAMO modules whose parameters are meaningful EQ targets.

``AttenuationFilter`` maps reverberation time to an in-loop SOS bank;
``OutputEQ`` maps gains in dB to a post-output SOS bank. Both use the
same static design functions as NumPy callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flamo.processor import dsp

from ..auxiliary.flamo import default_device
from ..eq.design import (
    EQDesign,
    _decay_to_gain_db,
    _design_buffers,
    _design_parameter_count,
    _design_section_count,
    _gain_to_design,
    _validate_filter_design,
)

MAX_ATTENUATION_DB = 60.0


def _filter_target(
    value: Any,
    value_nyquist: Any | None,
    design: EQDesign,
    n_channels: int,
    *,
    broadcast: bool,
) -> np.ndarray:
    """A design target as ``(n_parameters,)`` or ``(n_parameters, n_channels)``.

    ``graphic_eq`` takes one array of band values; the endpoint designs take a
    DC value and a Nyquist value (defaulting to the DC one, i.e. flat).
    """
    if design == "graphic_eq":
        if value_nyquist is not None:
            raise ValueError("graphic_eq uses one target array, not a Nyquist target")
        n_parameters = _design_parameter_count(design)
        target = np.asarray(value, dtype=np.float64)
        if target.ndim == 0:
            target = np.full(n_parameters, float(target))
        if target.ndim not in (1, 2):
            raise ValueError(
                f"target must be 1- or 2-dimensional, got shape {target.shape}"
            )
        if target.shape[0] != n_parameters:
            raise ValueError(
                f"{design} takes {n_parameters} values, got {target.shape[0]}"
            )
    else:
        nyquist = value if value_nyquist is None else value_nyquist
        target = np.stack(
            np.broadcast_arrays(
                np.asarray(value, dtype=np.float64),
                np.asarray(nyquist, dtype=np.float64),
            )
        )
        if target.ndim not in (1, 2):
            raise ValueError(
                "endpoint targets must be scalar or one value per channel, "
                f"got shape {target.shape}"
            )

    if target.ndim == 2 and target.shape[1] != n_channels:
        raise ValueError(
            f"a per-channel target must have {n_channels} columns, "
            f"got {target.shape[1]}"
        )
    if broadcast and target.ndim == 1:
        target = np.broadcast_to(target[:, None], (target.shape[0], n_channels))
    return np.ascontiguousarray(target)


class _DesignedSOS(dsp.parallelSOSFilter):  # type: ignore[misc]
    def __init__(
        self,
        value: np.ndarray,
        n_channels: int,
        fs: float,
        *,
        design: EQDesign,
        crossover: float | None,
        nfft: int,
        alias_decay_db: float,
        device: Any,
        dtype: Any,
        requires_grad: bool,
    ) -> None:
        import torch

        _validate_filter_design(design)
        super().__init__(
            size=(int(n_channels),),
            n_sections=_design_section_count(design),
            nfft=nfft,
            fs=int(fs),
            alias_decay_db=alias_decay_db,
            device=default_device(device),
            dtype=torch.float32 if dtype is None else dtype,
            normalize_a0=False,
        )
        torch_dtype, dev = self.param.dtype, self.param.device  # type: ignore[has-type]
        self.design: EQDesign = design
        self.crossover = None if crossover is None else float(crossover)
        self.fs_hz = float(fs)
        buffers = _design_buffers(design, self.fs_hz)
        self._design_buffer_names = tuple(buffers)
        for name, buffer in buffers.items():
            self.register_buffer(
                name, torch.tensor(buffer, dtype=torch_dtype, device=dev)
            )
        self.param = torch.nn.Parameter(
            torch.tensor(value, dtype=torch_dtype, device=dev),
            requires_grad=requires_grad,
        )

    def design_sos(self, gain_db: Any) -> Any:
        buffers = {name: getattr(self, name) for name in self._design_buffer_names}
        return _gain_to_design(
            gain_db,
            self.design,
            self.fs_hz,
            crossover=self.crossover,
            **buffers,
        )


class AttenuationFilter(_DesignedSOS):
    """Parallel in-loop SOS bank parametrized by reverberation time.

    For ``graphic_eq``, ``rt`` is the ten-band target on
    :data:`~pyFDN.eq.COMMAND_FREQUENCIES` (31.25 Hz through 16 kHz).
    First-order shelves and one-pole filters use ``rt`` at DC and the
    separately named ``rt_nyquist`` target; omitting the latter creates a
    flat target. Targets may additionally carry one value per delay line.
    The filter is implemented with FLAMO's ``parallelSOSFilter`` because an
    FDN applies one SOS cascade to each delay line in parallel.
    """

    def __init__(
        self,
        rt: Any,
        delays: Any,
        fs: float,
        *,
        rt_nyquist: Any | None = None,
        design: EQDesign = "graphic_eq",
        rt_crossover: float | None = None,
        nfft: int = 2**14,
        alias_decay_db: float = 0.0,
        device: Any = None,
        dtype: Any = None,
        requires_grad: bool = True,
    ) -> None:
        import torch

        _validate_filter_design(design)
        delays_array = np.asarray(delays, dtype=np.float64).ravel()
        rt_array = _filter_target(
            rt,
            rt_nyquist,
            design,
            delays_array.size,
            broadcast=False,
        )
        super().__init__(
            rt_array,
            delays_array.size,
            fs,
            design=design,
            crossover=rt_crossover,
            nfft=nfft,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        torch_dtype, dev = self.param.dtype, self.param.device
        self.register_buffer(
            "delays_samples",
            torch.tensor(delays_array, dtype=torch_dtype, device=dev),
        )
        floor = 60.0 / MAX_ATTENUATION_DB * delays_array / float(fs)
        self.register_buffer(
            "rt_floor",
            torch.tensor(
                floor if rt_array.ndim == 2 else floor.max(),
                dtype=torch_dtype,
                device=dev,
            ),
        )
        self.map = self.rt_to_sos

    def rt_to_sos(self, rt: Any) -> Any:
        per_line = rt if rt.ndim == 2 else rt[:, None]
        gain_db = _decay_to_gain_db(
            self._floored(per_line), self.delays_samples, self.fs_hz
        )
        return self.design_sos(gain_db)

    def _floored(self, rt: Any) -> Any:
        import torch

        floor = self.rt_floor
        return floor * (1.0 + torch.nn.functional.softplus((rt - floor) / floor))


class OutputEQ(_DesignedSOS):
    """Parallel SOS bank parametrized by gain in dB.

    For ``graphic_eq``, ``gain_db`` is the ten-band target on
    :data:`~pyFDN.eq.COMMAND_FREQUENCIES` (31.25 Hz through 16 kHz).
    First-order shelves and one-pole filters use ``gain_db`` at DC and the
    separately named ``gain_db_nyquist`` target; omitting it creates a
    flat target.
    """

    def __init__(
        self,
        gain_db: Any,
        n_channels: int,
        fs: float,
        *,
        gain_db_nyquist: Any | None = None,
        design: EQDesign = "graphic_eq",
        crossover: float | None = None,
        nfft: int = 2**14,
        alias_decay_db: float = 0.0,
        device: Any = None,
        dtype: Any = None,
        requires_grad: bool = True,
    ) -> None:
        _validate_filter_design(design)
        gains = _filter_target(
            gain_db,
            gain_db_nyquist,
            design,
            int(n_channels),
            broadcast=True,
        )
        super().__init__(
            gains,
            int(n_channels),
            fs,
            design=design,
            crossover=crossover,
            nfft=nfft,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        self.map = self.gain_to_sos

    def gain_to_sos(self, gain_db: Any) -> Any:
        return self.design_sos(gain_db)


__all__ = [
    "AttenuationFilter",
    "EQDesign",
    "MAX_ATTENUATION_DB",
    "OutputEQ",
]
