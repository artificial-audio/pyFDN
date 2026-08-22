"""The EQ designs of this package behind one interface.

pyFDN designs the same two filters -- the in-loop absorption that sets an FDN's
decay, and the output EQ that colours it -- at three levels of detail: a
ten-band graphic EQ, a first-order shelf, and a one-pole. :class:`EQDesign` is
their shared shape:

    ``(n_params,)`` targets in dB  ->  ``(n_sections, 6)`` biquads

Every design implements it in whichever array namespace it is handed (see
:mod:`._backend`), so one implementation serves both the numpy design path and
the differentiable ``map`` of a trainable filter in :mod:`pyFDN.train`: a
training loop and a numpy caller bake the same closed form, so a trained FDN and
the :class:`~pyFDN.FDNBuild` extracted from it hold the same coefficients.

A design carries its own target (``GraphicEQ`` ten numbers, the shelf and
one-pole two), which makes the parameter count a constructor-time invariant --
``FirstOrderShelf(np.zeros(10))`` fails where you wrote it. It carries nothing
about where the filter sits (delays, channel count, ``fs``); that belongs to the
*role* (:class:`~pyFDN.DecayFilter`, :class:`~pyFDN.OutputEQ`), which is why the
same design serves both an in-loop absorption and an output EQ.
:attr:`EQDesign.target` is the seed a trainable filter copies into its parameter;
:meth:`EQDesign.sos` still takes a target argument because during training that
is a tensor changing every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .design_geq import N_BANDS, geq_design_matrix, geq_sos
from .design_geq import N_SECTIONS as N_GEQ_SECTIONS
from .first_order import N_ENDPOINTS, first_order_shelf_sos, shelf_crossover_omega
from .one_pole import one_pole_sos


@dataclass(frozen=True, eq=False)
class EQDesign:
    """A filter design, and the target it starts from.

    Subclasses fill in :attr:`n_params`, :attr:`n_sections` and :meth:`sos`.
    Instances are frozen and carry whatever fixed choices the design has (the
    shelf's crossover, for instance), so a design is a value you can pass around
    and store, not a family of functions with keyword arguments.

    Parameters
    ----------
    target : array_like
        The design's target, ``(n_params,)`` or ``(n_params, n_channels)``; a
        scalar is spread across the design's parameters. What it *means* is the
        role's business -- a reverberation time in seconds for
        :class:`~pyFDN.DecayFilter`, a gain in dB for :class:`~pyFDN.OutputEQ`.
    """

    #: Declared as ``ArrayLike`` because that is what a caller may pass -- a
    #: tuple, a list, a scalar. ``__post_init__`` normalizes it, so what is
    #: stored and read back is always an ``(n_params,)`` or
    #: ``(n_params, n_channels)`` float array.
    target: ArrayLike

    #: How many numbers describe the target: one per design band or endpoint.
    n_params: ClassVar[int]
    #: How many biquad sections :meth:`sos` returns.
    n_sections: ClassVar[int]
    #: Human-readable parameter layout, used in error messages.
    param_description: ClassVar[str]

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        if target.ndim == 0:
            target = np.full(self.n_params, float(target))
        if target.ndim > 2:
            raise ValueError(
                f"target must be 1- or 2-dimensional, got shape {target.shape}"
            )
        if target.shape[0] != self.n_params:
            raise ValueError(
                f"{type(self).__name__} takes {self.n_params} values -- "
                f"{self.param_description} -- got {target.shape[0]}"
            )
        object.__setattr__(self, "target", np.ascontiguousarray(target))

    def buffers(self, fs: float) -> dict[str, np.ndarray]:
        """Constants :meth:`sos` needs at ``fs``, to be computed once.

        A trainable filter registers these as torch buffers and hands them back
        to :meth:`sos` on every call, so nothing per-``fs`` is recomputed inside
        a training loop. Designs with no such constants return ``{}``.
        """
        return {}

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        """Targets in dB to an SOS bank -- the closed-form, differentiable map.

        Parameters
        ----------
        target_db : array_like or torch.Tensor
            Target magnitude in dB, shape ``(n_params,)`` or
            ``(n_params, n_channels)``. Passed in rather than read from
            :attr:`target`, which is only the seed: in a training loop this is
            the live parameter.
        fs : float
            Sampling rate in Hz.
        **buffers
            The constants from :meth:`buffers`, when the caller holds its own.

        Returns
        -------
        np.ndarray or torch.Tensor
            SOS bank of shape ``(n_sections, 6) + target_db.shape[1:]``,
            normalized to ``a0 = 1``.
        """
        raise NotImplementedError

    def design(self, fs: float) -> np.ndarray:
        """The SOS bank for this design's own :attr:`target`, in numpy."""
        return self.sos(self.target, fs, **self.buffers(fs))


@dataclass(frozen=True, eq=False)
class GraphicEQ(EQDesign):
    """Ten-band graphic EQ: 10 band targets, 11 biquads.

    The design of Schlecht and Habets (DAFx 2017) -- see :mod:`.design_geq`.
    Uses the closed form of :func:`~.design_geq.geq_sos` rather than the bounded
    solve of :func:`~.design_geq.design_geq`; the bounds are inactive for the
    moderate band gains a decay or an output EQ asks for, and dropping them is
    what lets the numpy and torch paths share one implementation.
    """

    n_params: ClassVar[int] = N_BANDS
    n_sections: ClassVar[int] = N_GEQ_SECTIONS
    param_description: ClassVar[str] = "10 bands (DC, 63 Hz … 8 kHz, Nyquist)"

    def buffers(self, fs: float) -> dict[str, np.ndarray]:
        return {"geq_matrix": geq_design_matrix(fs)}

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        return geq_sos(target_db, fs, design_matrix=buffers.get("geq_matrix"))


@dataclass(frozen=True, eq=False)
class FirstOrderShelf(EQDesign):
    """First-order shelf: DC and Nyquist gains, 1 biquad.

    The design of Jot (AES 2015) -- see :mod:`.first_order`. Its crossover is
    fixed at construction, not trained.
    """

    crossover_frequency: float | None = None

    n_params: ClassVar[int] = N_ENDPOINTS
    n_sections: ClassVar[int] = 1
    param_description: ClassVar[str] = "2 endpoints (DC, Nyquist)"

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        h = 10.0 ** (target_db / 20.0)
        return first_order_shelf_sos(
            h[0], h[1], shelf_crossover_omega(fs, self.crossover_frequency)
        )


@dataclass(frozen=True, eq=False)
class OnePole(EQDesign):
    """One-pole: DC and Nyquist gains, 1 biquad -- see :mod:`.one_pole`."""

    n_params: ClassVar[int] = N_ENDPOINTS
    n_sections: ClassVar[int] = 1
    param_description: ClassVar[str] = "2 endpoints (DC, Nyquist)"

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        h = 10.0 ** (target_db / 20.0)
        return one_pole_sos(h[0], h[1])
