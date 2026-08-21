"""The EQ designs of this package behind one interface.

pyFDN designs the same two filters -- the in-loop absorption that sets an FDN's
decay, and the output EQ that colours it -- at three different levels of detail:
a ten-band graphic EQ, a first-order shelf, and a one-pole. They differ in how
many numbers describe them and how many biquads they produce, and in nothing
else that a caller cares about. :class:`EQDesign` is that shared shape:

    ``(n_params,)`` targets in dB  ->  ``(n_sections, 6)`` biquads

Every design implements it in whichever array namespace it is handed (see
:mod:`._backend`), so one implementation serves both the numpy design path and
the differentiable ``map`` of a trainable filter in :mod:`pyFDN.train`.

Two stages, deliberately named apart
------------------------------------

Going from a target magnitude to biquad coefficients is not always one step:

* :meth:`EQDesign.sos` is the **map**. Closed form, differentiable, no solver.
  It is what runs inside a training loop.
* :meth:`EQDesign.fit` is the **design**. It may solve a constrained problem to
  reach a target the map cannot express exactly -- :class:`GraphicEQ` fits its
  eleven command gains to ten band targets by bounded least squares. It is not
  differentiable and is for offline design only.

For the shelf and the one-pole the two coincide: their parameters *are* their
endpoints, so the map already meets the target exactly and ``fit`` is ``sos``.
For the graphic EQ they differ only in the bounds -- :func:`~.design_geq.geq_sos`
folds the same least-squares problem into one constant matrix, dropping the
bounds to buy a closed form. So a trainable graphic EQ still crosses the design
step every step; it just crosses a matrix instead of a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .design_geq import N_BANDS, design_geq, geq_design_matrix, geq_sos
from .design_geq import N_SECTIONS as N_GEQ_SECTIONS
from .first_order import N_ENDPOINTS, first_order_shelf_sos, shelf_crossover_omega
from .one_pole import one_pole_sos


@dataclass(frozen=True)
class EQDesign:
    """A filter design as a map from targets in dB to biquad sections.

    Subclasses fill in :attr:`n_params`, :attr:`n_sections` and :meth:`sos`.
    Instances are frozen and carry whatever fixed choices the design has (the
    shelf's crossover, for instance), so a design is a value you can pass around
    and store, not a family of functions with keyword arguments.
    """

    #: How many numbers describe the target: one per design band or endpoint.
    n_params: ClassVar[int]
    #: How many biquad sections :meth:`sos` returns.
    n_sections: ClassVar[int]
    #: Human-readable parameter layout, used in error messages.
    param_description: ClassVar[str]

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
            ``(n_params, n_channels)``.
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

    def fit(self, target_db: Any, fs: float) -> Any:
        """Best sections for a target -- the offline design, bounds and all.

        Defaults to :meth:`sos`, which is exact for designs whose parameters are
        their own targets. :class:`GraphicEQ` overrides it with the constrained
        solve. Not differentiable in general; use :meth:`sos` for training.
        """
        return self.sos(np.asarray(target_db, dtype=float), fs, **self.buffers(fs))


@dataclass(frozen=True)
class GraphicEQ(EQDesign):
    """Ten-band graphic EQ: 10 band targets, 11 biquads.

    The design of Schlecht and Habets (DAFx 2017) -- see :mod:`.design_geq`.
    """

    n_params: ClassVar[int] = N_BANDS
    n_sections: ClassVar[int] = N_GEQ_SECTIONS
    param_description: ClassVar[str] = "10 bands (DC, 63 Hz … 8 kHz, Nyquist)"

    def buffers(self, fs: float) -> dict[str, np.ndarray]:
        return {"geq_matrix": geq_design_matrix(fs)}

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        return geq_sos(target_db, fs, design_matrix=buffers.get("geq_matrix"))

    def fit(self, target_db: Any, fs: float) -> Any:
        """The bounded least-squares design of :func:`pyFDN.design_geq`."""
        return design_geq(np.asarray(target_db, dtype=float), fs=fs)[0]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class OnePole(EQDesign):
    """One-pole: DC and Nyquist gains, 1 biquad -- see :mod:`.one_pole`."""

    n_params: ClassVar[int] = N_ENDPOINTS
    n_sections: ClassVar[int] = 1
    param_description: ClassVar[str] = "2 endpoints (DC, Nyquist)"

    def sos(self, target_db: Any, fs: float, **buffers: Any) -> Any:
        h = 10.0 ** (target_db / 20.0)
        return one_pole_sos(h[0], h[1])


def default_design(
    n_params: int, *, crossover_frequency: float | None = None
) -> EQDesign:
    """The design a target of ``n_params`` numbers means when none is named.

    Two numbers are a first-order shelf and ten are a graphic EQ, which is the
    dispatch :func:`pyFDN.trainable_from_build` has always done on the length of
    ``absorption_rt``. :class:`OnePole` shares the shelf's parameter count and so
    is never chosen by length -- pass it explicitly to use it.
    """
    if n_params == N_ENDPOINTS:
        return FirstOrderShelf(crossover_frequency)
    if n_params == N_BANDS:
        return GraphicEQ()
    raise ValueError(
        f"no EQ design takes {n_params} parameters: expected "
        f"{N_ENDPOINTS} (first-order shelf) or {N_BANDS} (graphic EQ)"
    )
