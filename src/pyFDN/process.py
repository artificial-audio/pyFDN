"""Compact DSS processing and convenience rendering of complete FDN builds."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.build import FDNBuild
from pyFDN.translate.dss_to_td import build_to_td, dss_to_td


def process_dss(
    input_signal: ArrayLike,
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    *,
    post_delay: Any | None = None,
    post_matrix: Any | None = None,
    post_output: Any | None = None,
) -> np.ndarray:
    """Process a delay state-space system using block processing.

    One-shot form of ``dss_to_td(...).process_signal(input_signal)``: a fresh
    :mod:`pyFDN.td` graph is built for every call. The three optional hooks are
    runtime objects that implement ``process_block(block)`` (or SOS banks). To
    process an :class:`pyFDN.FDNBuild`, use :func:`pyFDN.process_fdn`.

    The recursion is

    ``delay -> post_delay -> C`` on the wet path and
    ``delay -> post_delay -> A -> post_matrix -> + B input`` in the loop.
    ``post_output`` processes the wet signal before the direct ``D`` path is
    added.

    Parameters
    ----------
    input_signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.
    delays
        Positive delay lengths in samples, shape ``(N,)``.
    A
        Static feedback matrix ``(N, N)`` or FIR polynomial matrix
        ``(N, N, order)`` in the ``z^-1`` convention.
    B, C, D
        Static input, output, and direct gain matrices.
    post_delay, post_matrix, post_output
        Optional hooks, see :func:`pyFDN.dss_to_td`.

    Returns
    -------
    np.ndarray
        Processed signal with singleton dimensions removed.
    """
    x = np.asarray(input_signal, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("Input signal must be a 1-D or 2-D array")
    graph = dss_to_td(
        delays,
        A,
        B,
        C,
        D,
        post_delay=post_delay,
        post_matrix=post_matrix,
        post_output=post_output,
    )
    return graph.process_signal(x, squeeze=True)


def process_fdn(input_signal: ArrayLike, build: FDNBuild) -> np.ndarray:
    """Process a signal through a fresh time-domain graph built from ``build``.

    This is the one-shot convenience form of
    ``build_to_td(build).process_signal(input_signal)``. A fresh graph is built
    for every call, so delay and filter state cannot leak between independent
    renders. Use :func:`pyFDN.build_to_td` directly to process a stream over
    multiple calls to :meth:`pyFDN.td.TimeOperator.process_block` or to reset
    and reuse the graph.

    Parameters
    ----------
    input_signal
        Input of shape ``(num_samples,)`` or ``(num_samples, num_inputs)``.
    build
        Complete baked FDN configuration.

    Returns
    -------
    np.ndarray
        Processed signal with singleton dimensions removed, matching the
        historical ``process_fdn`` output convention.
    """
    return build_to_td(build).process_signal(input_signal, squeeze=True)
