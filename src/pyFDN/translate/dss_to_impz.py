"""Impulse responses of DSS systems and complete FDN builds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from pyFDN.translate.dss_to_td import build_to_td, dss_to_td

if TYPE_CHECKING:
    from pyFDN.build import FDNBuild
    from pyFDN.td.operators import TimeOperator


def _impulse_response(graph: TimeOperator, num_inputs: int, ir_len: int) -> np.ndarray:
    """One Dirac per input channel through ``graph``, from zero state each time.

    Returns shape ``(ir_len, num_outputs, num_inputs)``.
    """
    responses = []
    for j in range(num_inputs):
        impulse = np.zeros((ir_len, num_inputs))
        impulse[0, j] = 1.0
        graph.reset()
        responses.append(graph.process_signal(impulse))
    return np.stack(responses, axis=-1)


def dss_to_impz(
    delays: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    ir_len: int,
) -> np.ndarray:
    """
    Compute MIMO impulse response from delay state-space (DSS) representation.

    Runs one simulation per input channel (Dirac at t=0 on that channel only)
    and stacks the results into a single array.

    Parameters
    ----------
    delays : list or array
        Delay lengths in samples
    A, B, C, D : array-like
        Delay state-space matrices (static, numeric only).
        For a complete :class:`pyFDN.FDNBuild` with filter hooks, use
        :func:`pyFDN.build_to_impz`.
    ir_len : int
        Length of impulse response in samples

    Returns
    -------
    impulse_response : ndarray
        Shape [ir_len, num_outputs, num_inputs]
    """
    graph = dss_to_td(delays, A, B, C, D)
    return _impulse_response(graph, np.asarray(B).shape[1], ir_len)


def build_to_impz(build: FDNBuild, ir_len: int) -> np.ndarray:
    """Render an :class:`FDNBuild` to a time-domain impulse response.

    Time-domain sibling of the FLAMO render path (:func:`pyFDN.build_to_flamo`
    -> :func:`pyFDN.flamo_time_response`): renders the :func:`pyFDN.build_to_td`
    graph once per input channel (a Dirac on that channel), from zero state. The graph contains
    the build's three filter hooks as :class:`pyFDN.td.SOSBank` nodes:
    ``post_delay`` on the delay output, ``post_matrix`` on the feedback path,
    and ``post_output`` on the wet signal. Unlike the FFT-based FLAMO render
    this does not time-alias, so a long or near-lossless decay is rendered
    faithfully up to ``ir_len``.

    Extends :func:`dss_to_impz` (numeric state-space only) with the build's
    filter hooks.

    Parameters
    ----------
    build : FDNBuild
        Complete FDN parameters.
    ir_len : int
        Impulse-response length in samples.

    Returns
    -------
    np.ndarray
        Impulse response of shape ``(ir_len, num_outputs, num_inputs)``. Use
        ``.squeeze()`` for a 1-D array from a single-in/single-out FDN.
    """
    return _impulse_response(build_to_td(build), np.asarray(build.B).shape[1], ir_len)
