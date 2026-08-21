"""marimo display helpers used by the example notebooks.

marimo is an optional dependency (the ``examples`` / ``test`` extras), so it is
imported lazily inside each helper -- importing pyFDN never requires marimo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _channel_major(signal: ArrayLike) -> np.ndarray:
    """Orient a 2-D audio array as ``(channels, samples)`` for ``mo.audio``.

    ``mo.audio`` reads a 2-D array as ``[NCHAN, NSAMPLES]``, while pyFDN renders
    are time-major -- ``dss_to_impz`` / ``build_to_impz`` return
    ``(ir_len, num_outputs, num_inputs)`` and ``process_fdn`` returns
    ``(num_samples, num_outputs)``. Handing a stereo ``(num_samples, 2)`` render
    straight to ``mo.audio`` therefore declares ``num_samples`` channels of two
    samples each, and the player plays garbage.

    An audio buffer always has far more samples than channels, so the longer
    axis is the time axis: a 2-D input with more rows than columns is
    transposed, anything else is passed through untouched.
    """
    x = np.asarray(signal)
    if x.ndim == 2 and x.shape[0] > x.shape[1]:
        return x.T
    return x


def labeled_audio(
    label: str,
    signal: ArrayLike,
    *,
    fs: float,
    label_size: str = "1.1em",
    gap: float = 0,
) -> Any:
    """Stack a text ``label`` above an audio player (a marimo element).

    Convenience for A/B listening layouts: returns ``mo.vstack([label, audio])``
    with ``label`` rendered as sized HTML and ``signal`` as an ``mo.audio``
    player at ``fs``. marimo is imported lazily, so this only requires marimo
    when actually called.

    Args:
        label: HTML/text shown above the player.
        signal: Audio samples. 1-D for mono; a 2-D array is treated as
            multi-channel and oriented for ``mo.audio``, so either
            ``(samples, channels)`` (the pyFDN render convention) or
            ``(channels, samples)`` plays correctly.
        fs: Sample rate in Hz.
        label_size: CSS ``font-size`` for the label (default ``"1.1em"``).
        gap: Vertical gap between the label and the player (default 0).

    Returns:
        A marimo ``vstack`` element.
    """
    import marimo as mo

    return mo.vstack(
        [
            mo.Html(label).style({"font-size": label_size}),
            mo.audio(_channel_major(signal), rate=int(fs)),
        ],
        gap=gap,
    )
