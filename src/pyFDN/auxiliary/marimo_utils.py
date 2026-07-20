"""marimo display helpers used by the example notebooks.

marimo is an optional dependency (the ``examples`` / ``test`` extras), so it is
imported lazily inside each helper -- importing pyFDN never requires marimo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


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
        signal: Audio samples passed to ``mo.audio``.
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
            mo.audio(np.asarray(signal), rate=int(fs)),
        ],
        gap=gap,
    )
