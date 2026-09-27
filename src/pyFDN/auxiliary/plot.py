"""Plot utilities (matrix heatmap, system matrix layout, impulse response grid).

Every function returns a Matplotlib figure (:func:`animate` returns a
Matplotlib animation). Figures are built without pyplot, so they do not
accumulate in pyplot's global state; a notebook (marimo, Jupyter) displays a
figure that is the last expression of a cell.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def downsample_minmax(
    x: ArrayLike | None,
    y: ArrayLike,
    *,
    max_points: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a line while preserving local minima and maxima.

    This is intended for dense time-domain traces such as impulse responses,
    where naive stride decimation can miss narrow peaks. The returned samples
    are sorted by their original order, include the first and last sample, and
    use at most ``max_points`` points for long inputs.

    Parameters
    ----------
    x : array-like or None
        X-values. If None, uses sample indices ``0 .. len(y)-1``.
    y : array-like
        Real-valued y-values.
    max_points : int, optional
        Maximum number of samples to return. Must be at least 4.

    Returns
    -------
    x_ds, y_ds : ndarray
        Downsampled x- and y-values.
    """
    if max_points < 4:
        raise ValueError("max_points must be at least 4")

    y_arr = np.asarray(y).ravel()
    if np.iscomplexobj(y_arr):
        raise ValueError("y must be real-valued")

    if x is None:
        x_arr = np.arange(y_arr.size)
    else:
        x_arr = np.asarray(x).ravel()

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    n_samples = y_arr.size
    if n_samples <= max_points:
        return x_arr, y_arr
    if n_samples == 0:
        return x_arr, y_arr

    # Two extrema per bin plus first/last point keeps the point count bounded.
    n_bins = max(1, (max_points - 2) // 2)
    edges = np.linspace(0, n_samples, n_bins + 1, dtype=int)
    indices: set[int] = {0, n_samples - 1}

    for start, stop in zip(edges[:-1], edges[1:], strict=False):
        if stop <= start:
            continue
        segment = y_arr[start:stop]
        finite = np.isfinite(segment)
        if not np.any(finite):
            indices.add(start)
            continue

        finite_positions = np.flatnonzero(finite)
        finite_segment = segment[finite]
        indices.add(start + int(finite_positions[np.argmin(finite_segment)]))
        indices.add(start + int(finite_positions[np.argmax(finite_segment)]))

    ordered = np.fromiter(sorted(indices), dtype=int)
    return x_arr[ordered], y_arr[ordered]


def downsample_lttb(
    x: ArrayLike | None,
    y: ArrayLike,
    *,
    max_points: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a line with Largest-Triangle-Three-Buckets.

    LTTB keeps points that preserve the visual shape of the connected line. It
    is a better default for line plots than min/max bucketing,
    because it avoids artificial vertical segments between bucket extrema.
    """
    if max_points < 3:
        raise ValueError("max_points must be at least 3")

    y_arr = np.asarray(y).ravel()
    if np.iscomplexobj(y_arr):
        raise ValueError("y must be real-valued")

    if x is None:
        x_arr = np.arange(y_arr.size, dtype=float)
    else:
        x_arr = np.asarray(x).ravel()

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    n_samples = y_arr.size
    if n_samples <= max_points:
        return x_arr, y_arr
    if n_samples == 0:
        return x_arr, y_arr

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(finite):
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size <= max_points:
            return x_arr[finite_indices], y_arr[finite_indices]
        x_work = x_arr[finite_indices]
        y_work = y_arr[finite_indices]
        original_indices = finite_indices
    else:
        x_work = x_arr
        y_work = y_arr
        original_indices = np.arange(n_samples)

    n_work = y_work.size
    if n_work <= max_points:
        return x_work, y_work

    bucket_count = max_points - 2
    bucket_size = (n_work - 2) / bucket_count
    sampled = np.empty(max_points, dtype=int)
    sampled[0] = 0
    sampled[-1] = n_work - 1

    anchor = 0
    for i in range(bucket_count):
        bucket_start = int(np.floor(i * bucket_size)) + 1
        bucket_stop = int(np.floor((i + 1) * bucket_size)) + 1

        next_start = int(np.floor((i + 1) * bucket_size)) + 1
        next_stop = int(np.floor((i + 2) * bucket_size)) + 1
        next_stop = min(next_stop, n_work)
        if next_start >= next_stop:
            avg_x = x_work[-1]
            avg_y = y_work[-1]
        else:
            avg_x = np.mean(x_work[next_start:next_stop])
            avg_y = np.mean(y_work[next_start:next_stop])

        bucket_stop = min(bucket_stop, n_work - 1)
        if bucket_start >= bucket_stop:
            selected = bucket_start
        else:
            bucket = np.arange(bucket_start, bucket_stop)
            area = np.abs(
                (x_work[anchor] - avg_x) * (y_work[bucket] - y_work[anchor])
                - (x_work[anchor] - x_work[bucket]) * (avg_y - y_work[anchor])
            )
            selected = int(bucket[np.argmax(area)])

        sampled[i + 1] = selected
        anchor = selected

    sampled = original_indices[np.unique(sampled)]
    return x_arr[sampled], y_arr[sampled]


_DPI = 100  # pixel sizes below are converted to inches at this resolution


def _new_figure(width_px: float, height_px: float) -> Figure:
    """A pyplot-free Matplotlib figure of the given size in pixels."""
    from matplotlib.figure import Figure

    return Figure(figsize=(width_px / _DPI, height_px / _DPI), dpi=_DPI)


def _heatmap(ax: Any, M: np.ndarray, zmin: float, zmax: float) -> Any:
    """Draw ``M`` as an RdBu heatmap with its origin at the top-left."""
    return ax.imshow(
        M,
        cmap="RdBu",
        vmin=zmin,
        vmax=zmax,
        interpolation="nearest",
        aspect="auto",
    )


def _hide_ticks(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def plot_matrix(
    A: ArrayLike,
    title: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    *,
    block_boundaries: Sequence[int] | None = None,
) -> Figure:
    """Plot a single matrix as a heatmap (RdBu, square pixels).

    Parameters
    ----------
    A : array-like
        2-D matrix to visualise.
    title : str, optional
        Figure title.
    zmin, zmax : float, optional
        Color limits. If both None, uses ``(-1, 1)``.
    block_boundaries : sequence of int, optional
        Indices at which to draw dashed dividing lines on both axes, e.g. to
        separate the sub-blocks of a coupled feedback matrix. A boundary at
        index ``k`` is drawn between rows/columns ``k-1`` and ``k``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    A = np.asarray(A, dtype=float)
    zmin, zmax = _shared_color_limits(zmin, zmax)

    n_rows, n_cols = A.shape[:2]
    size = min(max(240.0, 28.0 * max(n_rows, n_cols)), 480.0)
    fig = _new_figure(size, size + (30 if title else 0))
    ax = fig.add_subplot(111)
    _heatmap(ax, A, zmin, zmax)
    ax.set_aspect("equal")
    _hide_ticks(ax)
    for k in block_boundaries or ():
        style: dict[str, Any] = {
            "color": "black",
            "linewidth": 1,
            "linestyle": "--",
            "alpha": 0.5,
        }
        ax.axhline(k - 0.5, **style)
        ax.axvline(k - 0.5, **style)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_matrix_grid(
    matrices: Sequence[ArrayLike],
    *,
    titles: Sequence[str] | None = None,
    ncols: int = 2,
    zmin: float | None = None,
    zmax: float | None = None,
    show_ticks: bool = False,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> Figure:
    """Plot several matrices as a grid of heatmaps sharing one color scale.

    Each matrix is rendered like :func:`plot_matrix` (RdBu, zero-centered,
    top-left origin, square cells). Use this to compare several matrices side
    by side, e.g. a feedback matrix against its nearest orthogonal
    approximations.

    Parameters
    ----------
    matrices : sequence of array-like
        2-D matrices to visualise, filled row by row across the grid.
    titles : sequence of str, optional
        One subplot title per matrix (``"\\n"`` for line breaks).
    ncols : int, optional
        Number of columns in the grid. Default 2.
    zmin, zmax : float, optional
        Shared color limits. If both None, uses (-1, 1).
    show_ticks : bool, optional
        If True, label axes with integer row/column indices. Default False.
    title : str, optional
        Overall figure title.
    height, width : int, optional
        Figure size in pixels. Defaults scale with the grid shape.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mats = [np.asarray(m, dtype=float) for m in matrices]
    if not mats:
        raise ValueError("at least one matrix is required")
    if titles is not None and len(titles) != len(mats):
        raise ValueError("titles must have one entry per matrix")
    zmin, zmax = _shared_color_limits(zmin, zmax)

    n = len(mats)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil division

    if height is None:
        height = 300 * nrows + (60 if title else 20)
    if width is None:
        width = 300 * ncols + 80
    fig = _new_figure(width, height)
    fig.set_layout_engine("constrained")
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()

    image = None
    for idx, (ax, mat) in enumerate(zip(axes, mats, strict=False)):
        image = _heatmap(ax, mat, zmin, zmax)
        ax.set_aspect("equal")
        if show_ticks:
            ax.set_xticks(range(mat.shape[1]))
            ax.set_yticks(range(mat.shape[0]))
        else:
            _hide_ticks(ax)
        if titles is not None:
            ax.set_title(titles[idx])
    for ax in axes[n:]:
        ax.set_visible(False)
    assert image is not None  # at least one matrix was drawn
    fig.colorbar(image, ax=axes[:n].tolist(), shrink=0.8)
    if title:
        fig.suptitle(title)
    return fig


def _system_matrix_blocks(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the system matrix blocks to 2-D arrays and validate shapes."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    # Normalise to 2-D so every block is drawn as a matrix.
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if d.ndim == 0:
        d = d.reshape(1, 1)
    elif d.ndim == 1:
        d = d.reshape(1, -1)

    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have same number of rows as A")
    if c.shape[1] != A.shape[1]:
        raise ValueError("c must have same number of columns as A")
    return A, b, c, d


def _shared_color_limits(zmin: float | None, zmax: float | None) -> tuple[float, float]:
    """Resolve shared heatmap color limits, defaulting to a symmetric (-1, 1)."""
    if zmin is None and zmax is None:
        return -1.0, 1.0
    if zmin is None:
        assert zmax is not None  # both-None case handled above
        return (-abs(zmax) if zmax != 0 else -1.0), zmax
    if zmax is None:
        return zmin, (abs(zmin) if zmin != 0 else 1.0)
    return zmin, zmax


def _draw_system_matrix(
    fig: Figure,
    cells: Sequence[Any],
    blocks: Sequence[np.ndarray],
    zmin: float,
    zmax: float,
) -> list[Any]:
    """Draw the A, b, c, d blocks into four gridspec cells; return the axes."""
    axes = []
    for cell, blk, name in zip(cells, blocks, "Abcd", strict=True):
        ax = fig.add_subplot(cell)
        _heatmap(ax, blk, zmin, zmax)
        _hide_ticks(ax)
        ax.set_title(name)
        axes.append(ax)
    return axes


def plot_system_matrix(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Figure:
    """Plot system matrix [A b; c d] as 2x2 heatmaps, shared RdBu color scale.

    Subplot sizes are proportional to block dimensions so that each matrix element
    (pixel) has the same physical size across all four plots.

    Parameters
    ----------
    A, b, c, d : array-like
        Feedback matrix, input gain, output gain, direct gain.
    zmin, zmax : float, optional
        Shared color limits. If both None, uses (-1, 1).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    A, b, c, d = _system_matrix_blocks(A, b, c, d)
    zmin, zmax = _shared_color_limits(zmin, zmax)

    # Layout: [A (m×n)  b (m×p);  c (q×n)  d (q×p)], sized so one cell has the
    # same physical size in all four subplots.
    m, n = A.shape
    p = b.shape[1]
    q = c.shape[0]
    cell_px = 400.0 / max(m + q, n + p)
    fig = _new_figure(
        cell_px * (n + p) + 100, cell_px * (m + q) + (90 if title else 60)
    )
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[n, p], height_ratios=[m, q])
    axes = _draw_system_matrix(
        fig, [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]], [A, b, c, d], zmin, zmax
    )
    fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.8)
    if title:
        fig.suptitle(title)
    return fig


def _delay_colors(delays_arr: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """One RGBA color per delay line, mapped from the delay length via a colormap."""
    from matplotlib import colormaps

    span = float(delays_arr.max() - delays_arr.min()) if delays_arr.size else 0.0
    if span > 0:
        positions = (delays_arr - delays_arr.min()) / span
    else:
        positions = np.full(delays_arr.shape, 0.5)
    return colormaps[colormap](positions)


def _frequency_axis(w: np.ndarray, fs: float | None) -> np.ndarray:
    return w * fs / (2.0 * np.pi) if fs is not None else w


def _style_frequency_axis(ax: Any, fs: float | None) -> None:
    if fs is not None:
        ax.set_xscale("log")
        ax.set_xlabel("Frequency [Hz]")
    else:
        ax.set_xlabel("Frequency [rad/sample]")
    ax.grid(True, which="both", alpha=0.3)


def _plot_db_per_sample(
    ax: Any,
    sos: ArrayLike,
    delays_arr: np.ndarray,
    *,
    fs: float | None,
    nfft: int,
    colors: np.ndarray,
) -> None:
    """Draw SOS magnitude responses in dB divided by delay length."""
    from scipy.signal import sosfreqz

    from pyFDN.td.operators import SOSBank

    sos_bank = SOSBank(sos).sos  # (N, n_sections, 6)
    for i in range(delays_arr.size):
        w, h = sosfreqz(sos_bank[i], worN=nfft)
        mag_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).tiny)
        x = _frequency_axis(w, fs)
        if fs is not None:  # drop DC for the log frequency axis
            x, mag_db = x[1:], mag_db[1:]
        ax.plot(
            x,
            mag_db / delays_arr[i],
            color=colors[i],
            linewidth=1.2,
            label=f"delay={delays_arr[i]:g}",
        )
    _style_frequency_axis(ax, fs)


def plot_db_per_sample(
    sos: ArrayLike,
    delays: ArrayLike,
    *,
    fs: float | None = None,
    nfft: int = 512,
    title: str | None = None,
) -> Figure:
    """Plot SOS magnitude responses normalized by delay length (dB per sample).

    Each curve is the magnitude response of one delay line's filter cascade
    divided by its delay length, :math:`20 \\log_{10}|H_i| / m_i`. Filters
    designed for a homogeneous decay (a common T60 target) collapse onto the
    same gain-per-sample curve. Curve colors encode the delay length (Viridis,
    short = dark, long = bright).

    Parameters
    ----------
    sos : array-like
        Per-delay-line SOS bank, same layout as
        :class:`pyFDN.td.SOSBank`: ``(n_sections, 6, N)``.
    delays : array-like
        Delay lengths in samples, shape (N,).
    fs : float, optional
        Sample rate in Hz. If given, the responses are plotted over a
        logarithmic frequency axis in Hz; otherwise over rad/sample.
    nfft : int, optional
        Number of frequency points. Default 512.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    delays_arr = np.asarray(delays, dtype=float).ravel()
    fig = _new_figure(800, 420)
    ax = fig.add_subplot(111)
    _plot_db_per_sample(
        ax, sos, delays_arr, fs=fs, nfft=nfft, colors=_delay_colors(delays_arr)
    )
    ax.set_ylabel("Magnitude [dB/sample]")
    ax.legend(fontsize="small", loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_fdn_parameter(
    delays: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    post_delay_sos: ArrayLike | None = None,
    post_matrix_sos: ArrayLike | None = None,
    post_output_sos: ArrayLike | None = None,
    fs: float | None = None,
    nfft: int = 512,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Figure:
    """Plot all FDN parameters in one figure.

    Extends :func:`plot_system_matrix` with the delay lengths and, optionally,
    any of the three filter hooks, named as :func:`pyFDN.process_dss` and
    :class:`pyFDN.FDNBuild` name them:

    - the system matrix blocks ``A``, ``b``, ``c``, ``d`` as heatmaps with a
      shared RdBu color scale;
    - the delays as a bar plot whose bars are aligned with the columns of the
      feedback matrix ``A`` (one bar per delay line);
    - ``post_delay`` and ``post_matrix``, the two in-loop hooks, as
      gain-per-sample curves, as in :func:`plot_db_per_sample` -- both are per
      delay line and both act once per round trip, so the two rows are directly
      comparable;
    - ``post_output`` as plain magnitude response in dB.

    Bar and curve colors are matched per delay line and encode the delay
    length (Viridis, short = dark, long = bright).

    Parameters
    ----------
    delays : array-like
        Delay lengths in samples, shape (N,).
    A, b, c, d : array-like
        Feedback matrix, input gains, output gains, direct gains.
    post_delay_sos, post_matrix_sos : array-like, optional
        Per-delay-line SOS banks for the two in-loop hooks, same layout as
        :class:`pyFDN.td.SOSBank`: ``(n_sections, 6, N)``.
    post_output_sos : array-like, optional
        Output EQ as an SOS cascade in scipy format, shape ``(n_sections, 6)``
        (or ``(6,)`` for one section) for a single output, or
        ``(n_sections, 6, K)`` to draw one magnitude curve per output channel.
    fs : float, optional
        Sample rate in Hz. If given, the filter responses are plotted over a
        logarithmic frequency axis in Hz; otherwise over rad/sample.
    nfft : int, optional
        Number of frequency points for the filter responses. Default 512.
    zmin, zmax : float, optional
        Shared color limits for the heatmaps. If both None, uses (-1, 1).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.signal import sosfreqz

    A, b, c, d = _system_matrix_blocks(A, b, c, d)
    zmin, zmax = _shared_color_limits(zmin, zmax)
    delays_arr = np.asarray(delays, dtype=float).ravel()

    m, n = A.shape
    if delays_arr.size != n:
        raise ValueError("delays must have one entry per column of A")
    N = delays_arr.size

    # Color encodes the delay length, shared between bars and in-loop curves.
    colors = _delay_colors(delays_arr)

    # Row layout: delays | A b | c d | [post_delay] | [post_matrix] | [post_output]
    in_loop = [
        (sos, label)
        for sos, label in (
            (post_delay_sos, "post_delay [dB/sample]"),
            (post_matrix_sos, "post_matrix [dB/sample]"),
        )
        if sos is not None
    ]
    has_post_output = post_output_sos is not None
    matrix_px = 440.0
    row_px = [
        110.0,
        matrix_px * m / (m + c.shape[0]),
        matrix_px * c.shape[0] / (m + c.shape[0]),
    ]
    row_px += [190.0] * (len(in_loop) + int(has_post_output))
    total_px = float(sum(row_px))

    fig = _new_figure(560, total_px + 120 + (40 if title else 0))
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(
        len(row_px), 2, height_ratios=row_px, width_ratios=[n, b.shape[1]]
    )

    # The A, b, c, d heatmaps, as in plot_system_matrix.
    ax_A, *_ = _draw_system_matrix(
        fig, [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]], [A, b, c, d], zmin, zmax
    )

    # Delays as bars aligned with the columns of A.
    ax_delays = fig.add_subplot(gs[0, 0], sharex=ax_A)
    ax_delays.bar(np.arange(N), delays_arr, color=colors, width=0.8)
    ax_delays.set_ylabel("Delays [samples]")
    ax_delays.tick_params(labelbottom=False)
    ax_A.set_xlim(-0.5, N - 0.5)

    next_row = 3
    for sos, y_label in in_loop:
        ax = fig.add_subplot(gs[next_row, :])
        _plot_db_per_sample(ax, sos, delays_arr, fs=fs, nfft=nfft, colors=colors)
        ax.set_ylabel(y_label)
        next_row += 1

    if has_post_output:
        sos_eq = np.asarray(post_output_sos, dtype=float)
        if sos_eq.ndim == 1:
            sos_eq = sos_eq.reshape(1, 6, 1)
        elif sos_eq.ndim == 2:
            sos_eq = sos_eq[:, :, None]
        if sos_eq.ndim != 3 or sos_eq.shape[1] != 6:
            raise ValueError(
                "post_output_sos must have shape (n_sections, 6) or (n_sections, 6, K)"
            )
        n_out = sos_eq.shape[2]
        if n_out == 1:
            eq_colors: Any = ["black"]
        else:
            from matplotlib import colormaps

            eq_colors = colormaps["plasma"](np.linspace(0.0, 0.9, n_out))
        ax = fig.add_subplot(gs[next_row, :])
        for k in range(n_out):
            w, h = sosfreqz(sos_eq[:, :, k], worN=nfft)
            mag_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).tiny)
            x = _frequency_axis(w, fs)
            if fs is not None:
                x, mag_db = x[1:], mag_db[1:]
            ax.plot(
                x,
                mag_db,
                color=eq_colors[k],
                linewidth=1.5,
                label=f"out {k}" if n_out > 1 else "post_output",
            )
        _style_frequency_axis(ax, fs)
        ax.set_ylabel("post_output [dB]")
        if n_out > 1:
            ax.legend(fontsize="small", loc="best")

    if title:
        fig.suptitle(title)
    return fig


def plot_fdn_build(
    build: Any,
    *,
    nfft: int = 512,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the parameters stored in an :class:`pyFDN.FDNBuild`.

    This is a convenience wrapper around :func:`plot_fdn_parameter`. A
    multichannel ``build.post_output`` is rendered as one curve per output
    channel. Each of the build's three filter hooks becomes its own row, and
    only the ones it carries.
    """
    return plot_fdn_parameter(
        build.delays,
        build.A,
        build.B,
        build.C,
        build.D,
        post_delay_sos=build.post_delay,
        post_matrix_sos=build.post_matrix,
        post_output_sos=build.post_output,
        fs=build.fs,
        nfft=nfft,
        zmin=zmin,
        zmax=zmax,
        title=title,
    )


# Historical spelling, kept as an alias.
plot_FDN_build = plot_fdn_build  # noqa: N816


def _time_axis(n_samples: int, fs: float | None) -> np.ndarray:
    return np.arange(n_samples) / fs if fs is not None else np.arange(n_samples)


def _line_figure(
    ax_fn: Callable[[Any], None],
    *,
    fs: float | None,
    ylabel: str,
    title: str | None,
    legend: bool,
) -> Figure:
    """A single-axes time plot, filled by ``ax_fn``, with shared styling."""
    fig = _new_figure(800, 420)
    ax = fig.add_subplot(111)
    ax_fn(ax)
    ax.set_xlabel("Time [s]" if fs is not None else "Time [samples]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_impulse_response(
    *irs: ArrayLike,
    fs: float | None = None,
    labels: Sequence[str] | None = None,
    mulaw: bool = True,
    mu: float = 255.0,
    title: str | None = "Impulse response",
    max_points: int = 10_000,
) -> Figure:
    """Plot one or more impulse responses over time, mu-law compressed by default.

    Mu-law companding (:func:`pyFDN.mulaw_encode`) keeps the quiet late part of
    a reverberant decay visible alongside the early reflections. Dense traces
    are downsampled with LTTB before plotting.

    Parameters
    ----------
    *irs : array-like
        One or more 1-D impulse responses, plotted as overlaid lines.
    fs : float, optional
        Sample rate in Hz. If given, the time axis is in seconds; otherwise in
        samples.
    labels : sequence of str, optional
        One legend label per impulse response.
    mulaw : bool, optional
        Apply mu-law companding to the amplitudes. Default True.
    mu : float, optional
        Mu-law compression parameter. Default 255 (G.711).
    title : str, optional
        Figure title.
    max_points : int, optional
        Maximum number of points per trace after downsampling. Default 10000.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from pyFDN.auxiliary.utils import mulaw_encode

    if not irs:
        raise ValueError("at least one impulse response is required")
    if labels is not None and len(labels) != len(irs):
        raise ValueError("labels must have one entry per impulse response")

    def draw(ax: Any) -> None:
        for i, ir in enumerate(irs):
            y = np.asarray(ir, dtype=float).ravel()
            if mulaw:
                y = mulaw_encode(y, mu)
            x_ds, y_ds = downsample_lttb(
                _time_axis(y.size, fs), y, max_points=max_points
            )
            ax.plot(
                x_ds,
                y_ds,
                linewidth=1.0,
                alpha=0.7,
                label=labels[i] if labels is not None else f"IR {i + 1}",
            )

    return _line_figure(
        draw,
        fs=fs,
        ylabel="Amplitude [mu-law]" if mulaw else "Amplitude",
        title=title,
        legend=labels is not None or len(irs) > 1,
    )


def plot_edc(
    *irs: ArrayLike,
    fs: float | None = None,
    labels: Sequence[str] | None = None,
    db: bool = True,
    normalize: bool = False,
    dynamic_range: float | None = 100.0,
    title: str | None = "Energy decay curve",
    max_points: int = 10_000,
) -> Figure:
    """Plot the energy decay curve (EDC) of one or more impulse responses.

    The EDC is the backward energy integral (:func:`pyFDN.edc`); by default it
    is shown in dB (:func:`pyFDN.sq_to_db`). Dense traces are downsampled with
    LTTB before plotting.

    Parameters
    ----------
    *irs : array-like
        One or more 1-D impulse responses, plotted as overlaid curves.
    fs : float, optional
        Sample rate in Hz. If given, the time axis is in seconds; otherwise in
        samples.
    labels : sequence of str, optional
        One legend label per impulse response.
    db : bool, optional
        Plot the decay in dB. Default True.
    normalize : bool, optional
        Normalize each curve by its initial (total) energy so it starts at
        0 dB. Default False.
    dynamic_range : float, optional
        When plotting in dB, limit the y-axis to ``dynamic_range`` dB below the
        peak across all curves (default 100, i.e. a floor at peak - 100 dB).
        This keeps the late decay from blowing out the axis once the tail
        reaches silence (``-inf`` dB). Use None for auto scaling. Ignored when
        ``db`` is False.
    title : str, optional
        Figure title.
    max_points : int, optional
        Maximum number of points per trace after downsampling. Default 10000.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from pyFDN.auxiliary.acoustics import edc
    from pyFDN.auxiliary.utils import sq_to_db

    if not irs:
        raise ValueError("at least one impulse response is required")
    if labels is not None and len(labels) != len(irs):
        raise ValueError("labels must have one entry per impulse response")

    def draw(ax: Any) -> None:
        peak = -np.inf
        for i, ir in enumerate(irs):
            y = np.asarray(ir, dtype=float).ravel()
            decay = edc(y)
            if normalize and decay.size and decay[0] > 0:
                decay = decay / decay[0]
            y_plot = sq_to_db(decay) if db else decay
            finite = y_plot[np.isfinite(y_plot)]
            if finite.size:
                peak = max(peak, float(finite.max()))
            x_ds, y_ds = downsample_lttb(
                _time_axis(y.size, fs), y_plot, max_points=max_points
            )
            ax.plot(
                x_ds,
                y_ds,
                linewidth=1.0,
                alpha=0.8,
                label=labels[i] if labels is not None else f"IR {i + 1}",
            )
        if db and dynamic_range is not None and np.isfinite(peak):
            ax.set_ylim(peak - dynamic_range, peak)

    return _line_figure(
        draw,
        fs=fs,
        ylabel="Energy [dB]" if db else "Energy",
        title=title,
        legend=labels is not None or len(irs) > 1,
    )


def plot_impulse_response_matrix(
    t: ArrayLike | None,
    ir: ArrayLike,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fig: Figure | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, np.ndarray, np.ndarray]:
    """Plot matrix of impulse responses in a subplot grid (out x in).

    Parameters
    ----------
    t : array-like, optional
        x-values (e.g. time). If None, uses 0 .. size(ir,2)-1.
    ir : array-like
        Shape (n_samples, n_out, n_in). Each subplot is ir[:, out, in].
    xlabel, ylabel, title : str, optional
        Shared axis labels and title.
    xlim, ylim : tuple, optional
        Shared axis limits. If None, computed from data.
    fig : Figure, optional
        Figure to use.
    **plot_kwargs
        Passed to ax.plot().

    Returns
    -------
    fig : Figure
    plot_axes : ndarray of Axes
        Shape (n_out, n_in).
    plot_handles : ndarray of Line2D
        Shape (n_out, n_in).
    """
    from matplotlib import pyplot as plt

    ir = np.asarray(ir)
    if ir.ndim != 3:
        raise ValueError("ir must be 3-D (n_samples,n_out, n_in)")
    n_samples, n_out, n_in = ir.shape
    if t is None:
        t = np.arange(n_samples)
    t = np.asarray(t).ravel()

    if fig is None:
        fig, plot_axes = plt.subplots(
            n_out, n_in, sharex=True, sharey=True, squeeze=False
        )
    else:
        if len(fig.axes) != n_out * n_in:
            fig, plot_axes = plt.subplots(
                n_out, n_in, sharex=True, sharey=True, squeeze=False, figure=fig
            )
        else:
            plot_axes = np.array(fig.axes).reshape(n_out, n_in)

    plot_handles = np.empty((n_out, n_in), dtype=object)
    for i_out in range(n_out):
        for i_in in range(n_in):
            ax = plot_axes[i_out, i_in]
            (h,) = ax.plot(t, ir[:, i_out, i_in], **plot_kwargs)
            plot_handles[i_out, i_in] = h
            ax.grid(True)
    # Hide inner tick labels
    for ax in plot_axes[:, 1:].ravel():
        ax.set_yticklabels([])
    for ax in plot_axes[:-1, :].ravel():
        ax.set_xticklabels([])
    if xlabel:
        fig.supxlabel(xlabel)
    if ylabel:
        fig.supylabel(ylabel)
    if title:
        fig.suptitle(title)
    if xlim is not None:
        for ax in plot_axes.ravel():
            ax.set_xlim(xlim)
    if ylim is not None:
        for ax in plot_axes.ravel():
            ax.set_ylim(ylim)
    fig.tight_layout()
    return fig, plot_axes, plot_handles


def plot_spectrogram(
    ir: ArrayLike,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str | tuple[Any, ...] = "blackman",
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    dynamic_range: float | None = 80.0,
    title: str | None = "Spectrogram",
    xlabel: str = "Time [s]",
    ylabel: str = "Frequency [Hz]",
    height: int = 500,
    colorscale: str = "Viridis",
) -> Figure:
    """Plot spectrogram of a 1-D signal as a Matplotlib image.

    Uses the same default parameters as the Poletti example: Blackman window,
    1024-point segments, 75% overlap, log y-axis, dB magnitude.

    Parameters
    ----------
    ir : array-like, 1-D
        Time-domain signal (e.g. one channel of an impulse response).
    fs : float
        Sample rate in Hz (for axis labels and frequency scale).
    nperseg : int
        Length of each segment for the STFT. Default 1024.
    noverlap : int, optional
        Number of overlapping samples. Default nperseg // 4 * 3 (75% overlap).
    window : str or tuple
        Window name or (name, param). Default "blackman".
    xlim : tuple (xmin, xmax)
        Time axis limits in seconds. Use None for auto.
    ylim : tuple (ymin, ymax)
        Frequency axis limits in Hz. Use None for auto (ymax defaults to fs/2).
    dynamic_range : float, optional
        Color (magnitude) range in dB below the peak of the displayed
        spectrogram. Default 80. Use None for Matplotlib's auto scaling.
    title : str, optional
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    height : int
        Figure height in pixels.
    colorscale : str
        Colormap name (lowercased to a Matplotlib colormap). Default "Viridis".

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure
    from scipy.signal import spectrogram

    ir = np.asarray(ir, dtype=float).ravel()
    if noverlap is None:
        noverlap = nperseg // 4 * 3
    f, t, Sxx = spectrogram(
        ir, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    Sxx_db = 10 * np.log10(Sxx + np.finfo(float).tiny)

    ymin, ymax = ylim
    if ymin is None:
        ymin = 100.0
    if ymax is None:
        ymax = fs / 2.0
    # Log y-axis needs strictly positive frequencies; trim to [ymin, ymax].
    ymin = max(ymin, 1.0)
    mask = (f >= ymin) & (f <= ymax)
    f_plot = f[mask]
    Sxx_plot = Sxx_db[mask, :]

    xmin, xmax = xlim
    if xmin is None:
        xmin = float(t[0])
    if xmax is None:
        xmax = float(t[-1])

    cmap = colorscale.lower()
    vmax = float(np.max(Sxx_plot)) if Sxx_plot.size else 0.0
    vmin = vmax - float(dynamic_range) if dynamic_range is not None else None

    dpi = 100
    fig = Figure(figsize=(8.0, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(
        t, f_plot, Sxx_plot, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def _rasterize(fig: Any) -> np.ndarray:
    """Render a Matplotlib figure to an RGBA image array."""
    import io

    from matplotlib.image import imread

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi)
    buffer.seek(0)
    return imread(buffer)


def animate(
    plot_fn: Callable[[Any], Any],
    frames: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    label_prefix: str = "",
    label_format: str = "",
    frame_ms: int = 300,
    title: str | None = None,
) -> Any:
    """Animate a sequence of frames built by any per-frame plotting function.

    ``plot_fn(frame)`` is called for each entry in ``frames`` and must return a
    Matplotlib figure (e.g. :func:`plot_matrix`, :func:`plot_impulse_response`).
    Each figure is rendered to one image, and the images are played back as a
    :class:`matplotlib.animation.FuncAnimation`. Build every frame with the
    same size and fixed axis/color limits so the frames line up.

    This composes with the existing ``plot_*`` builders instead of re-deriving
    their styling. To animate a matrix ``C`` of shape ``(rows, cols, T)`` over
    time, with fixed color limits, and show it with player controls in a
    notebook::

        import functools

        anim = pyFDN.animate(
            functools.partial(pyFDN.plot_matrix, zmin=-1, zmax=1),
            [C[:, :, k] for k in range(C.shape[2])],
            labels=t,
            label_prefix="t = ",
            label_format=".2f",
        )
        mo.Html(anim.to_jshtml())  # or IPython.display.HTML(...)

    Parameters
    ----------
    plot_fn : callable
        Maps one ``frames`` entry to a Matplotlib figure. Use
        :func:`functools.partial` or a lambda to fix extra arguments (e.g.
        color limits) so every frame is built consistently.
    frames : sequence
        One argument per frame, passed positionally to ``plot_fn``.
    labels : sequence, optional
        Label shown above each frame. Defaults to the frame index.
    label_prefix : str, optional
        Prefix shown before the current label (e.g. ``"t = "``).
    label_format : str, optional
        Format spec applied to each label, e.g. ``".2f"``. Empty uses ``str``.
    frame_ms : int, optional
        Per-frame duration in milliseconds during playback. Default 300.
    title : str, optional
        Title shown above the frame label.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Use ``.to_jshtml()`` for an interactive player or ``.save()`` to write
        a GIF or video.
    """
    from matplotlib.animation import FuncAnimation

    if len(frames) == 0:
        raise ValueError("frames must contain at least one frame")
    if labels is not None and len(labels) != len(frames):
        raise ValueError("labels must have one entry per frame")

    images = [_rasterize(plot_fn(frame)) for frame in frames]
    if labels is None:
        label_texts = [str(i) for i in range(len(images))]
    else:
        label_texts = [
            format(value, label_format) if label_format else str(value)
            for value in labels
        ]

    height_px, width_px = images[0].shape[:2]
    header_px = 30 * (1 + int(title is not None))
    fig = _new_figure(width_px, height_px + header_px)
    ax = fig.add_axes((0.0, 0.0, 1.0, height_px / (height_px + header_px)))
    ax.set_axis_off()
    image = ax.imshow(images[0], interpolation="nearest")
    total_px = height_px + header_px
    if title is not None:
        fig.text(0.5, 1.0 - 15 / total_px, title, ha="center", va="center")
    label = fig.text(
        0.5,
        1.0 - (header_px - 15) / total_px,
        label_prefix + label_texts[0],
        ha="center",
        va="center",
    )

    def update(k: int) -> tuple[Any, ...]:
        image.set_data(images[k])
        label.set_text(label_prefix + label_texts[k])
        return image, label

    return FuncAnimation(fig, update, frames=len(images), interval=frame_ms)
