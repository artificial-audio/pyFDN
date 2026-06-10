"""Plot utilities (matrix heatmap, system matrix layout, impulse response grid)."""

from __future__ import annotations

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
    is a better default for Plotly ``mode="lines"`` than min/max bucketing,
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


def downsample_plotly_trace(
    trace: Any,
    *,
    max_points: int = 10_000,
    method: str = "lttb",
) -> Any:
    """Return a copy of a Plotly trace with downsampled ``x`` and ``y`` data.

    Traces without ``y`` data are returned unchanged. If a trace has no ``x``
    data, sample indices are generated.
    """
    y = getattr(trace, "y", None)
    if y is None:
        return trace

    x = getattr(trace, "x", None)
    if method == "lttb":
        x_ds, y_ds = downsample_lttb(x, y, max_points=max_points)
    elif method == "minmax":
        x_ds, y_ds = downsample_minmax(x, y, max_points=max_points)
    else:
        raise ValueError("method must be 'lttb' or 'minmax'")

    trace_data = trace.to_plotly_json()
    trace_data["x"] = x_ds
    trace_data["y"] = y_ds
    return trace.__class__(trace_data)


def downsampled_scatter(
    *args: Any,
    max_points: int = 10_000,
    method: str = "lttb",
    **kwargs: Any,
) -> Any:
    """Create a Plotly ``go.Scatter`` trace with downsampled line data.

    The call mirrors ``plotly.graph_objects.Scatter`` and only adds the
    ``max_points`` and ``method`` keywords:

    ``fig.add_trace(pyFDN.downsampled_scatter(x=t, y=ir, max_points=5000))``
    """
    import plotly.graph_objects as go

    return downsample_plotly_trace(
        go.Scatter(*args, **kwargs),
        max_points=max_points,
        method=method,
    )


def plot_matrix(
    A: ArrayLike,
    title: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> Any:
    """Plot a single matrix as a Plotly heatmap (RdBu, square pixels).

    Parameters
    ----------
    A : array-like
        2-D matrix to visualise.
    title : str, optional
        Figure title (supports HTML/``<sup>`` for subtitles).
    zmin, zmax : float, optional
        Color limits. Default ``(-1, 1)``.

    Returns
    -------
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go

    A = np.asarray(A, dtype=float)
    if zmin is None and zmax is None:
        zmin, zmax = -1.0, 1.0

    n = A.shape[0]
    size = max(160, 28 * n)
    fig = go.Figure(
        go.Heatmap(
            z=A,
            colorscale="RdBu",
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        width=size,
        height=size,
        margin={"t": 50 if title else 10, "b": 10, "l": 10, "r": 10},
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def plot_system_matrix(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Any:
    """Plot system matrix [A b; c d] as 2x2 Plotly heatmaps, shared RdBu color scale.

    Subplot sizes are proportional to block dimensions so that each matrix element
    (pixel) has the same physical size across all four plots.

    Parameters
    ----------
    A, b, c, d : array-like
        Feedback matrix, input gain, output gain, direct gain.
    zmin, zmax : float, optional
        Shared color limits. If both None, uses (-1, 1).
    title : str, optional
        Figure title (supports HTML/``<sup>`` for subtitles).

    Returns
    -------
    go.Figure
        Call .show() to display.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    # Normalise to 2-D so go.Heatmap always gets a matrix.
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if d.ndim == 0:
        d = d.reshape(1, 1)
    elif d.ndim == 1:
        d = d.reshape(1, -1)

    if zmin is None and zmax is None:
        zmin, zmax = -1.0, 1.0
    elif zmin is None:
        zmin = -float(np.abs(zmax)) if zmax is not None and zmax != 0 else -1.0
    elif zmax is None:
        zmax = np.abs(zmin) if zmin != 0 else 1.0

    # Proportional sizes so one "cell" has the same physical size in all four subplots.
    # Layout: [A (m×n)  b (m×p);  c (q×n)  d (q×p)]
    m, n = A.shape
    b_rows, b_cols = b.shape
    c_rows, c_cols = c.shape

    # Ensure block compatibilities for visualization.
    if b_rows != m:
        raise ValueError("b must have same number of rows as A")
    if c_cols != n:
        raise ValueError("c must have same number of columns as A")

    row_heights = [m / (m + c_rows), c_rows / (m + c_rows)]
    column_widths = [n / (n + b_cols), b_cols / (n + b_cols)]

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=row_heights,
        column_widths=column_widths,
        subplot_titles=["A", "b", "c", "d"],
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    blocks = [A, b, c, d]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), blk in zip(positions, blocks, strict=False):
        fig.add_trace(
            go.Heatmap(
                z=blk,
                colorscale="RdBu",
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                showscale=(row == 2 and col == 2),
            ),
            row=row,
            col=col,
        )
        # Origin at top-left: reverse y-axis per subplot.
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    size = 500
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        width=size,
        height=size + (40 if title else 0),
        margin={"t": 80 if title else 40},
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


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
    clim: tuple[float | None, float | None] = (None, None),
    title: str | None = "Spectrogram",
    xlabel: str = "Time [s]",
    ylabel: str = "Frequency [Hz]",
    height: int = 500,
    colorscale: str = "Viridis",
) -> Any:
    """Plot spectrogram of a 1-D signal as a Plotly heatmap.

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
    clim : tuple (zmin, zmax)
        Color (magnitude) limits in dB. Use None for auto. Default (None, None).
    title : str, optional
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    height : int
        Figure height in pixels.
    colorscale : str
        Plotly colorscale name. Default "Viridis".

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
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
    # For log y-axis, avoid f=0 and trim to [ymin, ymax] so range is applied correctly
    ymin = max(ymin, 1.0)  # log scale requires positive values
    mask = (f >= ymin) & (f <= ymax)
    f_plot = f[mask]
    Sxx_plot = Sxx_db[mask, :]

    xmin, xmax = xlim
    if xmin is None:
        xmin = float(t[0])
    if xmax is None:
        xmax = float(t[-1])
    xlim = (xmin, xmax)

    zmin, zmax = clim
    heatmap_kw = {
        "x": t,
        "y": f_plot,
        "z": Sxx_plot,
        "colorscale": colorscale,
        "colorbar": {"title": "dB"},
        "zauto": zmin is None and zmax is None,
    }
    if zmin is not None:
        heatmap_kw["zmin"] = zmin
    if zmax is not None:
        heatmap_kw["zmax"] = zmax

    fig = go.Figure(data=go.Heatmap(**heatmap_kw))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis={"range": [xmin, xmax]},
        yaxis={"type": "log", "range": [np.log10(ymin), np.log10(ymax)]},
        height=height,
        template="plotly_white",
    )
    return fig
