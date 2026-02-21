"""Plot utilities (matrix heatmap, system matrix layout, impulse response grid)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt


def plot_system_matrix(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    zmin: float | None = None,
    zmax: float | None = None,
):
    """Plot system matrix [A b; c d] as 2x2 Plotly heatmaps via px.imshow, shared color scale.

    Subplot sizes are proportional to block dimensions so that each matrix element
    (pixel) has the same physical size across all four plots.

    Parameters
    ----------
    A, b, c, d : array-like
        Feedback matrix, input gain, output gain, direct gain.
    zmin, zmax : float, optional
        Shared color limits. If both None, uses (-1, 1).

    Returns
    -------
    go.Figure
        Call .show() to display.
    """
    import plotly.express as px
    from plotly.subplots import make_subplots

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    if zmin is None and zmax is None:
        zmin, zmax = -1.0, 1.0
    elif zmin is None:
        zmin = -np.abs(zmax) if zmax != 0 else -1.0
    elif zmax is None:
        zmax = np.abs(zmin) if zmin != 0 else 1.0

    # Proportional sizes so one "cell" has the same physical size in all four subplots.
    # Layout: [A (m×n)  b (m×p);  c (q×n)  d (q×p)]
    m, n = A.shape
    b_rows, b_cols = b.shape if b.ndim == 2 else (b.shape[0], 1)
    c_rows, c_cols = c.shape if c.ndim == 2 else (1, c.shape[0])
    d_rows, d_cols = d.shape if d.ndim == 2 else (1, 1)

    # Ensure block compatibilities for visualization
    assert b_rows == m, "b must have same number of rows as A"
    assert c_cols == n, "c must have same number of columns as A"
    assert d_rows == c_rows and d_cols == b_cols, "d must match c rows and b cols"

    row_heights = [m / (m + c_rows), c_rows / (m + c_rows)]
    column_widths = [n / (n + b_cols), b_cols / (n + b_cols)]

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=row_heights,
        column_widths=column_widths,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    blocks = [A, b, c, d]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), blk in zip(positions, blocks):
        sub = px.imshow(
            blk,
            range_color=[zmin, zmax],
            color_continuous_midpoint=0,
            color_continuous_scale="RdBu",
            aspect="equal",
            origin="upper",
            binary_string=False,  # force Heatmap so RdBu is used (not grayscale)
        )
        trace = sub.data[0]
        trace.showscale = (row == 2 and col == 2)
        # Ensure colorscale is RdBu (in case layout overwrites)
        trace.update(colorscale="RdBu", zmin=zmin, zmax=zmax, zmid=0)
        fig.add_trace(trace, row=row, col=col)

    # Square figure so proportional row_heights/column_widths give square pixels overall
    size = 500
    fig.update_layout(
        width=size,
        height=size,
    )
    fig.update_xaxes(showticklabels=False, scaleanchor="y", scaleratio=1)
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
    fig: plt.Figure | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Plot matrix of impulse responses in a subplot grid (out x in).

    Parameters
    ----------
    t : array-like, optional
        x-values (e.g. time). If None, uses 0 .. size(ir,2)-1.
    ir : array-like
        Shape (n_out, n_in, n_samples). Each subplot is ir[out, in, :].
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
    ir = np.asarray(ir)
    if ir.ndim != 3:
        raise ValueError("ir must be 3-D (n_out, n_in, n_samples)")
    n_out, n_in, n_fir = ir.shape
    if t is None:
        t = np.arange(n_fir)
    t = np.asarray(t).ravel()

    if fig is None:
        fig, plot_axes = plt.subplots(n_out, n_in, sharex=True, sharey=True, squeeze=False)
    else:
        if len(fig.axes) != n_out * n_in:
            fig, plot_axes = plt.subplots(n_out, n_in, sharex=True, sharey=True, squeeze=False, figure=fig)
        else:
            plot_axes = np.array(fig.axes).reshape(n_out, n_in)

    plot_handles = np.empty((n_out, n_in), dtype=object)
    for i_out in range(n_out):
        for i_in in range(n_in):
            ax = plot_axes[i_out, i_in]
            (h,) = ax.plot(t, ir[i_out, i_in, :], **plot_kwargs)
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
    window: str | tuple = "blackman",
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    clim: tuple[float | None, float | None] = (None, None),
    title: str | None = "Spectrogram",
    xlabel: str = "Time [s]",
    ylabel: str = "Frequency [Hz]",
    height: int = 500,
    colorscale: str = "Viridis",
):
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
    from scipy.signal import spectrogram
    import plotly.graph_objects as go

    ir = np.asarray(ir, dtype=float).ravel()
    if noverlap is None:
        noverlap = nperseg // 4 * 3
    f, t, Sxx = spectrogram(ir, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
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
    heatmap_kw = dict(
        x=t,
        y=f_plot,
        z=Sxx_plot,
        colorscale=colorscale,
        colorbar=dict(title="dB"),
    )
    if zmin is not None:
        heatmap_kw["zmin"] = zmin
    if zmax is not None:
        heatmap_kw["zmax"] = zmax

    fig = go.Figure(data=go.Heatmap(**heatmap_kw))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(range=[xmin, xmax]),
        yaxis=dict(type="log", range=[np.log10(ymin), np.log10(ymax)]),
        height=height,
        template="plotly_white",
    )
    return fig
