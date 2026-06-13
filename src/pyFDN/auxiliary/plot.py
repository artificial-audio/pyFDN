"""Plot utilities (matrix heatmap, system matrix layout, impulse response grid)."""

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
    *,
    block_boundaries: Sequence[int] | None = None,
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
    block_boundaries : sequence of int, optional
        Indices at which to draw dashed dividing lines on both axes, e.g. to
        separate the sub-blocks of a coupled feedback matrix. A boundary at
        index ``k`` is drawn between rows/columns ``k-1`` and ``k``.

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
    for k in block_boundaries or ():
        line = {"color": "black", "width": 1, "dash": "dash"}
        fig.add_hline(y=k - 0.5, line=line, opacity=0.5)
        fig.add_vline(x=k - 0.5, line=line, opacity=0.5)
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
) -> Any:
    """Plot several matrices as a grid of Plotly heatmaps sharing one color scale.

    Each matrix is rendered like :func:`plot_matrix` (RdBu, zero-centered,
    top-left origin, square cells). Use this to compare several matrices side
    by side, e.g. a feedback matrix against its nearest orthogonal
    approximations.

    Parameters
    ----------
    matrices : sequence of array-like
        2-D matrices to visualise, filled row by row across the grid.
    titles : sequence of str, optional
        One subplot title per matrix (supports HTML/``<br>`` for line breaks).
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
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mats = [np.asarray(m, dtype=float) for m in matrices]
    if not mats:
        raise ValueError("at least one matrix is required")
    if titles is not None and len(titles) != len(mats):
        raise ValueError("titles must have one entry per matrix")
    zmin, zmax = _shared_color_limits(zmin, zmax)

    n = len(mats)
    ncols = max(1, ncols)
    nrows = -(-n // ncols)  # ceil division

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=list(titles) if titles is not None else None,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    for idx, mat in enumerate(mats):
        row, col = divmod(idx, ncols)
        row, col = row + 1, col + 1
        fig.add_trace(
            go.Heatmap(
                z=mat,
                colorscale="RdBu",
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                showscale=(idx == n - 1),
            ),
            row=row,
            col=col,
        )
        if show_ticks:
            n_rows, n_cols = mat.shape[:2]
            fig.update_xaxes(
                tickvals=list(range(n_cols)),
                ticktext=[str(i) for i in range(n_cols)],
                row=row,
                col=col,
            )
            fig.update_yaxes(
                tickvals=list(range(n_rows)),
                ticktext=[str(i) for i in range(n_rows)],
                row=row,
                col=col,
            )
        else:
            fig.update_xaxes(showticklabels=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, row=row, col=col)
        # Origin at top-left, like plot_matrix.
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    if height is None:
        height = 300 * nrows + (60 if title else 20)
    if width is None:
        width = 300 * ncols + 80
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        height=height,
        width=width,
        template="plotly_white",
    )
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

    # Normalise to 2-D so go.Heatmap always gets a matrix.
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

    A, b, c, d = _system_matrix_blocks(A, b, c, d)
    zmin, zmax = _shared_color_limits(zmin, zmax)

    # Proportional sizes so one "cell" has the same physical size in all four subplots.
    # Layout: [A (m×n)  b (m×p);  c (q×n)  d (q×p)]
    m, n = A.shape
    b_cols = b.shape[1]
    c_rows = c.shape[0]

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


def _delay_colors(delays_arr: np.ndarray, colorscale: str = "Viridis") -> list[str]:
    """One color per delay line, mapped from the delay length via a colorscale."""
    import plotly.colors as pcolors

    span = float(delays_arr.max() - delays_arr.min()) if delays_arr.size else 0.0
    if span > 0:
        positions = (delays_arr - delays_arr.min()) / span
    else:
        positions = np.full(delays_arr.shape, 0.5)
    return pcolors.sample_colorscale(colorscale, positions.tolist())


def _db_per_sample_traces(
    sos: ArrayLike,
    delays_arr: np.ndarray,
    *,
    fs: float | None,
    nfft: int,
    colors: list[str],
    show_legend: bool = False,
) -> list[Any]:
    """Scatter traces of SOS magnitude responses in dB divided by delay length."""
    import plotly.graph_objects as go
    from scipy.signal import sosfreqz

    from pyFDN.dsp.sos_filter_bank import SOSFilterBank

    N = delays_arr.size
    sos_bank = SOSFilterBank(sos, N).sos  # (N, n_sections, 6)
    traces = []
    for i in range(N):
        w, h = sosfreqz(sos_bank[i], worN=nfft)
        mag_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).tiny)
        x = w * fs / (2.0 * np.pi) if fs is not None else w
        if fs is not None:  # drop DC for the log frequency axis
            x, mag_db = x[1:], mag_db[1:]
        traces.append(
            go.Scatter(
                x=x,
                y=mag_db / delays_arr[i],
                mode="lines",
                line={"color": colors[i], "width": 1.2},
                showlegend=show_legend,
                name=f"delay={delays_arr[i]:g}",
            )
        )
    return traces


def plot_db_per_sample(
    sos: ArrayLike,
    delays: ArrayLike,
    *,
    fs: float | None = None,
    nfft: int = 512,
    title: str | None = None,
) -> Any:
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
        :class:`pyFDN.dsp.SOSFilterBank`: ``(n_sections, 6, N)``.
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
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go

    delays_arr = np.asarray(delays, dtype=float).ravel()
    colors = _delay_colors(delays_arr)
    fig = go.Figure(
        _db_per_sample_traces(
            sos, delays_arr, fs=fs, nfft=nfft, colors=colors, show_legend=True
        )
    )
    if fs is not None:
        fig.update_xaxes(type="log", title_text="Frequency [Hz]")
    else:
        fig.update_xaxes(title_text="Frequency [rad/sample]")
    fig.update_yaxes(title_text="Magnitude [dB/sample]")
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        template="plotly_white",
        height=420,
    )
    return fig


def plot_fdn_parameter(
    delays: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    attenuation_sos: ArrayLike | None = None,
    post_eq_sos: ArrayLike | None = None,
    fs: float | None = None,
    nfft: int = 512,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Any:
    """Plot all FDN parameters in one figure.

    Extends :func:`plot_system_matrix` with the delay lengths and, optionally,
    the attenuation filters and the post EQ:

    - the system matrix blocks ``A``, ``b``, ``c``, ``d`` as heatmaps with a
      shared RdBu color scale;
    - the delays as a bar plot whose bars are aligned with the columns of the
      feedback matrix ``A`` (one bar per delay line);
    - the attenuation filters as gain-per-sample curves, as in
      :func:`plot_db_per_sample`;
    - the post EQ as plain magnitude response in dB.

    Bar and curve colors are matched per delay line and encode the delay
    length (Viridis, short = dark, long = bright).

    Parameters
    ----------
    delays : array-like
        Delay lengths in samples, shape (N,).
    A, b, c, d : array-like
        Feedback matrix, input gains, output gains, direct gains.
    attenuation_sos : array-like, optional
        Per-delay-line SOS attenuation bank, same layout as
        :class:`pyFDN.dsp.SOSFilterBank`: ``(n_sections, 6, N)``.
    post_eq_sos : array-like, optional
        Post EQ as an SOS cascade in scipy format, shape ``(n_sections, 6)``
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
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import sosfreqz

    A, b, c, d = _system_matrix_blocks(A, b, c, d)
    zmin, zmax = _shared_color_limits(zmin, zmax)
    delays_arr = np.asarray(delays, dtype=float).ravel()

    m, n = A.shape
    if delays_arr.size != n:
        raise ValueError("delays must have one entry per column of A")
    N = delays_arr.size

    # Color encodes the delay length, shared between bars and attenuation curves.
    colors = _delay_colors(delays_arr)

    # Row layout: delays | A b | c d | [attenuation] | [post EQ]
    has_attenuation = attenuation_sos is not None
    has_post_eq = post_eq_sos is not None
    matrix_px = 440.0
    row_px = [
        110.0,
        matrix_px * m / (m + c.shape[0]),
        matrix_px * c.shape[0] / (m + c.shape[0]),
    ]
    specs: list[list[dict[str, Any] | None]] = [
        [{}, None],
        [{}, {}],
        [{}, {}],
    ]
    subplot_titles = ["delays [samples]", "A", "b", "c", "d"]
    if has_attenuation:
        specs.append([{"colspan": 2}, None])
        subplot_titles.append("attenuation [dB/sample]")
        row_px.append(190.0)
    if has_post_eq:
        specs.append([{"colspan": 2}, None])
        subplot_titles.append("post EQ [dB]")
        row_px.append(190.0)
    total_px = float(sum(row_px))

    fig = make_subplots(
        rows=len(row_px),
        cols=2,
        specs=specs,
        row_heights=[h / total_px for h in row_px],
        column_widths=[n / (n + b.shape[1]), b.shape[1] / (n + b.shape[1])],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=45.0 / total_px,
    )

    # Delays as bars aligned with the columns of A (axis "x2" is A's x-axis).
    fig.add_trace(
        go.Bar(
            x=np.arange(N),
            y=delays_arr,
            marker_color=colors,
            showlegend=False,
            name="delays",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(matches="x2", showticklabels=False, row=1, col=1)

    # The A, b, c, d heatmaps are lifted from the system matrix plot.
    matrix_fig = plot_system_matrix(A, b, c, d, zmin=zmin, zmax=zmax)
    positions = [(2, 1), (2, 2), (3, 1), (3, 2)]
    for (row, col), trace in zip(positions, matrix_fig.data, strict=True):
        trace.update(showscale=False)
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(autorange="reversed", showticklabels=False, row=row, col=col)

    def _frequency_axis(w: np.ndarray) -> np.ndarray:
        if fs is not None:
            return w * fs / (2.0 * np.pi)
        return w

    def _style_frequency_xaxis(row: int) -> None:
        if fs is not None:
            fig.update_xaxes(type="log", title_text="Frequency [Hz]", row=row, col=1)
        else:
            fig.update_xaxes(title_text="Frequency [rad/sample]", row=row, col=1)

    next_row = 4
    if attenuation_sos is not None:
        for trace in _db_per_sample_traces(
            attenuation_sos, delays_arr, fs=fs, nfft=nfft, colors=colors
        ):
            fig.add_trace(trace, row=next_row, col=1)
        _style_frequency_xaxis(next_row)
        next_row += 1

    if has_post_eq:
        sos_eq = np.asarray(post_eq_sos, dtype=float)
        if sos_eq.ndim == 1:
            sos_eq = sos_eq.reshape(1, 6, 1)
        elif sos_eq.ndim == 2:
            sos_eq = sos_eq[:, :, None]
        if sos_eq.ndim != 3 or sos_eq.shape[1] != 6:
            raise ValueError(
                "post_eq_sos must have shape (n_sections, 6) or (n_sections, 6, K)"
            )
        n_out = sos_eq.shape[2]
        if n_out == 1:
            eq_colors = ["black"]
        else:
            import plotly.colors as pc

            eq_colors = pc.sample_colorscale("Plasma", np.linspace(0.0, 0.9, n_out))
        for k in range(n_out):
            w, h = sosfreqz(sos_eq[:, :, k], worN=nfft)
            mag_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).tiny)
            x = _frequency_axis(w)
            if fs is not None:
                x, mag_db = x[1:], mag_db[1:]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mag_db,
                    mode="lines",
                    line={"color": eq_colors[k], "width": 1.5},
                    showlegend=n_out > 1,
                    name=f"out {k}" if n_out > 1 else "post EQ",
                ),
                row=next_row,
                col=1,
            )
        _style_frequency_xaxis(next_row)

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        width=560,
        height=int(total_px + 120 + (40 if title else 0)),
        margin={"t": 80 if title else 50},
        template="plotly_white",
        bargap=0.2,
    )
    return fig


def plot_FDN_build(
    build: Any,
    *,
    nfft: int = 512,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
) -> Any:
    """Plot the parameters stored in an :class:`pyFDN.FDNBuild`.

    This is a convenience wrapper around :func:`plot_fdn_parameter`. A
    multichannel ``build.post_eq`` is rendered as one curve per output channel.
    """
    return plot_fdn_parameter(
        build.delays,
        build.A,
        build.B,
        build.C,
        build.D,
        attenuation_sos=build.filters,
        post_eq_sos=build.post_eq,
        fs=build.fs,
        nfft=nfft,
        zmin=zmin,
        zmax=zmax,
        title=title,
    )


def plot_impulse_response(
    *irs: ArrayLike,
    fs: float | None = None,
    labels: Sequence[str] | None = None,
    mulaw: bool = True,
    mu: float = 255.0,
    title: str | None = "Impulse response",
    max_points: int = 10_000,
) -> Any:
    """Plot one or more impulse responses over time, mu-law compressed by default.

    Mu-law companding (:func:`pyFDN.mulaw_encode`) keeps the quiet late part of
    a reverberant decay visible alongside the early reflections. Dense traces
    are downsampled with LTTB (:func:`downsampled_scatter`) before plotting.

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
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go

    from pyFDN.auxiliary.utils import mulaw_encode

    if not irs:
        raise ValueError("at least one impulse response is required")
    if labels is not None and len(labels) != len(irs):
        raise ValueError("labels must have one entry per impulse response")

    fig = go.Figure()
    for i, ir in enumerate(irs):
        y = np.asarray(ir, dtype=float).ravel()
        x = np.arange(y.size) / fs if fs is not None else np.arange(y.size)
        if mulaw:
            y = mulaw_encode(y, mu)
        fig.add_trace(
            downsampled_scatter(
                x=x,
                y=y,
                mode="lines",
                line={"width": 1.0},
                opacity=0.7,
                name=labels[i] if labels is not None else f"IR {i + 1}",
                max_points=max_points,
            )
        )
    fig.update_xaxes(title_text="Time [s]" if fs is not None else "Time [samples]")
    fig.update_yaxes(title_text="Amplitude [mu-law]" if mulaw else "Amplitude")
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        template="plotly_white",
        height=420,
        showlegend=labels is not None or len(irs) > 1,
    )
    return fig


def plot_edc(
    *irs: ArrayLike,
    fs: float | None = None,
    labels: Sequence[str] | None = None,
    db: bool = True,
    normalize: bool = False,
    dynamic_range: float | None = 100.0,
    title: str | None = "Energy decay curve",
    max_points: int = 10_000,
) -> Any:
    """Plot the energy decay curve (EDC) of one or more impulse responses.

    The EDC is the backward energy integral (:func:`pyFDN.edc`); by default it
    is shown in dB (:func:`pyFDN.sq_to_db`). Dense traces are downsampled with
    LTTB (:func:`downsampled_scatter`) before plotting.

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
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go

    from pyFDN.auxiliary.acoustics import edc
    from pyFDN.auxiliary.utils import sq_to_db

    if not irs:
        raise ValueError("at least one impulse response is required")
    if labels is not None and len(labels) != len(irs):
        raise ValueError("labels must have one entry per impulse response")

    fig = go.Figure()
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
        x = np.arange(y.size) / fs if fs is not None else np.arange(y.size)
        fig.add_trace(
            downsampled_scatter(
                x=x,
                y=y_plot,
                mode="lines",
                line={"width": 1.0},
                opacity=0.8,
                name=labels[i] if labels is not None else f"IR {i + 1}",
                max_points=max_points,
            )
        )
    fig.update_xaxes(title_text="Time [s]" if fs is not None else "Time [samples]")
    fig.update_yaxes(title_text="Energy [dB]" if db else "Energy")
    if db and dynamic_range is not None and np.isfinite(peak):
        fig.update_yaxes(range=[peak - dynamic_range, peak])
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"} if title else None,
        template="plotly_white",
        height=420,
        showlegend=labels is not None or len(irs) > 1,
    )
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
    dynamic_range: float | None = 80.0,
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
    dynamic_range : float, optional
        Color (magnitude) range in dB below the peak of the displayed
        spectrogram. Default 80. Use None for Plotly's auto scaling.
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

    heatmap_kw = {
        "x": t,
        "y": f_plot,
        "z": Sxx_plot,
        "colorscale": colorscale,
        "colorbar": {"title": "dB"},
        "zauto": dynamic_range is None,
    }
    if dynamic_range is not None:
        zmax = float(np.max(Sxx_plot))
        heatmap_kw["zmin"] = zmax - float(dynamic_range)
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


def animate(
    plot_fn: Callable[[Any], Any],
    frames: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    label_prefix: str = "",
    label_format: str = "",
    frame_ms: int = 300,
    transition_ms: int = 0,
    title: str | None = None,
) -> Any:
    """Animate a sequence of frames built by any per-frame plotting function.

    ``plot_fn(frame)`` is called for each entry in ``frames`` and must return a
    single-subplot Plotly figure (e.g. :func:`plot_matrix`,
    :func:`plot_impulse_response`). The traces of each figure become one
    animation frame; the first figure supplies the base layout (size, axes,
    color scale), to which a play/pause button and a slider are added.

    This composes with the existing ``plot_*`` builders instead of re-deriving
    their styling. To animate a matrix ``C`` of shape ``(rows, cols, T)`` over
    time, with fixed color limits::

        import functools

        fig = pyFDN.animate(
            functools.partial(pyFDN.plot_matrix, zmin=-1, zmax=1),
            [C[:, :, k] for k in range(C.shape[2])],
            labels=t,
            label_prefix="t = ",
            label_format=".2f",
        )
        fig.show()

    Parameters
    ----------
    plot_fn : callable
        Maps one ``frames`` entry to a Plotly figure. Use
        :func:`functools.partial` or a lambda to fix extra arguments (e.g.
        color limits) so every frame is built consistently.
    frames : sequence
        One argument per frame, passed positionally to ``plot_fn``.
    labels : sequence, optional
        Slider label per frame. Defaults to the frame index.
    label_prefix : str, optional
        Prefix shown before the current label (e.g. ``"t = "``).
    label_format : str, optional
        Format spec applied to each label, e.g. ``".2f"``. Empty uses ``str``.
    frame_ms : int, optional
        Per-frame duration in milliseconds during playback. Default 300.
    transition_ms : int, optional
        Tween duration between frames in milliseconds. Default 0.
    title : str, optional
        Figure title. If None, the first frame's title is kept.

    Returns
    -------
    go.Figure
        Call ``.show()`` to display.
    """
    import plotly.graph_objects as go

    if len(frames) == 0:
        raise ValueError("frames must contain at least one frame")
    if labels is not None and len(labels) != len(frames):
        raise ValueError("labels must have one entry per frame")

    figs = [plot_fn(frame) for frame in frames]
    names = [str(i) for i in range(len(figs))]
    go_frames = [
        go.Frame(data=fig_i.data, name=name)
        for name, fig_i in zip(names, figs, strict=True)
    ]

    if labels is None:
        label_texts = names
    else:
        label_texts = [
            format(value, label_format) if label_format else str(value)
            for value in labels
        ]

    play_args = {
        "frame": {"duration": frame_ms, "redraw": True},
        "fromcurrent": True,
        "transition": {"duration": transition_ms},
    }
    pause_args = {
        "frame": {"duration": 0, "redraw": False},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
    slider = {
        "active": 0,
        "currentvalue": {"prefix": label_prefix, "visible": True},
        "steps": [
            {
                "label": label_texts[i],
                "method": "animate",
                "args": [
                    [names[i]],
                    {
                        "frame": {"duration": frame_ms, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": transition_ms},
                    },
                ],
            }
            for i in range(len(figs))
        ],
    }
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {"label": "▶", "method": "animate", "args": [None, play_args]},
                {"label": "⏸", "method": "animate", "args": [[None], pause_args]},
            ],
        }
    ]

    fig = figs[0]
    fig.frames = tuple(go_frames)
    fig.update_layout(sliders=[slider], updatemenus=updatemenus)
    if title is not None:
        fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"})
    return fig
