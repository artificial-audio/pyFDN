"""Tests for plotting helpers."""

import functools
from typing import Any

import numpy as np
import pytest

from pyFDN.auxiliary.plot import (
    animate,
    downsample_minmax,
    downsampled_scatter,
    plot_edc,
    plot_FDN_build,
    plot_matrix,
    plot_matrix_grid,
)


def test_downsample_minmax_keeps_short_inputs_unchanged():
    x = np.arange(5)
    y = np.array([0.0, 1.0, -1.0, 0.5, 0.0])

    x_ds, y_ds = downsample_minmax(x, y, max_points=10)

    np.testing.assert_array_equal(x_ds, x)
    np.testing.assert_array_equal(y_ds, y)


def test_downsample_minmax_preserves_endpoints_and_peak_budget():
    x = np.arange(1000)
    y = np.zeros(1000)
    y[123] = 10.0
    y[456] = -8.0

    x_ds, y_ds = downsample_minmax(x, y, max_points=101)

    assert len(x_ds) <= 101
    assert x_ds[0] == x[0]
    assert x_ds[-1] == x[-1]
    assert 10.0 in y_ds
    assert -8.0 in y_ds


def test_downsample_minmax_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same length"):
        downsample_minmax([0, 1], [0.0])

    with pytest.raises(ValueError, match="at least 4"):
        downsample_minmax(None, [0.0, 1.0], max_points=3)

    with pytest.raises(ValueError, match="real-valued"):
        downsample_minmax(None, np.array([1.0 + 1.0j, 0.0]))


def test_downsampled_scatter_mirrors_plotly_scatter_kwargs():
    x = np.arange(100)
    y = np.sin(x)

    trace = downsampled_scatter(
        x=x,
        y=y,
        max_points=21,
        mode="lines",
        name="ir",
        line={"width": 0.5},
    )

    assert trace.mode == "lines"
    assert trace.name == "ir"
    assert trace.line.width == 0.5
    assert len(trace.x) <= 21


def test_plot_edc_overlays_one_trace_per_ir_in_db():
    ir = np.exp(-np.arange(2000) / 200.0)

    fig = plot_edc(ir, ir, fs=48000.0, labels=["a", "b"])

    assert len(fig.data) == 2
    assert [t.name for t in fig.data] == ["a", "b"]
    # dB EDC of a decaying signal is monotonically non-increasing.
    y = np.asarray(fig.data[0].y)
    assert np.all(np.diff(y) <= 1e-9)
    assert fig.layout.yaxis.title.text == "Energy [dB]"


def test_plot_edc_normalize_starts_at_zero_db():
    ir = np.exp(-np.arange(1000) / 100.0)

    fig = plot_edc(ir, normalize=True, max_points=1000)

    assert float(np.asarray(fig.data[0].y)[0]) == pytest.approx(0.0)
    # Default dynamic range floors the axis 100 dB below the 0 dB peak.
    lo, hi = fig.layout.yaxis.range
    assert hi == pytest.approx(0.0)
    assert lo == pytest.approx(-100.0)


def test_plot_edc_default_dynamic_range_clamps_yaxis_below_peak():
    ir = np.exp(-np.arange(2000) / 200.0)

    fig = plot_edc(ir)

    lo, hi = fig.layout.yaxis.range
    peak = float(np.asarray(fig.data[0].y)[0])  # LTTB preserves the endpoints
    assert hi == pytest.approx(peak)
    assert hi - lo == pytest.approx(100.0)


def test_plot_edc_dynamic_range_none_leaves_axis_auto():
    fig = plot_edc(np.exp(-np.arange(500) / 50.0), dynamic_range=None)

    assert fig.layout.yaxis.range is None


def test_plot_edc_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="one entry per"):
        plot_edc(np.zeros(10), labels=["a", "b"])


def test_plot_FDN_build_forwards_build_parameters(monkeypatch):
    from pyFDN.generate.fdn_matrix_gallery import FDNBuild

    build = FDNBuild(
        A=np.eye(2),
        B=np.ones((2, 1)),
        C=np.ones((1, 2)),
        D=np.zeros((1, 1)),
        delays=np.array([11, 13]),
        fs=48000.0,
        filters=np.ones((1, 6, 2)),
        post_eq=np.ones((1, 6, 1)),
    )
    captured: dict[str, Any] = {}

    def fake_plot(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "figure"

    monkeypatch.setattr("pyFDN.auxiliary.plot.plot_fdn_parameter", fake_plot)

    result = plot_FDN_build(build, nfft=1024, title="FDN")

    assert result == "figure"
    forwarded = captured["args"]
    assert forwarded[0] is build.delays
    assert forwarded[1] is build.A
    assert forwarded[2] is build.B
    assert forwarded[3] is build.C
    assert forwarded[4] is build.D
    assert captured["kwargs"]["attenuation_sos"] is build.filters
    assert build.post_eq is not None
    # The full (possibly multichannel) post EQ bank is forwarded unchanged.
    assert captured["kwargs"]["post_eq_sos"] is build.post_eq
    assert captured["kwargs"]["fs"] == build.fs
    assert captured["kwargs"]["nfft"] == 1024
    assert captured["kwargs"]["title"] == "FDN"


def test_plot_FDN_build_renders_multichannel_post_eq():
    import pyFDN

    build = pyFDN.fdn_build_gallery(
        4,
        num_outputs=3,
        rt60=2.0,
        rt60_nyquist=0.5,
        post_eq_db_dc=[0.0, -3.0, -6.0],
        post_eq_db_nyquist=-6.0,
        rng=0,
    )
    fig = pyFDN.plot_FDN_build(build)

    eq_traces = [t for t in fig.data if t.name and t.name.startswith("out ")]
    assert len(eq_traces) == 3


def test_plot_matrix_block_boundaries_draws_dividing_lines():
    fig = plot_matrix(np.eye(4), block_boundaries=[2])

    # One horizontal and one vertical dashed line.
    shapes = fig.layout.shapes
    assert len(shapes) == 2
    assert all(s.line.dash == "dash" for s in shapes)


def test_plot_matrix_grid_lays_out_all_matrices():
    mats = [np.eye(3), np.ones((3, 3)), -np.eye(3)]

    fig = plot_matrix_grid(mats, titles=["a", "b", "c"], ncols=2)

    assert len(fig.data) == 3
    # Only the last heatmap carries the shared colorbar.
    assert [bool(t.showscale) for t in fig.data] == [False, False, True]


def test_plot_matrix_grid_rejects_mismatched_titles():
    with pytest.raises(ValueError, match="one entry per"):
        plot_matrix_grid([np.eye(2)], titles=["a", "b"])


def test_animate_builds_one_frame_per_input_over_plot_matrix():
    C = np.random.default_rng(0).standard_normal((4, 4, 5))
    t = np.linspace(0.0, 1.0, 5)

    fig = animate(
        functools.partial(plot_matrix, zmin=-1, zmax=1),
        [C[:, :, k] for k in range(C.shape[2])],
        labels=t,
        label_prefix="t = ",
        label_format=".2f",
    )

    assert len(fig.frames) == 5
    # First frame is the initial display; its heatmap matches C[..., 0].
    np.testing.assert_allclose(np.asarray(fig.data[0].z), C[:, :, 0])
    steps = fig.layout.sliders[0].steps
    assert [s.label for s in steps] == [f"{v:.2f}" for v in t]
    assert fig.layout.sliders[0].currentvalue.prefix == "t = "
    assert fig.layout.updatemenus  # play/pause controls present


def test_animate_works_with_arbitrary_plot_fn():
    # Any builder returning a single-subplot figure should animate.
    fig = animate(plot_edc, [np.exp(-np.arange(50) / 10.0) * a for a in (1.0, 0.5)])

    assert len(fig.frames) == 2


def test_animate_rejects_empty_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        animate(plot_matrix, [])
