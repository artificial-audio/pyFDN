"""Tests for plotting helpers."""

import functools
from typing import Any

import numpy as np
import pytest

from pyFDN.auxiliary.plot import (
    animate,
    downsample_minmax,
    plot_edc,
    plot_fdn_build,
    plot_fdn_parameter,
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


def test_plot_edc_overlays_one_trace_per_ir_in_db():
    ir = np.exp(-np.arange(2000) / 200.0)

    fig = plot_edc(ir, ir, fs=48000.0, labels=["a", "b"])

    ax = fig.axes[0]
    assert [line.get_label() for line in ax.get_lines()] == ["a", "b"]
    # dB EDC of a decaying signal is monotonically non-increasing.
    y = np.asarray(ax.get_lines()[0].get_ydata())
    assert np.all(np.diff(y) <= 1e-9)
    assert ax.get_ylabel() == "Energy [dB]"


def test_plot_edc_normalize_starts_at_zero_db():
    ir = np.exp(-np.arange(1000) / 100.0)

    fig = plot_edc(ir, normalize=True, max_points=1000)

    ax = fig.axes[0]
    assert float(np.asarray(ax.get_lines()[0].get_ydata())[0]) == pytest.approx(0.0)
    # Default dynamic range floors the axis 100 dB below the 0 dB peak.
    lo, hi = ax.get_ylim()
    assert hi == pytest.approx(0.0)
    assert lo == pytest.approx(-100.0)


def test_plot_edc_default_dynamic_range_clamps_yaxis_below_peak():
    ir = np.exp(-np.arange(2000) / 200.0)

    fig = plot_edc(ir)

    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    peak = float(np.asarray(ax.get_lines()[0].get_ydata())[0])  # LTTB keeps endpoints
    assert hi == pytest.approx(peak)
    assert hi - lo == pytest.approx(100.0)


def test_plot_edc_dynamic_range_none_leaves_axis_auto():
    fig = plot_edc(np.exp(-np.arange(500) / 50.0), dynamic_range=None)

    # Autoscaled: Matplotlib's default 5 % margin around the data.
    ax = fig.axes[0]
    y = np.asarray(ax.get_lines()[0].get_ydata())
    lo, hi = ax.get_ylim()
    assert lo == pytest.approx(y.min() - 0.05 * (y.max() - y.min()))


def test_plot_edc_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="one entry per"):
        plot_edc(np.zeros(10), labels=["a", "b"])


def test_plot_fdn_build_forwards_build_parameters(monkeypatch):
    from pyFDN.build import FDNBuild

    build = FDNBuild(
        A=np.eye(2),
        B=np.ones((2, 1)),
        C=np.ones((1, 2)),
        D=np.zeros((1, 1)),
        delays=np.array([11, 13]),
        fs=48000.0,
        post_delay=np.ones((1, 6, 2)),
        post_output=np.ones((1, 6, 1)),
    )
    captured: dict[str, Any] = {}

    def fake_plot(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "figure"

    monkeypatch.setattr("pyFDN.auxiliary.plot.plot_fdn_parameter", fake_plot)

    result = plot_fdn_build(build, nfft=1024, title="FDN")

    assert result == "figure"
    forwarded = captured["args"]
    assert forwarded[0] is build.delays
    assert forwarded[1] is build.A
    assert forwarded[2] is build.B
    assert forwarded[3] is build.C
    assert forwarded[4] is build.D
    assert captured["kwargs"]["post_delay_sos"] is build.post_delay
    assert build.post_output is not None
    # The full (possibly multichannel) post EQ bank is forwarded unchanged.
    assert captured["kwargs"]["post_output_sos"] is build.post_output
    assert captured["kwargs"]["fs"] == build.fs
    assert captured["kwargs"]["nfft"] == 1024
    assert captured["kwargs"]["title"] == "FDN"


def test_plot_fdn_build_renders_multichannel_post_eq():
    import pyFDN

    build = pyFDN.fdn_build_gallery(
        4,
        num_outputs=3,
        rt=2.0,
        rt_nyquist=0.5,
        output_gain_db=[0.0, -3.0, -6.0],
        output_gain_db_nyquist=-6.0,
        rng=0,
    )
    fig = pyFDN.plot_fdn_build(build)

    eq_lines = [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_label().startswith("out ")
    ]
    assert len(eq_lines) == 3


def test_plot_fdn_parameter_labels_quantities_on_y_axes():
    identity_sos = np.array([[[1.0], [0.0], [0.0], [1.0], [0.0], [0.0]]])
    per_line_sos = np.repeat(identity_sos, 2, axis=2)

    fig = plot_fdn_parameter(
        delays=[11, 13],
        A=np.eye(2),
        b=np.ones((2, 1)),
        c=np.ones((1, 2)),
        d=np.zeros((1, 1)),
        post_delay_sos=per_line_sos,
        post_matrix_sos=per_line_sos,
        post_output_sos=identity_sos[:, :, 0],
        fs=48000.0,
    )

    # one row per hook the caller supplied, labelled with the hook's own name
    ylabels = [ax.get_ylabel() for ax in fig.axes]
    assert "Delays [samples]" in ylabels
    assert "post_delay [dB/sample]" in ylabels
    assert "post_matrix [dB/sample]" in ylabels
    assert "post_output [dB]" in ylabels

    subplot_titles = [ax.get_title() for ax in fig.axes]
    assert subplot_titles[:4] == ["A", "b", "c", "d"]
    assert "post_delay [dB/sample]" not in subplot_titles
    assert "post_output [dB]" not in subplot_titles


def test_plot_fdn_parameter_omits_the_rows_for_hooks_that_are_absent():
    """Each hook is its own row, and only the ones that are there."""
    identity_sos = np.array([[[1.0], [0.0], [0.0], [1.0], [0.0], [0.0]]])
    base = {
        "delays": [11, 13],
        "A": np.eye(2),
        "b": np.ones((2, 1)),
        "c": np.ones((1, 2)),
        "d": np.zeros((1, 1)),
        "fs": 48000.0,
    }
    bare = plot_fdn_parameter(**base)
    one = plot_fdn_parameter(**base, post_matrix_sos=np.repeat(identity_sos, 2, axis=2))
    assert len(one.axes) == len(bare.axes) + 1
    assert "post_matrix [dB/sample]" in [ax.get_ylabel() for ax in one.axes]
    assert "post_delay [dB/sample]" not in [ax.get_ylabel() for ax in bare.axes]


def test_plot_matrix_block_boundaries_draws_dividing_lines():
    fig = plot_matrix(np.eye(4), block_boundaries=[2])

    # One horizontal and one vertical dashed line.
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    assert all(line.get_linestyle() == "--" for line in lines)


def test_plot_matrix_grid_lays_out_all_matrices():
    mats = [np.eye(3), np.ones((3, 3)), -np.eye(3)]

    fig = plot_matrix_grid(mats, titles=["a", "b", "c"], ncols=2)

    heatmaps = [ax for ax in fig.axes if ax.get_visible() and ax.images]
    assert [ax.get_title() for ax in heatmaps] == ["a", "b", "c"]
    # One shared colorbar; the unused fourth grid cell is hidden.
    assert len(fig.axes) == 5
    assert sum(not ax.get_visible() for ax in fig.axes) == 1


def test_plot_matrix_grid_rejects_mismatched_titles():
    with pytest.raises(ValueError, match="one entry per"):
        plot_matrix_grid([np.eye(2)], titles=["a", "b"])


def test_animate_builds_one_frame_per_input_over_plot_matrix():
    C = np.random.default_rng(0).standard_normal((4, 4, 5))
    t = np.linspace(0.0, 1.0, 5)

    anim = animate(
        functools.partial(plot_matrix, zmin=-1, zmax=1),
        [C[:, :, k] for k in range(C.shape[2])],
        labels=t,
        label_prefix="t = ",
        label_format=".2f",
    )

    frames = list(anim.new_frame_seq())
    assert frames == list(range(5))
    label = anim._fig.texts[-1]
    assert label.get_text() == "t = 0.00"
    anim._func(3)
    assert label.get_text() == f"t = {t[3]:.2f}"
    # The interactive player renders without a display.
    assert "<script" in anim.to_jshtml()


def test_animate_works_with_arbitrary_plot_fn():
    # Any builder returning a single-subplot figure should animate.
    anim = animate(plot_edc, [np.exp(-np.arange(50) / 10.0) * a for a in (1.0, 0.5)])

    assert list(anim.new_frame_seq()) == [0, 1]
    assert "<script" in anim.to_jshtml()


def test_animate_rejects_empty_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        animate(plot_matrix, [])


def test_plot_FDN_build_alias():
    import pyFDN

    assert pyFDN.plot_FDN_build is pyFDN.plot_fdn_build
