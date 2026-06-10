"""Tests for plotting helpers."""

import numpy as np
import pytest

from pyFDN.auxiliary.plot import downsample_minmax, downsampled_scatter


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
