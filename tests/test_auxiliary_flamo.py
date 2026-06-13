"""Tests for NumPy-facing FLAMO helpers."""

import numpy as np
import torch

from pyFDN.auxiliary.flamo import flamo_time_response


class _TimeResponseModel:
    def __init__(self, response):
        self.response = response
        self.call = None

    def get_time_response(self, *, fs, identity):
        self.call = (fs, identity)
        return self.response


def test_flamo_time_response_returns_numpy_and_forwards_options():
    response = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3)
    model = _TimeResponseModel(response)

    result = flamo_time_response(model, fs=96000, identity=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4, 3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, response.numpy())
    assert model.call == (96000, True)


def test_flamo_time_response_accepts_existing_numpy_output():
    response = np.arange(6, dtype=np.float32)
    result = flamo_time_response(_TimeResponseModel(response))

    assert result is response
