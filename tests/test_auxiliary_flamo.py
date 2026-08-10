"""Tests for NumPy-facing FLAMO helpers."""

import numpy as np
import pytest
import torch

from pyFDN.auxiliary.flamo import (
    assemble_fdn_core,
    delay_module,
    flamo_time_response,
    gain_module,
    wrap_fdn_shell,
)
from pyFDN.auxiliary.flamo_graph import extract_build
from pyFDN.generate.random_orthogonal import random_orthogonal
from pyFDN.translate.dss_to_flamo import dss_to_flamo
from pyFDN.translate.dss_to_impz import dss_to_impz


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


# ============================================================================
# Shared FDN assembler (assemble_fdn_core / wrap_fdn_shell) + dss_to_flamo
#
# These guard the refactor that moved dss_to_flamo's wiring into the shared
# assembler in auxiliary/flamo.py: the render path must stay behaviourally
# identical, and the named topology must keep round-tripping through the
# extractor (so both the render and the future training builder can rely on it).
# ============================================================================


def _small_fdn():
    np.random.seed(0)
    n = 4
    a = random_orthogonal(n) * 0.9
    b = np.ones((n, 1))
    c = np.ones((1, n))
    d = np.zeros((1, 1))
    m = np.array([13.0, 17.0, 19.0, 23.0])
    return n, a, b, c, d, m


def test_dss_to_flamo_render_matches_impz():
    # The frequency-sampled FLAMO render must agree with the independent
    # time-domain dss_to_impz recursion over the early (pre-wrap) samples.
    _, a, b, c, d, m = _small_fdn()
    nfft = 2**13
    model = dss_to_flamo(a, b, c, d, m, Fs=48000, nfft=nfft, device="cpu")
    ir = np.asarray(flamo_time_response(model, fs=48000)).reshape(-1)
    ref = dss_to_impz(200, m, a, b, c, d).reshape(-1)
    np.testing.assert_allclose(ir[:120], ref[:120], atol=1e-4)


def test_dss_to_flamo_roundtrips_through_extractor():
    # Leaf names / topology survive the refactor: the extractor recovers A, B,
    # C, D and the delays from the named graph dss_to_flamo builds.
    n, a, b, c, d, m = _small_fdn()
    model = dss_to_flamo(a, b, c, d, m, Fs=48000, nfft=2**12, device="cpu")
    params = extract_build(model)
    np.testing.assert_allclose(params.A, a, atol=1e-5)
    np.testing.assert_allclose(params.B.reshape(n, 1), b, atol=1e-5)
    np.testing.assert_allclose(params.C.reshape(1, n), c, atol=1e-5)
    np.testing.assert_array_equal(params.delays, m.astype(int))


def _leaf_names(model):
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    return {
        node["name"]
        for node in flamo_nodes_flat(flamo_model_to_nodes(model))
        if node["type"] == "Leaf"
    }


def _assemble_kw(nfft):
    n, a, b, c, d, m = _small_fdn()
    return {
        "input_gain": gain_module(b, nfft, device="cpu"),
        "feedback": gain_module(a, nfft, device="cpu"),
        "delays": delay_module(m / 48000.0, nfft, Fs=48000, device="cpu"),
        "output_gain": gain_module(c, nfft, device="cpu"),
    }


def test_assemble_fdn_core_always_parallel():
    nfft = 2**11
    d = np.zeros((1, 1))
    core = assemble_fdn_core(direct=gain_module(d, nfft, device="cpu"), **_assemble_kw(nfft))
    assert type(core).__name__ == "Parallel"


def test_assemble_fdn_core_requires_direct():
    nfft = 2**11
    with pytest.raises(TypeError):
        assemble_fdn_core(**_assemble_kw(nfft))


def test_dss_to_flamo_names_delay_and_direct_leaves():
    n, a, b, c, d, m = _small_fdn()
    model = dss_to_flamo(a, b, c, d, m, Fs=48000, nfft=2**11, device="cpu")
    names = _leaf_names(model)
    assert {"input_gain", "output_gain", "delay", "direct_gain"} <= names


def test_dss_to_flamo_direct_none_defaults_to_zeros():
    n, a, b, c, d, m = _small_fdn()
    model = dss_to_flamo(a, b, c, None, m, Fs=48000, nfft=2**11, device="cpu")
    params = extract_build(model)
    assert params.D.shape == (1, 1)
    np.testing.assert_allclose(params.D, 0.0, atol=1e-6)


def test_feedback_leaf_is_mixing_matrix():
    n, a, b, c, d, m = _small_fdn()
    model = dss_to_flamo(a, b, c, d, m, Fs=48000, nfft=2**11, device="cpu")
    assert "mixing_matrix" in _leaf_names(model)
    np.testing.assert_allclose(extract_build(model).A, a, atol=1e-5)


def test_graph_branch_labels_are_descriptive():
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    n, a, b, c, d, m = _small_fdn()
    model = dss_to_flamo(a, b, c, d, m, Fs=48000, nfft=2**11, device="cpu")
    paths = "\n".join(
        node["path"] for node in flamo_nodes_flat(flamo_model_to_nodes(model))
    )
    for seg in ("/fdn/", "/direct/", "/feedforward/", "/feedback/"):
        assert seg in paths, seg


def test_wrap_fdn_shell_output_modes():
    n, a, b, c, d, m = _small_fdn()
    nfft = 2**11
    core = assemble_fdn_core(
        input_gain=gain_module(b, nfft, device="cpu"),
        feedback=gain_module(a, nfft, device="cpu"),
        delays=delay_module(m / 48000.0, nfft, Fs=48000, device="cpu"),
        output_gain=gain_module(c, nfft, device="cpu"),
        direct=gain_module(d, nfft, device="cpu"),
    )
    impulse = torch.zeros(1, nfft, 1)
    impulse[:, 0, :] = 1.0

    mag = wrap_fdn_shell(core, nfft=nfft, output="magnitude")(impulse)
    assert mag.shape == (1, nfft // 2 + 1, 1)
    assert bool((mag >= 0).all())

    time = wrap_fdn_shell(core, nfft=nfft, output="time")(impulse)
    assert time.shape == (1, nfft, 1)

    with pytest.raises(ValueError, match="output"):
        wrap_fdn_shell(core, nfft=nfft, output="phase")
