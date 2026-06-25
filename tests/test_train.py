"""Tests for the three-step pyFDN.train API (require torch + flamo)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flamo")

import pyFDN  # noqa: E402
from pyFDN.generate.fdn_matrix_gallery import FDNBuild  # noqa: E402
from pyFDN.train import (  # noqa: E402
    Trainable,
    build_fdn,
    train_fdn,
    trainable_from_build,
    with_decay,
)

# Tiny / CPU / fast optimization settings.
_FAST = {"lr": 3e-3, "device": "cpu"}


def _flatness(magnitude):
    """Spectral flatness (geometric/arithmetic mean of power, DC excluded)."""
    power = np.abs(magnitude).ravel()[1:] ** 2
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def _magnitude(model, nfft, n_in=1):
    """|H| at DFT bins from a magnitude-output model, summed over channels."""
    import torch

    x = torch.zeros(1, nfft, n_in)
    x[:, 0, :] = 1.0
    with torch.no_grad():
        return np.asarray(model(x).detach())[0].sum(axis=-1)


def _leaf(model, name):
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    for node in flamo_nodes_flat(flamo_model_to_nodes(model)):
        if node["type"] == "Leaf" and node["name"] == name:
            return node["module"]
    raise AssertionError(f"no {name!r} leaf in model")


# --- build + extract -------------------------------------------------------


def test_build_fdn_default_is_so_n_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # default draw must already be in SO(N)
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    b = pyFDN.extract_build(model)
    np.testing.assert_allclose(b.A.T @ b.A, np.eye(4), atol=1e-4)
    assert np.linalg.det(b.A) > 0


def test_extract_roundtrip_with_direct_always_present():
    model = build_fdn(N=4, rt=None, nfft=2**11, device="cpu", rng=3)
    b = pyFDN.extract_build(model)
    assert isinstance(b, FDNBuild)
    np.testing.assert_allclose(b.A.T @ b.A, np.eye(4), atol=1e-4)
    assert b.B.shape == (4, 1) and b.C.shape == (1, 4)
    # direct path always exists, zero by default
    assert b.D.shape == (1, 1)
    np.testing.assert_allclose(b.D, 0.0, atol=1e-6)
    assert b.filters is None  # lossless (rt=None)


def test_build_rt_sets_absorption_and_renders():
    model = build_fdn(N=6, rt=2.0, nfft=2**12, device="cpu", rng=1)
    ir = np.asarray(pyFDN.flamo_time_response(model, fs=48000)).reshape(-1)
    assert np.all(np.isfinite(ir))
    b = pyFDN.extract_build(model)
    assert b.filters is not None and b.filters.shape[1] == 6


def test_extracted_build_renders_through_build_to_flamo():
    model = build_fdn(N=4, rt=None, nfft=2**11, device="cpu", rng=5)
    b = pyFDN.extract_build(model)
    ir = pyFDN.flamo_time_response(
        pyFDN.build_to_flamo(b, nfft=2**12, device="cpu"), fs=48000
    )
    assert np.all(np.isfinite(np.asarray(ir)))


def test_trainable_from_build_threads_requires_grad():
    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    build = pyFDN.fdn_build_gallery(N=4, rt=None, rng=0)
    model = trainable_from_build(
        build,
        trainable=Trainable(input_gain=True, output_gain=False),
        nfft=2**10,
        device="cpu",
    )
    assert feedback_matrix_module(model).param.requires_grad is True
    assert _leaf(model, "input_gain").param.requires_grad is True
    assert _leaf(model, "output_gain").param.requires_grad is False


def test_det_negative_orthogonal_warns_and_projects():
    build = pyFDN.fdn_build_gallery(N=4, rt=None, rng=1)
    if np.linalg.det(build.A) > 0:
        build.A[:, -1] *= -1.0  # force det = -1 (not in SO(N))
    with pytest.warns(UserWarning, match="SO"):
        model = trainable_from_build(build, nfft=2**10, device="cpu")
    out = pyFDN.extract_build(model)
    np.testing.assert_allclose(out.A.T @ out.A, np.eye(4), atol=1e-4)
    assert np.linalg.det(out.A) > 0


# --- train -----------------------------------------------------------------


def test_colorless_improves_and_preserves_structure():
    nfft = 2**10
    model = build_fdn(N=4, rt=None, nfft=nfft, device="cpu", rng=0)
    init_twin = trainable_from_build(
        pyFDN.extract_build(model), nfft=nfft, output="magnitude", device="cpu"
    )
    init = _flatness(_magnitude(init_twin, nfft))
    delays0 = pyFDN.extract_build(model).delays

    log = train_fdn(model, "colorless", max_steps=200, rng=0, **_FAST)

    assert log.train_loss[-1] < log.train_loss[0]
    # the model now emits |H| (output-domain swap) -> flatter than init
    assert _flatness(_magnitude(model, nfft)) > init
    out = pyFDN.extract_build(model)
    np.testing.assert_allclose(out.A.T @ out.A, np.eye(4), atol=1e-4)
    np.testing.assert_array_equal(out.delays, delays0)  # delays frozen
    assert "_ColorlessSparsity" in log.loss_log and "mse_loss" in log.loss_log
    assert log.steps_run == len(log.train_loss)


def test_train_is_reproducible():
    def run():
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=2)
        return train_fdn(model, "colorless", max_steps=50, rng=0, **_FAST)

    np.testing.assert_allclose(run().train_loss, run().train_loss, rtol=1e-6)


def test_match_spectrogram_runs_and_renders():
    nfft = 2**11
    target = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=7)
    target_ir = np.asarray(pyFDN.flamo_time_response(target, fs=48000)).reshape(-1)
    fresh = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=11)

    log = train_fdn(
        fresh,
        "match_spectrogram",
        target=target_ir,
        mss_nfft=(256, 512),
        max_steps=20,
        rng=0,
        **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])
    out_ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(
                pyFDN.extract_build(fresh), nfft=nfft, device="cpu"
            ),
            fs=48000,
        )
    ).reshape(-1)
    # the trained model extracts and renders to a finite IR
    assert out_ir.size > 0 and np.all(np.isfinite(out_ir))


def test_match_mode_requires_target():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="requires target"):
        train_fdn(model, "match_spectrogram", **_FAST)


def _mimo_ir(model, nfft, n_in, n_out):
    """Full MIMO IR matrix (n_samples, n_out, n_in) from a time-output model.

    Each input is excited on its own batch row, so model(x)[i, :, j] is the IR
    from input i to output j; transpose to the (n_samples, n_out, n_in) layout
    train_fdn expects.
    """
    import torch

    x = torch.zeros((n_in, nfft, n_in))
    for i in range(n_in):
        x[i, 0, i] = 1.0
    with torch.no_grad():
        out = np.asarray(model(x).detach())  # (n_in, nfft, n_out) = [i, t, j]
    return np.transpose(out, (1, 2, 0))  # -> (nfft, n_out, n_in) = [t, j, i]


def test_match_spectrogram_mimo_target():
    nfft, N, n_in, n_out = 2**11, 4, 2, 2
    rng = np.random.default_rng(0)
    ref = build_fdn(
        N=N, rt=0.05, nfft=nfft,
        input_gain=rng.standard_normal((N, n_in)),
        output_gain=rng.standard_normal((n_out, N)),
        device="cpu", rng=0,
    )
    target = _mimo_ir(ref, nfft, n_in, n_out)
    assert target.shape == (nfft, n_out, n_in)

    fresh = build_fdn(
        N=N, rt=0.05, nfft=nfft,
        input_gain=rng.standard_normal((N, n_in)),
        output_gain=rng.standard_normal((n_out, N)),
        device="cpu", rng=9,
    )
    log = train_fdn(
        fresh, "match_spectrogram", target=target,
        mss_nfft=(256, 512), max_steps=20, rng=0, **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])
    assert log.train_loss[-1] <= log.train_loss[0]


def test_mimo_target_wrong_shape_raises():
    model = build_fdn(
        N=4, rt=None, nfft=2**10,
        input_gain=np.ones((4, 2)), output_gain=np.ones((2, 4)),
        device="cpu", rng=0,
    )
    bad = np.zeros((128, 3, 2))  # n_out=3 != model's 2
    with pytest.raises(ValueError, match="MIMO target must have shape"):
        train_fdn(model, "match_spectrogram", target=bad, max_steps=2, **_FAST)


# --- analytic decay (the exact RT path) ------------------------------------


def test_with_decay_realizes_rt():
    build = pyFDN.extract_build(
        build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    )
    build = with_decay(build, 0.3)
    assert build.filters is not None and build.filters.shape == (1, 6, 6)

    ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(build, nfft=2**16, device="cpu"), fs=48000
        )
    ).reshape(-1)
    rt, _ = pyFDN.estimate_rt_bands(ir, 48000.0)
    assert 0.3 * 0.7 < float(np.nanmean(rt)) < 0.3 * 1.3
