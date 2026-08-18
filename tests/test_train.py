"""Tests for the three-step pyFDN.train API (require torch + flamo)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flamo")

import pyFDN  # noqa: E402
from pyFDN.generate.fdn_matrix_gallery import FDNBuild  # noqa: E402
from pyFDN.train import (  # noqa: E402
    LOSSLESS_ALIAS_DECAY_DB,
    FlatMagnitude,
    FlatSpectrogram,
    MatchSpectrogram,
    Sparsity,
    Trainable,
    build_fdn,
    build_set_decay,
    model_response,
    param,
    params,
    train_fdn,
    trainable_from_build,
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


def _decayed_flatness(build, rt=1.0, nfft=2**14):
    """Flatness of |H| after homogeneous decay -- the well-posed colour measure.

    A lossless FDN has its poles on the unit circle, so the flatness of its own
    |H| is set by whichever bins land nearest a pole and swings with nfft.
    Decay does not change colouration, so measuring the decayed response on a
    fine grid is what actually tracks "colorless".
    """
    model = trainable_from_build(
        build_set_decay(build, rt), nfft=nfft, output="magnitude", device="cpu"
    )
    return _flatness(_magnitude(model, nfft))


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
    model = build_fdn(N=6, rt=None, nfft=2**11, device="cpu", rng=0)
    init_build = pyFDN.extract_build(model)
    init = _decayed_flatness(init_build)
    delays0 = init_build.delays

    log = train_fdn(model, "colorless", max_steps=500, lr=1e-2, device="cpu", rng=0)

    assert log.train_loss[-1] < log.train_loss[0]
    # the fit must flatten the FDN itself, not just the bins it was scored on
    assert _decayed_flatness(pyFDN.extract_build(model)) > init + 0.1
    out = pyFDN.extract_build(model)
    np.testing.assert_allclose(out.A.T @ out.A, np.eye(6), atol=1e-4)
    np.testing.assert_array_equal(out.delays, delays0)  # delays frozen
    assert "FlatMagnitude" in log.loss_log and "Sparsity[fB]" in log.loss_log
    assert log.steps_run == len(log.train_loss)


def test_lossless_build_defaults_to_alias_decay():
    """rt=None puts the poles on the unit circle; the gamma envelope moves them in.

    Without it |H| is unbounded and a magnitude fit just scales the gains down.
    The envelope must not leak into the extracted build.
    """
    from pyFDN.auxiliary.flamo_graph import feedback_matrix_module

    lossless = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    decaying = build_fdn(N=4, rt=2.0, nfft=2**10, device="cpu", rng=0)
    opted_out = build_fdn(
        N=4, rt=None, nfft=2**10, alias_decay_db=0.0, device="cpu", rng=0
    )

    assert float(feedback_matrix_module(lossless).gamma) < 1.0
    assert float(feedback_matrix_module(decaying).gamma) == 1.0  # decay already inside
    assert float(feedback_matrix_module(opted_out).gamma) == 1.0

    # gamma enters get_freq_response, not the parameter map -> build is unchanged
    damped, undamped = pyFDN.extract_build(lossless), pyFDN.extract_build(opted_out)
    np.testing.assert_allclose(damped.A, undamped.A, atol=1e-6)
    np.testing.assert_allclose(damped.B, undamped.B, atol=1e-6)
    np.testing.assert_allclose(damped.C, undamped.C, atol=1e-6)

    # ...but it does bound |H|, which is the whole point
    assert (
        _magnitude(
            trainable_from_build(
                undamped,
                nfft=2**10,
                output="magnitude",
                alias_decay_db=LOSSLESS_ALIAS_DECAY_DB,
                device="cpu",
            ),
            2**10,
        ).max()
        < _magnitude(
            trainable_from_build(
                undamped, nfft=2**10, output="magnitude", device="cpu"
            ),
            2**10,
        ).max()
    )


def test_colorless_without_alias_decay_warns():
    model = build_fdn(N=4, rt=None, nfft=2**10, alias_decay_db=0.0, device="cpu", rng=0)
    with pytest.warns(UserWarning, match="alias_decay_db=0"):
        train_fdn(model, "colorless", max_steps=2, rng=0, **_FAST)


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
        MatchSpectrogram(target_ir, nfft=(256, 512)),
        max_steps=20,
        rng=0,
        **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])
    out_ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(pyFDN.extract_build(fresh), nfft=nfft, device="cpu"),
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
    a Response (and build_to_impz) uses.
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
        N=N,
        rt=0.05,
        nfft=nfft,
        input_gain=rng.standard_normal((N, n_in)),
        output_gain=rng.standard_normal((n_out, N)),
        device="cpu",
        rng=0,
    )
    target = _mimo_ir(ref, nfft, n_in, n_out)
    assert target.shape == (nfft, n_out, n_in)

    fresh = build_fdn(
        N=N,
        rt=0.05,
        nfft=nfft,
        input_gain=rng.standard_normal((N, n_in)),
        output_gain=rng.standard_normal((n_out, N)),
        device="cpu",
        rng=9,
    )
    log = train_fdn(
        fresh,
        MatchSpectrogram(target, nfft=(256, 512)),
        max_steps=20,
        rng=0,
        **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])
    assert log.train_loss[-1] <= log.train_loss[0]


def test_mimo_target_wrong_shape_raises():
    model = build_fdn(
        N=4,
        rt=None,
        nfft=2**10,
        input_gain=np.ones((4, 2)),
        output_gain=np.ones((2, 4)),
        device="cpu",
        rng=0,
    )
    bad = np.zeros((128, 3, 2))  # n_out=3 != model's 2
    with pytest.raises(ValueError, match="target has 3 outputs"):
        train_fdn(model, "match_spectrogram", target=bad, max_steps=2, **_FAST)


# --- analytic decay (the exact RT path) ------------------------------------


def test_build_set_decay_realizes_rt():
    build = pyFDN.extract_build(
        build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    )
    build = build_set_decay(build, 0.3)
    assert build.filters is not None and build.filters.shape == (1, 6, 6)

    ir = np.asarray(
        pyFDN.flamo_time_response(
            pyFDN.build_to_flamo(build, nfft=2**16, device="cpu"), fs=48000
        )
    ).reshape(-1)
    rt, _ = pyFDN.estimate_rt_bands(ir, 48000.0)
    assert 0.3 * 0.7 < float(np.nanmean(rt)) < 0.3 * 1.3


# --- the response contract --------------------------------------------------


def test_response_is_the_true_ir_matrix():
    """h is the impulse response itself, shaped like build_to_impz's."""
    nfft, n_in, n_out = 2**12, 2, 3
    model = build_fdn(
        N=5,
        rt=None,
        nfft=nfft,
        input_gain=np.eye(5, n_in),
        output_gain=np.eye(n_out, 5),
        device="cpu",
        rng=4,
    )
    r = model_response(model)
    assert r.h.shape == (nfft, n_out, n_in)
    assert (r.n_samples, r.n_out, r.n_in) == (nfft, n_out, n_in)
    assert r.spectrum.shape == (nfft // 2 + 1, n_out, n_in)

    # the alias envelope is removed again, so h matches the exact (non-aliasing)
    # block simulation. Compared in RMS, not pointwise: in float32 the
    # reconstruction envelope amplifies round-off at the end of the buffer.
    exact = pyFDN.build_to_impz(pyFDN.extract_build(model), nfft)
    assert exact.shape == r.h.shape
    error = r.h.detach().numpy() - exact
    error_db = 20 * np.log10(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(exact**2)))
    assert error_db < -40


def test_alias_decay_db_is_the_response_accuracy_in_db():
    """The residual against the exact IR is alias_decay_db, by construction."""
    import torch

    nfft = 2**13
    for alias_db in (30.0, 60.0):
        model = build_fdn(
            N=6,
            rt=None,
            nfft=nfft,
            alias_decay_db=alias_db,
            dtype=torch.float64,
            device="cpu",
            rng=2,
        )
        h = model_response(model).h.detach().numpy().squeeze()
        exact = pyFDN.build_to_impz(pyFDN.extract_build(model), nfft).squeeze()
        error_db = 20 * np.log10(
            np.sqrt(np.mean((h - exact) ** 2)) / np.sqrt(np.mean(exact**2))
        )
        assert -alias_db - 2 < error_db < -alias_db + 2


def test_response_spectrum_is_computed_once():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    r = model_response(model)
    assert r.spectrum is r.spectrum  # cached, so several losses share one FFT


# --- parameter references ---------------------------------------------------


def test_params_lists_every_parameter_with_trainability():
    model = build_fdn(N=5, rt=None, nfft=2**10, device="cpu", rng=0)
    by_name = {p.name: p for p in params(model)}
    assert {"input_gain", "output_gain", "fB", "fF"} <= set(by_name)
    assert by_name["fB"].shape == (5, 5)
    assert by_name["fB"].trainable is True
    assert by_name["fF"].trainable is False  # delays are always frozen


def test_param_resolves_alias_and_returns_mapped_value():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    ref = param(model, "feedback")
    assert ref.name == "fB"
    assert ref is not None and ref.module is param(model, "fB").module

    # the mapped value is the orthogonal matrix the system uses, still in the
    # autograd graph -- not the raw (skew-symmetric) parameter.
    value = ref.value()
    assert value.requires_grad
    np.testing.assert_allclose((value.T @ value).detach().numpy(), np.eye(4), atol=1e-5)
    np.testing.assert_allclose(
        value.detach().numpy(), pyFDN.extract_build(model).A, atol=1e-5
    )


def test_param_unknown_name_lists_what_is_available():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="no parameter named 'A'.*available"):
        param(model, "A")


def test_param_accepts_a_module_directly():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    module = param(model, "feedback").module
    assert param(module).module is module


# --- composition ------------------------------------------------------------


def test_losses_compose_and_weights_reach_the_leaves():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    flat, sparse = FlatMagnitude(), Sparsity(param(model, "feedback"))
    loss = 0.5 * (flat + 0.4 * sparse)

    weights = {term.name: w for w, term in loss.terms()}
    assert weights == {"FlatMagnitude": 0.5, "Sparsity[fB]": 0.2}

    # the composed value is the weighted sum of the parts
    r = model_response(model)

    def value(objective):
        return float(objective(r).detach())

    np.testing.assert_allclose(
        value(loss), 0.5 * value(flat) + 0.2 * value(sparse), rtol=1e-5
    )


def test_sparsity_rejects_a_non_square_parameter():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="square matrix"):
        Sparsity(param(model, "input_gain"))


def test_loss_object_and_preset_name_agree():
    """train_fdn(model, "colorless") is shorthand for the preset expression."""

    def run(objective):
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=2)
        loss = objective(model) if callable(objective) else objective
        return train_fdn(model, loss, max_steps=30, rng=0, **_FAST).train_loss

    explicit = run(lambda m: FlatMagnitude() + 0.2 * Sparsity(param(m, "feedback")))
    np.testing.assert_allclose(run("colorless"), explicit, rtol=1e-6)


def test_a_loss_object_rejects_a_stray_target():
    model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="loss object holds its own"):
        train_fdn(model, FlatMagnitude(), target=np.zeros(16), max_steps=2, **_FAST)


def test_one_objective_can_hold_two_different_references():
    """Each loss owns its target, so an objective can fit more than one."""
    nfft = 2**11
    ref_a = np.asarray(
        pyFDN.flamo_time_response(
            build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=7), fs=48000
        )
    ).reshape(-1)
    ref_b = np.asarray(
        pyFDN.flamo_time_response(
            build_fdn(N=4, rt=0.08, nfft=nfft, device="cpu", rng=8), fs=48000
        )
    ).reshape(-1)

    model = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=11)
    loss = MatchSpectrogram(ref_a, nfft=(256,)) + 0.5 * pyFDN.MatchMagnitude(ref_b)
    log = train_fdn(model, loss, max_steps=10, rng=0, **_FAST)
    assert set(log.loss_log) == {"MatchSpectrogram", "MatchMagnitude"}
    assert np.isfinite(log.train_loss[-1])


# --- multi-scale flatness ---------------------------------------------------


def test_flat_spectrogram_is_gain_invariant():
    """Each scale is normalized by its own mean, so only shape is measured."""
    from pyFDN.train import Response

    model = build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    loss, r = FlatSpectrogram(), model_response(model)
    scaled = Response(h=r.h * 37.0, fs=r.fs)
    np.testing.assert_allclose(
        float(loss(r).detach()), float(loss(scaled).detach()), rtol=1e-5
    )


def test_flat_spectrogram_resolution_comes_from_its_own_windows():
    """Its windows set its resolution, so the model's nfft barely moves it.

    FlatMagnitude has no windows of its own: on a lossless FDN its value is the
    spectrum of a rectangularly truncated, non-decaying IR, which grows steeply
    with the truncation length. That is the sensitivity FlatSpectrogram removes.
    """
    build = pyFDN.extract_build(
        build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    )

    def spread(loss):
        values = [
            float(
                loss(
                    model_response(
                        trainable_from_build(
                            build, nfft=n, alias_decay_db=LOSSLESS_ALIAS_DECAY_DB
                        )
                    )
                ).detach()
            )
            for n in (2**12, 2**13, 2**14)
        ]
        return max(values) / min(values)

    assert spread(FlatSpectrogram(nfft=(256, 512))) < 2.0
    assert spread(FlatMagnitude()) > 4.0


def test_flat_spectrogram_rejects_a_window_longer_than_the_response():
    model = build_fdn(N=4, rt=None, nfft=2**9, device="cpu", rng=0)
    with pytest.raises(ValueError, match="longer than the response"):
        FlatSpectrogram(nfft=(2048,))(model_response(model))


def test_flat_spectrogram_flattens_and_densifies():
    """It reaches FlatMagnitude's flatness, and unlike it also improves density."""
    model = build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=0)
    init = _decayed_flatness(pyFDN.extract_build(model))

    log = train_fdn(
        model,
        FlatSpectrogram(nfft=(256, 512, 1024))
        + 0.2 * Sparsity(param(model, "feedback")),
        max_steps=500,
        lr=1e-2,
        patience=100,
        device="cpu",
        rng=0,
    )

    assert _decayed_flatness(pyFDN.extract_build(model)) > init + 0.1
    sparsity = log.loss_log["Sparsity[fB]"]
    assert sparsity[-1] < sparsity[0]  # the feedback matrix got denser
