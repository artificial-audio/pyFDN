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
    MatchEnergyDecay,
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


def _as_h(ir):
    """A 1-D impulse response as a Response's (n_samples, n_out, n_in) tensor."""
    import torch

    return torch.as_tensor(np.asarray(ir, dtype=np.float32))[:, None, None]


def _impulse_target(n):
    """A short decaying reference IR, enough to give a spectral loss something."""
    t = np.arange(n)
    return 0.05 * np.exp(-t / (n / 8)) * np.cos(2 * np.pi * 0.01 * t)


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


# The delay set of ``example_train_colorless_FDN``, so the peak/dip numbers the
# losses are documented with are measured on the same FDN.
_COLORLESS_DELAYS = pyFDN.sample_delay_lengths(
    8, (200, 600), distribution="uniform", coprime=True, sort=True, rng=2
)


def _decayed_db(build, rt=1.0, nfft=2**14):
    """|H| in dB relative to its own median, after decay -- see _decayed_flatness."""
    model = trainable_from_build(
        build_set_decay(build, rt), nfft=nfft, output="magnitude", device="cpu"
    )
    db = 20 * np.log10(np.maximum(_magnitude(model, nfft).ravel()[1:], 1e-12))
    return db - np.median(db)


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


def test_absorption_rt_reproduces_the_designed_absorption_filters():
    """The differentiable GEQ design agrees with pyFDN.absorption_geq."""
    from scipy.signal import sosfreqz

    fs = 48000.0
    delays = np.array([809, 1153, 1583, 2069])
    rt = np.linspace(2.5, 1.0, 10)
    build = FDNBuild(
        A=np.eye(4),
        B=np.ones((4, 1)),
        C=np.ones((1, 4)),
        D=np.zeros((1, 1)),
        delays=delays,
        fs=fs,
    )
    model = trainable_from_build(build, absorption_rt=rt, nfft=2**12, device="cpu")
    designed = pyFDN.absorption_geq(rt, delays, fs)
    trained = param(model, "absorption").value().detach().numpy().astype(float)
    assert trained.shape == designed.shape

    for channel in range(len(delays)):
        _, h_designed = sosfreqz(designed[:, :, channel], worN=256, fs=fs)
        _, h_trained = sosfreqz(trained[:, :, channel], worN=256, fs=fs)
        db = 20 * np.log10(np.abs(h_designed)) - 20 * np.log10(np.abs(h_trained))
        assert np.abs(db).max() < 0.1


def test_absorption_rt_parameter_is_the_rt_and_trains():
    fs = 48000.0
    delays = np.array([809, 1153, 1583, 2069])
    rt = np.full(10, 2.0)
    build = FDNBuild(
        A=pyFDN.fdn_build_gallery(N=4, rt=None, rng=0).A,
        B=np.ones((4, 1)) / 2,
        C=np.ones((1, 4)) / 2,
        D=np.zeros((1, 1)),
        delays=delays,
        fs=fs,
    )
    frozen = trainable_from_build(build, absorption_rt=rt, nfft=2**12, device="cpu")
    assert param(frozen, "absorption").trainable is False

    model = trainable_from_build(
        build,
        trainable=Trainable(absorption=True),
        absorption_rt=rt,
        nfft=2**12,
        device="cpu",
    )
    ref = param(model, "absorption")
    assert ref.trainable is True
    np.testing.assert_allclose(ref.raw().detach().numpy(), rt, atol=1e-5)

    train_fdn(model, MatchSpectrogram(_impulse_target(2**12)), max_steps=3, **_FAST)
    assert not np.allclose(ref.raw().detach().numpy(), rt)

    # The parameter is a copy of the caller's array, not a view of it. In
    # float64 torch.as_tensor would have shared its memory, and every optimizer
    # step would have rewritten the argument.
    import torch

    float64 = trainable_from_build(
        build,
        trainable=Trainable(absorption=True),
        absorption_rt=rt,
        nfft=2**12,
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        param(float64, "absorption").raw().add_(1.0)
    np.testing.assert_allclose(rt, 2.0)


def test_a_trained_rt_still_decays():
    """The RT parametrization is why the fit cannot leave the stable region.

    Run long enough that a band's RT is driven across zero -- the crossing the
    floor in ``DecayGEQ._floored`` exists for, and where a hard floor overflowed
    the GEQ design into NaN.
    """
    fs = 48000.0
    delays = np.array([809, 1153, 1583, 2069])
    build = FDNBuild(
        A=pyFDN.fdn_build_gallery(N=4, rt=None, rng=0).A,
        B=np.ones((4, 1)) / 2,
        C=np.ones((1, 4)) / 2,
        D=np.zeros((1, 1)),
        delays=delays,
        fs=fs,
    )
    model = trainable_from_build(
        build,
        trainable=Trainable(absorption=True),
        absorption_rt=np.full(10, 1.0),
        nfft=2**13,
        device="cpu",
    )
    # a target louder than the model everywhere: the direction that would raise
    # the loop gain past 1 if the parameter allowed it.
    train_fdn(
        model,
        MatchSpectrogram(20 * _impulse_target(2**13)),
        max_steps=150,
        lr=1e-1,
        patience=150,
        device="cpu",
    )
    rt = param(model, "absorption").raw().detach().numpy()
    assert np.all(np.isfinite(rt))
    assert rt.min() < 0.0, "the run never crossed zero, so it tested nothing"
    out = pyFDN.extract_build(model)
    assert out.filters is not None
    assert np.all(np.isfinite(out.filters))
    ir = pyFDN.build_to_impz(out, 2**15).squeeze()
    assert np.all(np.isfinite(ir))
    # a decaying system: the last eighth is quieter than the first
    head = float(np.sqrt(np.mean(ir[: 2**12] ** 2)))
    tail = float(np.sqrt(np.mean(ir[-(2**12) :] ** 2)))
    assert tail < head


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


# --- asymmetric flatness ----------------------------------------------------


def _asymmetric_reference(magnitude, peak_power):
    """AsymmetricFlatMagnitude, written out again in numpy."""
    rms = np.sqrt(np.mean(magnitude**2, axis=0, keepdims=True))
    deviation = magnitude / rms - 1.0
    peaks = np.maximum(deviation, 0.0) ** peak_power
    dips = np.minimum(deviation, 0.0) ** 2
    return float(np.mean(peaks + dips))


def test_asymmetric_flat_magnitude_matches_its_definition():
    """The peak/dip split, spelled out independently."""
    model = build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=1)
    magnitude = model_response(model).magnitude.detach().numpy()

    for peak_power in (2.0, 3.0, 4.0, 6.0):
        value = float(
            pyFDN.AsymmetricFlatMagnitude(peak_power=peak_power)(
                model_response(model)
            ).detach()
        )
        np.testing.assert_allclose(
            value, _asymmetric_reference(magnitude, peak_power), rtol=1e-4
        )


def test_asymmetric_flat_magnitude_is_zero_only_on_a_flat_response():
    """Flat stays the global minimum -- the exponent changes the route, not it."""
    import torch

    from pyFDN.train import Response

    loss = pyFDN.AsymmetricFlatMagnitude(peak_power=6.0)
    dirac = torch.zeros(64, 1, 1)
    dirac[0, 0, 0] = 1.0  # |H| = 1 at every bin
    assert float(loss(Response(h=dirac, fs=48000.0)).detach()) == pytest.approx(
        0.0, abs=1e-9
    )
    peaky = model_response(build_fdn(N=4, rt=None, nfft=2**10, rng=0))
    assert float(loss(peaky).detach()) > 0.0


def test_asymmetric_flat_magnitude_is_gain_invariant():
    """The reference is the response's own RMS, so the overall level cancels."""
    from pyFDN.train import Response

    model = build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    loss, r = pyFDN.AsymmetricFlatMagnitude(), model_response(model)
    np.testing.assert_allclose(
        float(loss(r).detach()),
        float(loss(Response(h=r.h * 37.0, fs=r.fs)).detach()),
        rtol=1e-5,
    )


def test_peak_power_raises_the_cost_of_tall_peaks_only():
    """A taller peak costs disproportionately more; the dip term never moves."""
    import torch

    from pyFDN.train import Response

    def value(peak_height, peak_power):
        # flat except for one bin, built via irfft so |H| is exactly this array.
        # Many bins keep the RMS -- and so the deviation of the peak bin --
        # essentially unchanged as the peak grows.
        magnitude = torch.ones(1025, 1, 1, dtype=torch.float64)
        magnitude[7] = peak_height
        response = Response(h=torch.fft.irfft(magnitude, dim=0), fs=48000.0)
        return float(
            pyFDN.AsymmetricFlatMagnitude(peak_power=peak_power)(response).detach()
        )

    # doubling the excursion costs 2**peak_power
    np.testing.assert_allclose(value(1.2, 2.0) / value(1.1, 2.0), 4.0, rtol=0.1)
    np.testing.assert_allclose(value(1.2, 4.0) / value(1.1, 4.0), 16.0, rtol=0.15)


def test_peak_power_lowers_the_tallest_mode():
    """What the loss exists to do, against its own peak_power=2 reference.

    A fixed-seed reproduction of one row of the loss's docstring table, at the
    settings that table was measured with. Both parts of that matter: the
    quartic fit runs out the whole step budget rather than converging inside
    `patience`, and at nfft=2**13 the advantage is absent however long it runs.
    """

    def train(peak_power):
        model = build_fdn(
            delays=_COLORLESS_DELAYS, rt=None, nfft=2**14, device="cpu", rng=2
        )
        train_fdn(
            model,
            pyFDN.AsymmetricFlatMagnitude(peak_power=peak_power),
            max_steps=2000,
            lr=1e-2,
            patience=400,
            device="cpu",
            rng=1,
        )
        return _decayed_db(pyFDN.extract_build(model))

    # measured: 18.5 dB above the median at p=2, 16.5 dB at p=4
    assert train(4.0).max() < train(2.0).max() - 1.0


def test_asymmetric_flat_magnitude_rejects_a_peak_power_below_two():
    with pytest.raises(ValueError, match="peak_power must be at least 2"):
        pyFDN.AsymmetricFlatMagnitude(peak_power=1.5)


def test_asymmetric_flat_magnitude_warns_without_alias_decay():
    model = build_fdn(N=4, rt=None, nfft=2**10, alias_decay_db=0.0, device="cpu", rng=0)
    with pytest.warns(UserWarning, match="AsymmetricFlatMagnitude fits"):
        train_fdn(model, pyFDN.AsymmetricFlatMagnitude(), max_steps=2, rng=0, **_FAST)


# --- the decay, as a loss ---------------------------------------------------


def _decaying_noise(n, fs, rt, rng):
    """White noise under an exponential envelope: a known energy decay."""
    t = np.arange(n) / fs
    return rng.standard_normal(n) * 10 ** (-3 * t / rt)


def test_match_energy_decay_reads_the_decay_and_not_the_level():
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(0)
    reference = _decaying_noise(n, fs, 0.4, rng)
    loss = MatchEnergyDecay(reference, window=2048)

    def score(ir):
        return float(loss(Response(h=_as_h(ir), fs=fs)))

    same_decay = _decaying_noise(n, fs, 0.4, np.random.default_rng(1))
    assert score(same_decay) < 3.0
    # ten times louder, same decay: the curves are normalized, so nothing moves
    assert score(10 * same_decay) == pytest.approx(score(same_decay), rel=1e-4)
    # half the decay time: a large error
    assert score(_decaying_noise(n, fs, 0.2, rng)) > 3 * score(same_decay)


def test_match_energy_decay_is_minimized_at_the_reference_decay():
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(2)
    loss = MatchEnergyDecay(_decaying_noise(n, fs, 0.4, rng), window=2048)
    scores = {
        rt: float(loss(Response(h=_as_h(_decaying_noise(n, fs, rt, rng)), fs=fs)))
        for rt in (0.2, 0.3, 0.4, 0.6, 0.9)
    }
    assert min(scores, key=scores.get) == 0.4


def test_match_energy_decay_rejects_a_window_longer_than_the_response():
    model = build_fdn(N=4, rt=0.5, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="longer than"):
        MatchEnergyDecay(np.zeros(2**10), window=2**12).check(model)
