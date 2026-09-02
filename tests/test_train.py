"""Tests for the three-step pyFDN.train API (require torch + flamo)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flamo")
pytest.importorskip("auraloss")

import pyFDN  # noqa: E402
from pyFDN.build import FDNBuild  # noqa: E402
from pyFDN.train import (  # noqa: E402
    LOSSLESS_ALIAS_DECAY_DB,
    FlatMagnitude,
    FlatSpectrogram,
    MatchCumulativeEnergy,
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


def _magnitude(model):
    """|H| at DFT bins as a loss sees it, summed over channels."""
    return np.asarray(model_response(model).magnitude.detach()).sum(axis=(1, 2))


def _decayed_flatness(build, rt=1.0, nfft=2**14):
    """Flatness of |H| after homogeneous decay -- the well-posed colour measure.

    A lossless FDN has its poles on the unit circle, so the flatness of its own
    |H| is set by whichever bins land nearest a pole and swings with nfft.
    Decay does not change colouration, so measuring the decayed response on a
    fine grid is what actually tracks "colorless".
    """
    model = trainable_from_build(build_set_decay(build, rt), nfft=nfft, device="cpu")
    return _flatness(_magnitude(model))


# The delay set of ``example_train_colorless_FDN``, so the peak/dip numbers the
# losses are documented with are measured on the same FDN.
_COLORLESS_DELAYS = pyFDN.sample_delay_lengths(
    8, (200, 600), distribution="uniform", coprime=True, sort=True, rng=2
)


def _decayed_db(build, rt=1.0, nfft=2**14):
    """|H| in dB relative to its own median, after decay -- see _decayed_flatness."""
    model = trainable_from_build(build_set_decay(build, rt), nfft=nfft, device="cpu")
    db = 20 * np.log10(np.maximum(_magnitude(model).ravel()[1:], 1e-12))
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
    assert b.post_delay is None  # lossless (rt=None)


def test_build_rt_sets_absorption_and_renders():
    model = build_fdn(N=6, rt=2.0, nfft=2**12, device="cpu", rng=1)
    ir = np.asarray(pyFDN.flamo_time_response(model, fs=48000)).reshape(-1)
    assert np.all(np.isfinite(ir))
    b = pyFDN.extract_build(model)
    assert b.post_delay is not None and b.post_delay.shape[1] == 6


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


def test_an_attenuation_filter_reproduces_the_designed_absorption_filters():
    """The trainable attenuation filter agrees with ``decay_to_geq``."""
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
    model = trainable_from_build(
        build, post_delay=_decay(build, rt), nfft=2**12, device="cpu"
    )
    designed = pyFDN.decay_to_geq(rt, delays, fs)
    trained = param(model, "post_delay").value().detach().numpy().astype(float)
    assert trained.shape == designed.shape

    for channel in range(len(delays)):
        _, h_designed = sosfreqz(designed[:, :, channel], worN=256, fs=fs)
        _, h_trained = sosfreqz(trained[:, :, channel], worN=256, fs=fs)
        db = 20 * np.log10(np.abs(h_designed)) - 20 * np.log10(np.abs(h_trained))
        assert np.abs(db).max() < 0.1


def test_the_decay_parameter_is_the_rt_and_trains():
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
    frozen = trainable_from_build(
        build,
        post_delay=_decay(build, rt, requires_grad=False),
        nfft=2**12,
        device="cpu",
    )
    assert param(frozen, "post_delay").trainable is False

    model = trainable_from_build(
        build,
        post_delay=_decay(build, rt),
        nfft=2**12,
        device="cpu",
    )
    ref = param(model, "post_delay")
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
        post_delay=_decay(build, rt, dtype=torch.float64),
        nfft=2**12,
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        param(float64, "post_delay").raw().add_(1.0)
    np.testing.assert_allclose(rt, 2.0)


@pytest.mark.skip(
    reason="#222: GEQ design does not yet guarantee attenuation-only feedback filters"
)
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
        post_delay=_decay(build, np.full(10, 1.0), nfft=2**13),
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
    rt = param(model, "post_delay").raw().detach().numpy()
    assert np.all(np.isfinite(rt))
    assert rt.min() < 0.0, "the run never crossed zero, so it tested nothing"
    out = pyFDN.extract_build(model)
    assert out.post_delay is not None
    assert np.all(np.isfinite(out.post_delay))
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

    loss = FlatMagnitude() + 0.2 * Sparsity(param(model, "feedback"))
    log = train_fdn(model, loss, max_steps=500, lr=1e-2, device="cpu", rng=0)

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

    # ...but it does bound what a loss sees, which is the whole point: without
    # it the rendered response is the near-singular evaluation itself, peaks and
    # all, and a magnitude fit spends its steps scaling the gains down.
    assert (
        _magnitude(
            trainable_from_build(
                undamped,
                nfft=2**10,
                alias_decay_db=LOSSLESS_ALIAS_DECAY_DB,
                device="cpu",
            )
        ).max()
        < _magnitude(trainable_from_build(undamped, nfft=2**10, device="cpu")).max()
    )


def test_colorless_without_alias_decay_warns():
    model = build_fdn(N=4, rt=None, nfft=2**10, alias_decay_db=0.0, device="cpu", rng=0)
    with pytest.warns(UserWarning, match="alias_decay_db=0"):
        train_fdn(model, FlatMagnitude(), max_steps=2, rng=0, **_FAST)


def test_train_is_reproducible():
    def run():
        model = build_fdn(N=4, rt=None, nfft=2**10, device="cpu", rng=2)
        loss = FlatMagnitude() + 0.2 * Sparsity(param(model, "feedback"))
        return train_fdn(model, loss, max_steps=50, rng=0, **_FAST)

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
        train_fdn(model, MatchSpectrogram(bad), max_steps=2, **_FAST)


# --- analytic decay (the exact RT path) ------------------------------------


def test_build_set_decay_realizes_rt():
    build = pyFDN.extract_build(
        build_fdn(N=6, rt=None, nfft=2**12, device="cpu", rng=3)
    )
    build = build_set_decay(build, 0.3)
    assert build.post_delay is not None and build.post_delay.shape == (1, 6, 6)

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


# --- the doubly-cumulated energy loss --------------------------------------


def _colored_noise(n, fs, rt_low, rt_high, rng, split=64):
    """Noise with one decay below ~fs/split and another above it."""
    t = np.arange(n) / fs
    noise = rng.standard_normal(n)
    low = np.convolve(noise, np.ones(split) / split, mode="same")
    high = noise - low
    return low * 10 ** (-3 * t / rt_low) + high * 10 ** (-3 * t / rt_high)


def test_cumulative_energy_surface_is_cumulative_in_both_directions():
    fs, n = 48000.0, 2**14
    rng = np.random.default_rng(0)
    ir = _decaying_noise(n, fs, 0.4, rng)
    loss = MatchCumulativeEnergy(ir, window=512)
    (tensor,) = loss._surfaces(_as_h(ir).double())
    surface = tensor.numpy()

    # non-increasing towards later times and towards higher frequencies
    assert np.all(np.diff(surface, axis=-1) <= 1e-9)
    assert np.all(np.diff(surface, axis=-2) <= 1e-9)
    # the corner is the total energy: everything after frame 0, above bin 0
    assert surface[..., 0, 0] == pytest.approx(
        surface.reshape(surface.shape[0], -1).max()
    )


def test_cumulative_energy_is_minimized_at_the_reference():
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(2)
    loss = MatchCumulativeEnergy(_decaying_noise(n, fs, 0.4, rng), window=1024)
    scores = {
        rt: float(loss(Response(h=_as_h(_decaying_noise(n, fs, rt, rng)), fs=fs)))
        for rt in (0.2, 0.3, 0.4, 0.6, 0.9)
    }
    assert min(scores, key=scores.get) == 0.4


def test_cumulative_energy_sees_colour_as_well_as_decay():
    """A per-band decay error the full-band energy curve alone would miss."""
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(3)
    reference = _colored_noise(n, fs, 0.6, 0.3, rng)
    loss = MatchCumulativeEnergy(reference, window=1024)

    def score(ir):
        return float(loss(Response(h=_as_h(ir), fs=fs)))

    same = score(_colored_noise(n, fs, 0.6, 0.3, np.random.default_rng(4)))
    # swapped: the same total energy decay, the wrong way round in frequency
    swapped = score(_colored_noise(n, fs, 0.3, 0.6, np.random.default_rng(4)))
    assert swapped > 3 * same


def test_cumulative_energy_level_error_is_an_error():
    """Unlike MatchEnergyDecay, this loss is not blind to overall level."""
    from pyFDN.train import Response

    fs, n = 48000.0, 2**14
    rng = np.random.default_rng(5)
    ir = _decaying_noise(n, fs, 0.4, rng)
    loss = MatchCumulativeEnergy(ir, window=512)
    exact = float(loss(Response(h=_as_h(ir), fs=fs)))
    louder = float(loss(Response(h=_as_h(2 * ir), fs=fs)))
    assert exact < 1e-6
    assert louder > 0.1


def test_stronger_compression_weights_the_quiet_end():
    """Lower ``power`` moves weight from the loud head onto the quiet tail."""
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(6)
    reference = _decaying_noise(n, fs, 0.4, rng)

    head_error = reference.copy()
    head_error[: n // 8] *= 1.05  # a small error where the energy is
    tail_error = reference.copy()
    tail_error[-(n // 8) :] *= 4.0  # a large error where there is none left

    def ratio(power):
        loss = MatchCumulativeEnergy(reference, window=1024, power=power)

        def score(ir):
            return float(loss(Response(h=_as_h(ir), fs=fs)))

        return score(tail_error) / score(head_error)

    ratios = [ratio(power) for power in (1.0, 0.5, 0.25)]
    assert ratios[0] < ratios[1] < ratios[2]


def test_cumulative_energy_rejects_a_power_outside_the_unit_interval():
    for power in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="power must be"):
            MatchCumulativeEnergy(np.zeros(16), power=power)


def test_cumulative_energy_rejects_a_window_longer_than_the_response():
    model = build_fdn(N=4, rt=0.5, nfft=2**10, device="cpu", rng=0)
    with pytest.raises(ValueError, match="longer than"):
        MatchCumulativeEnergy(np.zeros(2**10), window=2**12).check(model)


# --- the trainable output EQ -----------------------------------------------


def _post_output_db(model, fs, freqs):
    """Magnitude of the model's output filter, in dB at ``freqs``."""
    from scipy.signal import sosfreqz

    sos = param(model, "post_output").value().detach().numpy().astype(float)
    _, h = sosfreqz(sos[:, :, 0], worN=freqs, fs=fs)
    return 20 * np.log10(np.abs(h))


def _decay(build, rt, *, design="graphic_eq", nfft=2**12, **kw):
    """The build's in-loop decay as a trainable module, on a named design."""
    rt_value, rt_nyquist = (rt, None) if design == "graphic_eq" else rt
    return pyFDN.AttenuationFilter(
        rt_value,
        build.delays,
        build.fs,
        rt_nyquist=rt_nyquist,
        design=design,
        nfft=nfft,
        device="cpu",
        **kw,
    )


def _out_eq(build, gain_db, *, design="graphic_eq", nfft=2**12, **kw):
    """The build's output EQ as a trainable module, on a named design."""
    gain_value, gain_nyquist = (gain_db, None) if design == "graphic_eq" else gain_db
    return pyFDN.OutputEQ(
        gain_value,
        np.shape(build.C)[0],
        build.fs,
        gain_db_nyquist=gain_nyquist,
        design=design,
        nfft=nfft,
        device="cpu",
        **kw,
    )


def _plain_build(n=4, fs=48000.0, post_output=None):
    delays = np.array([809, 1153, 1583, 2069])[:n]
    return FDNBuild(
        A=pyFDN.fdn_build_gallery(N=n, rt=None, rng=0).A,
        B=np.ones((n, 1)) / 2,
        C=np.ones((1, n)) / 2,
        D=np.zeros((1, 1)),
        delays=delays,
        fs=fs,
        post_output=post_output,
    )


def test_trainable_post_eq_starts_flat_and_is_the_gain_in_db():
    fs = 48000.0
    build = _plain_build()
    model = trainable_from_build(
        build,
        post_output=_out_eq(build, 0.0),
        nfft=2**12,
        device="cpu",
    )
    ref = param(model, "post_output")
    assert ref.trainable is True
    # ten bands, one output channel -- and flat, because nothing said otherwise
    np.testing.assert_allclose(ref.raw().detach().numpy(), np.zeros((10, 1)), atol=1e-6)
    freqs = np.array([100.0, 500.0, 1000.0, 4000.0, 10000.0])
    np.testing.assert_allclose(_post_output_db(model, fs, freqs), 0.0, atol=0.05)

    # a 6 dB parameter is 6 dB of filter
    lifted = trainable_from_build(
        _plain_build(),
        post_output=_out_eq(_plain_build(), 6.0),
        nfft=2**12,
        device="cpu",
    )
    np.testing.assert_allclose(_post_output_db(lifted, fs, freqs), 6.0, atol=0.2)

    # and a per-band parameter reaches the band it names
    tilt = np.zeros(10)
    tilt[4] = 12.0  # the 500 Hz band (index 0 is DC, then 63, 125, 250, 500 …)
    tilted = trainable_from_build(
        _plain_build(),
        post_output=_out_eq(_plain_build(), tilt),
        nfft=2**12,
        device="cpu",
    )
    db = _post_output_db(tilted, fs, np.array([500.0, 8000.0]))
    # a single-band spike is the hardest target for a graphic EQ, so it lands a
    # little short of 12 dB -- what matters is that it lands on the right band
    assert 10.0 < db[0] < 12.5
    assert abs(db[1]) < 1.0


def test_a_baked_post_eq_is_frozen_and_training_it_is_a_module_you_build():
    sos = pyFDN.gain_to_geq(np.linspace(-6, 6, 10), fs=48000.0)
    build = _plain_build(post_output=sos[:, :, np.newaxis])

    frozen = trainable_from_build(build, nfft=2**12, device="cpu")
    assert param(frozen, "post_output").trainable is False
    # given as coefficients, it stays coefficients -- (n_sections, 6, n_out)
    assert param(frozen, "post_output").shape == build.post_output.shape

    # raw coefficients train only if the caller builds the module that says so
    thawed = trainable_from_build(
        build,
        post_output=pyFDN.sos_filter_module(
            build.post_output, 2**12, device="cpu", requires_grad=True
        ),
        nfft=2**12,
        device="cpu",
    )
    assert param(thawed, "post_output").trainable is True

    # no post EQ at all: no output filter
    assert not any(p.name == "post_output" for p in params(_plain_model()))


def _plain_model():
    return trainable_from_build(_plain_build(), nfft=2**12, device="cpu")


def test_post_eq_trains_and_survives_extraction():
    model = trainable_from_build(
        _plain_build(),
        post_delay=_decay(_plain_build(), np.full(10, 0.5), nfft=2**13),
        post_output=_out_eq(_plain_build(), 0.0, nfft=2**13),
        nfft=2**13,
        device="cpu",
    )
    gains = param(model, "post_output")
    before = gains.raw().detach().numpy().copy()
    train_fdn(
        model,
        MatchCumulativeEnergy(_impulse_target(2**13), window=512),
        max_steps=10,
        lr=1e-1,
        patience=10,
        device="cpu",
    )
    after = gains.raw().detach().numpy()
    assert not np.allclose(before, after)

    out = pyFDN.extract_build(model)
    assert out.post_output is not None
    assert out.post_output.shape == (11, 6, 1)
    assert np.all(np.isfinite(out.post_output))
    np.testing.assert_allclose(out.post_output[:, 3, :], 1.0, atol=1e-6)


def test_cumulative_frequency_direction_moves_the_weight_across_the_spectrum():
    """Which end of the spectrum the loss defends is the cumulation's direction.

    A band's error moves every row of the surface the cumulation reaches it
    from, and those rows are the *largest* -- the ones compression weights
    least. So cumulating downwards defends the top of the spectrum, upwards the
    bottom, and averaging both sits in between.
    """
    from pyFDN.train import Response

    fs, n = 48000.0, 2**15
    rng = np.random.default_rng(7)
    reference = _colored_noise(n, fs, 0.6, 0.6, rng)
    low_wrong = _colored_noise(n, fs, 0.3, 0.6, np.random.default_rng(8))
    high_wrong = _colored_noise(n, fs, 0.6, 0.3, np.random.default_rng(8))

    def bias(frequency):
        loss = MatchCumulativeEnergy(reference, window=1024, frequency=frequency)

        def score(ir):
            return float(loss(Response(h=_as_h(ir), fs=fs)))

        return score(high_wrong) / score(low_wrong)

    assert bias("descending") > bias("both") > bias("ascending")


def test_cumulative_energy_rejects_an_unknown_frequency_direction():
    with pytest.raises(ValueError, match="frequency must be"):
        MatchCumulativeEnergy(np.zeros(16), frequency="upwards")


# --- first-order shelves: the two-endpoint decay and output EQ ---------------


def test_shelf_absorption_is_the_numpy_design_and_survives_extraction():
    """A first-order shelf attenuation is the NumPy design, differentiably."""
    import torch

    fs = 48000.0
    build = _plain_build()
    rt = (2.5, 0.8)
    model = trainable_from_build(
        build,
        post_delay=_decay(build, rt, design="first_order_shelf", dtype=torch.float64),
        nfft=2**12,
        device="cpu",
        dtype=torch.float64,
    )
    ref = param(model, "post_delay")
    assert ref.trainable is True
    # the parameter is the RT pair itself, not 6 coefficients per section
    np.testing.assert_allclose(ref.raw().detach().numpy(), rt, rtol=1e-6)

    # and the filter it maps onto is the numpy design, to float64
    expected = pyFDN.decay_to_first_order_shelf(rt[0], rt[1], None, build.delays, fs)
    np.testing.assert_allclose(ref.value().detach().numpy(), expected, atol=1e-8)

    out = pyFDN.extract_build(model)
    assert out.post_delay is not None and out.post_delay.shape == (
        1,
        6,
        len(build.delays),
    )
    np.testing.assert_allclose(out.post_delay, expected, atol=1e-8)


def test_shelf_post_eq_is_the_numpy_design():
    """A first-order shelf output EQ is the NumPy design, differentiably."""
    import torch

    fs = 48000.0
    gains = (3.0, -6.0)
    model = trainable_from_build(
        _plain_build(),
        post_output=_out_eq(
            _plain_build(),
            gains,
            design="first_order_shelf",
            dtype=torch.float64,
        ),
        nfft=2**12,
        device="cpu",
        dtype=torch.float64,
    )
    ref = param(model, "post_output")
    assert ref.trainable is True
    np.testing.assert_allclose(ref.raw().detach().numpy(), [[3.0], [-6.0]], rtol=1e-6)

    expected = pyFDN.gain_to_first_order_shelf(gains[0], gains[1], None, fs)[:, :, None]
    np.testing.assert_allclose(ref.value().detach().numpy(), expected, atol=1e-10)

    # the shelf reaches its endpoints: the gain at DC and at Nyquist is the parameter
    db = _post_output_db(model, fs, np.array([1.0, 23999.0]))
    np.testing.assert_allclose(db, [3.0, -6.0], atol=0.05)


def test_shelf_crossover_moves_the_transition():
    """The fixed crossover is where the shelf sits, and it is not trained."""
    import torch

    fs = 48000.0

    def shelved(crossover):
        build = _plain_build()
        return trainable_from_build(
            build,
            post_output=_out_eq(
                build,
                (0.0, -12.0),
                design="first_order_shelf",
                crossover=crossover,
                dtype=torch.float64,
            ),
            nfft=2**12,
            device="cpu",
            dtype=torch.float64,
        )

    low, high = shelved(1000.0), shelved(8000.0)
    probe = np.array([3000.0])
    # at 3 kHz the 1 kHz shelf has already fallen and the 8 kHz one has not
    assert (
        _post_output_db(low, fs, probe)[0] < _post_output_db(high, fs, probe)[0] - 6.0
    )


def test_endpoint_targets_are_separate_and_validate_the_channel_axis():
    with pytest.raises(ValueError, match="graphic_eq uses one target array"):
        pyFDN.AttenuationFilter(
            np.ones(10),
            np.array([100.0, 150.0]),
            48000.0,
            rt_nyquist=1.0,
            design="graphic_eq",
            nfft=2**10,
        )

    with pytest.raises(ValueError, match="must have 2 columns"):
        pyFDN.AttenuationFilter(
            np.ones(3),
            np.array([100.0, 150.0]),
            48000.0,
            rt_nyquist=np.ones(3),
            design="first_order_shelf",
            nfft=2**10,
        )
    with pytest.raises(ValueError, match="must have 1 columns"):
        pyFDN.OutputEQ(
            np.ones(3),
            1,
            48000.0,
            gain_db_nyquist=np.ones(3),
            design="first_order_shelf",
            nfft=2**10,
        )


def test_shelf_decay_trains_and_stays_contractive():
    """The RT floor holds even when the fit is pushed hard at both endpoints."""
    model = trainable_from_build(
        _plain_build(),
        post_delay=_decay(
            _plain_build(),
            (1.0, 1.0),
            design="first_order_shelf",
            nfft=2**13,
        ),
        post_output=_out_eq(
            _plain_build(),
            (0.0, 0.0),
            design="first_order_shelf",
            nfft=2**13,
        ),
        nfft=2**13,
        device="cpu",
    )
    rt = param(model, "post_delay")
    eq = param(model, "post_output")
    before = rt.raw().detach().numpy().copy()

    log = train_fdn(
        model,
        MatchCumulativeEnergy(_impulse_target(2**13), window=512),
        max_steps=20,
        lr=1e-1,
        patience=20,
        device="cpu",
    )
    assert np.all(np.isfinite(log.train_loss))
    after = rt.raw().detach().numpy()
    assert not np.allclose(before, after)
    assert not np.allclose(eq.raw().detach().numpy(), 0.0)

    # the target decays in a fraction of the buffer, so the fit is pushed
    # towards zero RT -- and the mapped filter must still be contractive
    sos = rt.value().detach().numpy()
    assert np.all(np.isfinite(sos))
    assert np.max(np.abs(np.roots([1.0, sos[0, 4, 0]]))) < 1.0
    gain_dc = sos[0, 0, :] + sos[0, 1, :]
    assert np.all(np.abs(gain_dc) < 1.0)


def test_shelf_decay_pulled_below_zero_stays_at_the_floor():
    """A negative RT is an amplifying loop; the floor maps it to fast decay."""
    import torch

    delays = np.array([809.0, 1153.0, 1583.0, 2069.0])
    module = pyFDN.AttenuationFilter(
        -5.0,
        delays,
        48000.0,
        rt_nyquist=1.0,
        design="first_order_shelf",
        nfft=2**10,
        dtype=torch.float64,
    )
    sos = module.map(module.param).detach().numpy()
    assert np.all(np.isfinite(sos))
    # at DC the shelf is |b0 + b1| -- an attenuation, not a gain
    assert np.all(np.abs(sos[0, 0, :] + sos[0, 1, :]) < 1.0)
    # and it saturates: many knees below zero is the same filter as -5 s, the
    # floor's own, rather than an ever-faster decay
    deeper = pyFDN.AttenuationFilter(
        -50.0,
        delays,
        48000.0,
        rt_nyquist=1.0,
        design="first_order_shelf",
        nfft=2**10,
        dtype=torch.float64,
    )
    np.testing.assert_allclose(
        sos, deeper.map(deeper.param).detach().numpy(), rtol=1e-6
    )
    # the floor is one round trip of the longest line, i.e. MAX_ATTENUATION_DB
    # of attenuation there -- an instantaneous decay, not a silent one
    assert np.all(np.abs(sos[0, 0, :] + sos[0, 1, :]) > 1e-4)


def test_the_geq_keeps_its_ten_bands_through_both_roles():
    """Ten numbers in, ten trained numbers out -- in both hooks at once."""
    build = _plain_build()
    model = trainable_from_build(
        build,
        post_delay=_decay(build, np.full(10, 1.0)),
        post_output=_out_eq(build, 0.0),
        nfft=2**12,
        device="cpu",
    )
    assert param(model, "post_delay").raw().shape == (10,)
    assert param(model, "post_output").raw().shape == (10, 1)
    assert param(model, "post_delay").shape == (11, 6, 4)


def test_per_line_rt_gives_every_delay_line_its_own_decay():
    """A (bands, n_delays) target trains one reverberation time per line."""
    fs = 48000.0
    delays = np.array([809, 1153, 1583, 2069])
    build = FDNBuild(
        A=np.eye(4),
        B=np.ones((4, 1)),
        C=np.ones((1, 4)),
        D=np.zeros((1, 1)),
        delays=delays,
        fs=fs,
    )
    # line 0 decays fast, line 3 slowly -- a decay no shared RT can produce
    rt = np.tile(np.linspace(0.4, 3.0, 4), (10, 1))
    model = trainable_from_build(
        build, post_delay=_decay(build, rt), nfft=2**12, device="cpu"
    )

    absorption = param(model, "post_delay")
    assert absorption.raw().shape == (10, 4)
    sos = absorption.value().detach().numpy()
    assert sos.shape == (11, 6, 4)

    # each line's round-trip attenuation follows its own RT, not a shared one
    from scipy.signal import sosfreqz

    attenuation_db = [
        20 * np.log10(np.abs(sosfreqz(sos[:, :, ch], worN=64, fs=fs)[1])).mean()
        for ch in range(4)
    ]
    expected = -60.0 * delays / (np.linspace(0.4, 3.0, 4) * fs)
    np.testing.assert_allclose(attenuation_db, expected, rtol=0.05)


def test_per_line_rt_floor_is_each_line_s_own_round_trip():
    """The shared floor is the longest line's; a per-line floor is each line's."""
    import torch

    fs, nfft = 48000.0, 2**10
    delays = np.array([809.0, 4096.0])
    shared = pyFDN.AttenuationFilter(
        np.full(10, 1.0), delays, fs, nfft=nfft, dtype=torch.float64
    )
    per_line = pyFDN.AttenuationFilter(
        np.full((10, 2), 1.0), delays, fs, nfft=nfft, dtype=torch.float64
    )
    assert shared.rt_floor.ndim == 0
    np.testing.assert_allclose(float(shared.rt_floor), 4096.0 / fs)
    np.testing.assert_allclose(per_line.rt_floor.numpy(), delays / fs)

    # a negative RT saturates at the floor for both, and stays contractive
    for module, rt in ((shared, np.full(10, -5.0)), (per_line, np.full((10, 2), -5.0))):
        floored = module.map(torch.tensor(rt, dtype=torch.float64)).detach().numpy()
        assert np.all(np.isfinite(floored))


def test_one_pole_and_the_shelf_are_told_apart_by_name_not_by_length():
    """Both take two endpoints, so the EQDesign name distinguishes them."""
    build = _plain_build()
    shelf_model = trainable_from_build(
        build,
        post_delay=_decay(build, (1.5, 0.6), design="first_order_shelf"),
        nfft=2**12,
        device="cpu",
    )
    one_pole_model = trainable_from_build(
        build,
        post_delay=_decay(build, (1.5, 0.6), design="one_pole"),
        nfft=2**12,
        device="cpu",
    )
    assert param(shelf_model, "post_delay").module.design == "first_order_shelf"
    assert param(one_pole_model, "post_delay").module.design == "one_pole"

    designed = pyFDN.decay_to_one_pole(1.5, 0.6, build.delays, build.fs)
    trained = param(one_pole_model, "post_delay").value().detach().numpy()
    np.testing.assert_allclose(trained, designed, atol=1e-6)


@pytest.mark.parametrize(
    "make_loss",
    [
        lambda ref: MatchCumulativeEnergy(ref, window=512),
        lambda ref: MatchEnergyDecay(ref, window=512),
        lambda ref: pyFDN.MatchMagnitude(ref),
        lambda ref: pyFDN.MatchImpulseResponse(ref),
        lambda ref: MatchSpectrogram(ref, nfft=(256,)),
        lambda ref: pyFDN.MatchMelSpectrogram(ref, nfft=(256,)),
    ],
)
def test_a_loss_reused_on_a_new_response_rebuilds_its_reference(make_loss):
    """A loss caches its reference, but only for the response it was built for.

    Reusing one loss object across runs is the normal way a notebook cell works,
    and the second run may be at another length, dtype or device -- a GPU run
    after a CPU one used to hand a CPU reference to a CUDA step.
    """
    import torch

    from pyFDN.train import Response

    fs = 48000.0
    rng = np.random.default_rng(7)
    reference = _decaying_noise(2**13, fs, 0.4, rng)
    ir = _decaying_noise(2**13, fs, 0.3, rng)

    loss = make_loss(reference)
    float(loss(Response(h=_as_h(ir), fs=fs)))  # caches against this response

    # Longer, and in another dtype: a fresh loss is the ground truth.
    long_ir = np.concatenate([ir, np.zeros(2**12)])
    moved = Response(h=_as_h(long_ir).to(torch.float64), fs=fs)
    assert float(loss(moved)) == pytest.approx(float(make_loss(reference)(moved)))

# --- auraloss matching losses -----------------------------------------------

@pytest.mark.parametrize(
    "loss_cls",
    [
        pyFDN.MatchESR,
        pyFDN.MatchSISDR,
    ],
)

def test_auraloss_reused_on_new_response_rebuilds_reference(loss_cls):
    """Ensure _CachedTarget dynamically handles length, device, and dtype changes."""
    import torch
    from pyFDN.train import Response

    fs = 48000.0
    rng = np.random.default_rng(7)
    reference = _decaying_noise(2**12, fs, 0.4, rng)
    ir = _decaying_noise(2**12, fs, 0.3, rng)

    loss = loss_cls(reference)
    val1 = float(loss(Response(h=_as_h(ir), fs=fs)).detach())
    assert np.isfinite(val1)

    # Re-evaluate with altered length and dtype to test cache invalidation
    long_ir = np.concatenate([ir, np.zeros(2**11)])
    moved = Response(h=_as_h(long_ir).to(torch.float64), fs=fs)
    val2 = float(loss(moved).detach())

    assert np.isfinite(val2)
    assert val2 == pytest.approx(float(loss_cls(reference)(moved).detach()))

@pytest.mark.parametrize(
    "loss_cls",
    [
        pyFDN.MatchESR,
        pyFDN.MatchSISDR,
    ],
)    
def test_auraloss_trains_fdn_and_propagates_gradients(loss_cls):
    """Ensure train_fdn successfully steps and updates trainable parameters."""
    nfft = 2**11
    target = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=7)
    target_ir = np.asarray(pyFDN.flamo_time_response(target, fs=48000)).reshape(-1)
    fresh = build_fdn(N=4, rt=0.05, nfft=nfft, device="cpu", rng=11)

    fb_before = param(fresh, "feedback").raw().detach().numpy().copy()

    log = train_fdn(
        fresh,
        loss_cls(target_ir),
        max_steps=5,
        rng=0,
        **_FAST,
    )

    assert log.steps_run == 5
    assert np.isfinite(log.train_loss[-1])
    assert not np.allclose(fb_before, param(fresh, "feedback").raw().detach().numpy())


def test_auraloss_mimo_target():
    """Verify that multi-channel MIMO IRs (n_samples, n_out, n_in) train correctly."""
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
        pyFDN.MatchSISDR(target),
        max_steps=5,
        rng=0,
        **_FAST,
    )
    assert np.isfinite(log.train_loss[-1])