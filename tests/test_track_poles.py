"""seed_poles / track_poles against the full eig decomposition of the same DSS."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pyFDN

DELAYS = np.array([37, 53, 71, 97])


def _system(seed: int = 7) -> tuple[pyFDN.DSS, object]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((4, 4))
    A /= np.linalg.norm(A, 2)
    dss = pyFDN.DSS(DELAYS, A, np.ones((4, 1)), np.ones((1, 4)), np.zeros((1, 1)))
    return dss, pyFDN.dss_to_flamo(*dss, 48_000.0, nfft=2**10, dtype=torch.float64)


def _eig_poles(dss: pyFDN.DSS, A: np.ndarray) -> np.ndarray:
    poles = pyFDN.dss_to_pr(dss.delays, A, *dss[2:], mode="eig")[0].poles
    return poles[poles.imag >= -1e-12]


def _nearest(reference: np.ndarray, poles: np.ndarray) -> tuple[float, set[int]]:
    """Largest distance to the nearest reference pole, and which ones they are."""
    distance = np.abs(reference[:, None] - poles[None, :])
    return float(distance.min(axis=0).max()), set(distance.argmin(axis=0).tolist())


def _top(reference: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-np.abs(reference))[:k].tolist())


def test_a_gain_path_tracks_the_dominant_poles_backwards() -> None:
    dss, model = _system()
    A = torch.as_tensor(dss.A)
    gain = lambda t: 0.5 * 3.0**t  # noqa: E731  -6 dB .. +3.5 dB, uniform in dB

    def path(t):
        return {"feedback.param": gain(t) * A}

    seeds = pyFDN.seed_poles(model, 6, path=path, t=1.0)
    tracks = pyFDN.track_poles(model, path, seeds, t0=1.0, t1=0.0)

    assert tracks.t[0] == 1.0 and tracks.t[-1] == 0.0
    assert tracks.poles.shape == (len(tracks.t), 6)
    assert tracks.right.shape == (len(tracks.t), 6, 4)
    assert not np.isnan(tracks.poles).any()
    end = _eig_poles(dss, gain(1.0) * dss.A)
    error, which = _nearest(end, tracks.poles[0])
    assert error < 1e-10 and which == _top(end, 6)  # seeded: the 6 largest
    for i in range(len(tracks.t)):
        error, _ = _nearest(_eig_poles(dss, gain(tracks.t[i]) * dss.A), tracks.poles[i])
        assert error < 1e-10


def test_any_parameter_can_be_on_the_path() -> None:
    dss, model = _system()
    A1 = 0.9 * _system(seed=8)[0].A
    A0, A1t = torch.as_tensor(dss.A), torch.as_tensor(A1)

    def path(t):  # straight line between two feedback matrices
        return {"feedback.param": (1 - t) * A0 + t * A1t}

    seeds = pyFDN.seed_poles(model, 4, path=path)
    tracks = pyFDN.track_poles(model, path, seeds)
    assert not np.isnan(tracks.poles).any()
    error, _ = _nearest(_eig_poles(dss, A1), tracks.poles[-1])
    assert error < 1e-10


def test_seeding_ranks_poles_well_inside_the_unit_circle() -> None:
    dss, model = _system()
    A = torch.as_tensor(dss.A)
    seeds = pyFDN.seed_poles(model, 6, path=lambda t: {"feedback.param": 0.3 * A})
    reference = _eig_poles(dss, 0.3 * dss.A)
    error, which = _nearest(reference, seeds)
    assert error < 1e-10 and which == _top(reference, 6)


def test_bad_input_is_refused() -> None:
    _, model = _system()
    recursion = pyFDN.flamo_decompose_for_pr(model).recursion_module
    with pytest.raises(ValueError, match="order="):
        pyFDN.seed_poles(recursion, 2)
    with pytest.raises(ValueError, match="unknown parameters"):
        pyFDN.seed_poles(model, 2, path=lambda t: {"nope": t})
    with pytest.raises(ValueError, match="not poles"):
        pyFDN.track_poles(
            model, lambda t: {}, [0.5 + 0.1j], t0=0.0, t1=1.0, max_newton=1
        )
    with pytest.raises(ValueError, match="not poles"):  # z^-97 overflows: no SVD error
        pyFDN.track_poles(model, lambda t: {}, [1e-4 + 1e-4j], t0=0.0, t1=1.0)
