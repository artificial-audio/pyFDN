"""Independent diagonal-FDN oracles and rejection tests for residual bounds."""

from importlib import import_module

import numpy as np
import pytest

import pyFDN
from pyFDN.auxiliary.modal import RankOneResidues, validate_modal_response
from pyFDN.auxiliary.poles import reduce_conjugate_pairs


def diagonal_modes(unstable=False, mixed=False):
    n = 32
    delays = np.arange(3, 35) if unstable else np.full(n, 20)
    gains = np.random.default_rng(83).uniform(0.15, 0.65, n)
    if mixed:
        gains = np.random.default_rng(83).uniform(0.25, 0.35, n)
    if unstable:
        gains[-1] = 1.3
    roots = np.concatenate(
        [
            gain ** (1 / m) * np.exp(2j * np.pi * np.arange(m) / m)
            for gain, m in zip(gains, delays, strict=True)
        ]
    )
    poles, pairs, _ = reduce_conjugate_pairs(roots, strict=True)
    line = np.argmin(abs(poles[:, None] ** delays - gains), axis=1)
    weights = 1 / (delays[line] * poles ** (delays[line] - 1))
    transform = np.eye(n)
    inverse = np.eye(n)
    if mixed:
        transform += 0.03 * np.random.default_rng(31).normal(size=(n, n)) / np.sqrt(n)
        inverse = np.linalg.inv(transform)
    factors = RankOneResidues(transform[:, line].T, inverse[line], weights)

    def probe(z):
        return transform @ np.diag(1 / (z**delays - gains)) @ inverse

    return poles, pairs, factors, transform @ np.diag(gains) @ inverse, delays, probe


def check_fixture(fixture, fast=True, tolerance=1e-8):
    p, pairs, factors, a, m, probe = fixture
    kwargs = {"static_matrix": a, "delays": m} if fast else {}
    return validate_modal_response(
        p, factors, pairs, probe, int(m.sum()), tolerance, **kwargs
    )


@pytest.mark.parametrize("perturbation", [0.0, 1e-10])
@pytest.mark.parametrize("mixed", [False, True])
def test_bound_encloses_direct_response_error_on_the_same_grid(perturbation, mixed):
    fixture = diagonal_modes(mixed=mixed)
    factors = fixture[2]
    factors.weights = factors.weights * (1 + perturbation)
    direct = check_fixture(fixture, fast=False)
    bound = check_fixture(fixture)
    assert bound["errorBound"] and not direct["errorBound"]
    assert bound["points"] == direct["points"] == 1281
    assert bound["referenceProbes"] == 64 and direct["referenceProbes"] == 1281
    assert bound["radius"] == direct["radius"]
    assert direct["relativeL2"] <= bound["relativeL2"] < 1e-8
    assert direct["normalizedPeak"] <= bound["normalizedPeak"] < 1e-8


@pytest.mark.parametrize(
    "damage",
    ["weights", "near_tolerance", "missing_pair", "missing_port", "pole_location"],
)
def test_bound_cannot_approve_damaged_modal_data(damage):
    p, pairs, factors, a, m, probe = diagonal_modes()
    if damage in ("weights", "near_tolerance"):
        scale = 1.01 if damage == "weights" else 1 + 1.1e-8
        factors = RankOneResidues(
            factors.output_vectors, factors.input_rows, scale * factors.weights
        )
    elif damage == "missing_pair":
        mask = np.ones(p.size, dtype=bool)
        mask[np.flatnonzero(pairs)[0]] = False
        p, pairs = p[mask], pairs[mask]
        factors = RankOneResidues(
            factors.output_vectors[mask],
            factors.input_rows[mask],
            factors.weights[mask],
        )
    elif damage == "pole_location":
        p = p.copy()
        p[0] *= 0.4
    else:
        inputs = factors.input_rows.copy()
        inputs[:, -1] = 0
        factors = RankOneResidues(factors.output_vectors, inputs, factors.weights)
    with pytest.raises(RuntimeError, match="All-port"):
        validate_modal_response(
            p, factors, pairs, probe, int(m.sum()), 1e-8, static_matrix=a, delays=m
        )


def test_inconclusive_bound_uses_full_direct_check(monkeypatch):
    module = import_module("pyFDN.auxiliary._modal_bound")
    monkeypatch.setattr(module, "dss_response_bound", lambda *args: None)
    result = check_fixture(diagonal_modes())
    assert not result["errorBound"] and result["referenceProbes"] == result["points"]


def test_noncontractive_bound_falls_back_without_rejecting_valid_poles():
    result = check_fixture(diagonal_modes(unstable=True))
    assert not result["errorBound"] and result["relativeL2"] < 1e-8


def test_static_refinement_visits_only_representatives(monkeypatch):
    module = import_module("pyFDN.auxiliary.modal")
    original = module.refine_dss_poles
    counts = []

    def refine(poles, *args):
        counts.append(poles.size)
        assert np.all(poles.imag >= 0)
        return original(poles, *args)

    monkeypatch.setattr(module, "refine_dss_poles", refine)
    result, info = pyFDN.dss_to_pr(
        [3, 5],
        [[0.4, 0.1], [-0.1, 0.3]],
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
        mode="lindsey_fox",
    )
    assert info["solver"] == "lindsey_fox"
    assert counts == [result.poles.size]
    assert counts[0] < 8 and np.sum(1 + result.is_conjugate_pole_pair) == 8
