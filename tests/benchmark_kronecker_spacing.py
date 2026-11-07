"""Compare positive-frequency gaps with fixed delays and seed 20261107.

Run from the checkout root: python tests/benchmark_kronecker_spacing.py
The controls use 20 SO(8) matrices and 20 independent angle perturbations
uniform in [-0.05, 0.05] radians. This is a small diagnostic, not a theorem
about modal spacing in arbitrary Kronecker FDNs.
"""

import contextlib
import json
import sys

import benchmark_modal_examples as bench
import numpy as np

from pyFDN.auxiliary.math import general_char_poly_fft

with contextlib.redirect_stdout(sys.stderr):
    cases = bench.cases()
c = next(c for c in cases if c["name"] == "example_kronecker_matrix.py/00")
rng = np.random.default_rng(20261107)


def measure(A):
    roots = np.roots(general_char_poly_fft(c["delays"], A))
    f = np.sort(np.angle(roots[roots.imag > 1e-8]) * 48000 / (2 * np.pi))
    gaps = np.diff(f)
    return {
        "minimum_gap_hz": float(gaps.min()),
        "gaps_below_original_finest_bin": int(np.sum(gaps < 48000 / 65536)),
    }


print(json.dumps({"kind": "baseline", **measure(c["A"])}), flush=True)
for kind in ["random_SO8", "perturbed_angles"]:
    for i in range(20):
        if kind == "random_SO8":
            q, r = np.linalg.qr(rng.normal(size=(8, 8)))
            A = q @ np.diag(np.sign(np.diag(r)))
            if np.linalg.det(A) < 0:
                A[:, 0] *= -1
        else:
            angles = np.pi / 4 + rng.uniform(-0.05, 0.05, size=3)
            A = bench.pyFDN.kronecker_matrix(angles)
        print(json.dumps({"kind": kind, "index": i, **measure(A)}), flush=True)
