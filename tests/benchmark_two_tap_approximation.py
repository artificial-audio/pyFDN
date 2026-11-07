"""Omit the unresolved eight-root cluster in the two-tap absorption example.

Run with example dependencies from an editable checkout::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_two_tap_approximation.py

Add --all-absorption to include the first-order IIR example. This is approximate
modal analysis, not a complete decomposition or a public API. Known filter
zeros supply the group factor, removed using FFT samples. The LF kernel and
original-model residue extractor are unchanged. Assertions check numerical
accuracy, not timing. The inner radius is certified for each example.
"""

import argparse
import contextlib
import json
import sys
from time import perf_counter

import benchmark_modal_examples as bench
import numpy as np


def run_case(case):
    build, model = case["build"], case["model"]
    n = len(build.delays)
    degree = int(sum(build.delays))
    assert build.post_delay.shape == (1, 6, n)
    b0, b1, b2, a0, a1, a2 = build.post_delay[0]
    assert np.all(b0 != 0) and np.all(b2 == 0) and np.all(a2 == 0)
    zeros = -b1 / b0
    assert np.all(np.abs(zeros) < 0.9)
    # Matrix Rouche bound counts the per-line numerator-zero modes inside.
    radii = np.linspace(0.9, 0.9999, 101)
    bounds = (
        np.max(
            radii[:, None] ** build.delays
            * (np.abs(a0) * radii[:, None] + np.abs(a1))
            / (np.abs(b0) * radii[:, None] - np.abs(b1)),
            axis=1,
        )
        / np.linalg.svd(build.A, compute_uv=False)[-1]
    )
    # Avoid a large flat search interior while retaining a bound below one.
    admissible = np.flatnonzero(bounds < 0.5)
    assert admissible.size > 0
    inner, bound = float(radii[admissible[-1]]), float(bounds[admissible[-1]])
    print(
        json.dumps(
            {
                "example": case["name"],
                "inner_radius": inner,
                "boundary_norm_bound": bound,
            }
        ),
        flush=True,
    )
    timings = []
    for _ in range(3):
        start = perf_counter()
        coefficients, decomposition = bench.model_polynomial(model, degree + n)
        nfft = 1 << (degree + n).bit_length()
        w = np.exp(-2j * np.pi * np.arange(nfft // 2 + 1) / nfft)
        quotient = np.fft.irfft(
            np.fft.rfft(coefficients, nfft) / np.prod(1 - zeros[:, None] * w, axis=0),
            n=nfft,
        )
        tail = float(np.max(np.abs(quotient[degree + 1 :])))
        assert tail < 1e-10
        outer = quotient[: degree + 1]
        outer[0] = 1
        poles = bench.lindsey_fox_roots(outer[::-1], inner, 1)
        loop = bench._FDNLoopFlamo(decomposition.recursion_module)
        poles = np.array([bench._refine_pole_via_svd(complex(z), loop) for z in poles])
        poles = bench._conjugate_symmetrize(poles)
        poles, paired, _ = bench.reduce_conjugate_pairs(poles, strict=True)
        residues, direct, _, _ = bench._dss_to_res_flamo(poles, loop, decomposition)
        timings.append(perf_counter() - start)
    assert int(np.sum(1 + paired)) == degree
    print(
        json.dumps(
            {
                "example": case["name"],
                "seconds": timings,
                "median": float(np.median(timings)),
                "retained_poles": degree,
                "dropped_poles": n,
                "quotient_tail": tail,
            }
        ),
        flush=True,
    )
    length = 48000
    ir = bench.pyFDN.pr_to_impz(
        residues, poles, direct, paired, length, mode="lowMemory"
    )
    reference = bench.pyFDN.build_to_impz(build, length)
    error = ir - reference
    wet = reference.copy()
    wet[0] -= build.D
    relative = float(np.linalg.norm(error) / np.linalg.norm(wet))
    print(
        json.dumps(
            {
                "example": case["name"],
                "ir_samples": length,
                "ir_max_error": float(np.max(np.abs(error))),
                "ir_relative_wet_l2": relative,
                "wet_ir_peak": float(np.max(np.abs(wet))),
                "ir_relative_wet_l2_dB": float(20 * np.log10(relative)),
            }
        ),
        flush=True,
    )
    full_p, full_r = [], []
    for pole, residue, pair in zip(poles, residues[:, 0, 0], paired, strict=True):
        full_p.append(pole)
        full_r.append(residue)
        if pair:
            full_p.append(pole.conjugate())
            full_r.append(residue.conjugate())
    full_p, full_r = np.array(full_p), np.array(full_r)
    ws = np.exp(-1j * np.linspace(0, np.pi, 32769))
    max_error, max_wet, error_sq, ref_sq = 0.0, 0.0, 0.0, 0.0
    for start in range(0, len(ws), 256):
        w = ws[start : start + 256]
        absorption = (b0 + b1 * w[:, None]) / (a0 + a1 * w[:, None])
        forward = w[:, None] ** build.delays * absorption
        matrices = np.eye(n) - forward[:, :, None] * build.A
        hf = (build.C @ np.linalg.solve(matrices, forward[:, :, None] * build.B))[
            ..., 0, 0
        ]
        hm = w * ((1 / (1 - w[:, None] * full_p)) @ full_r)
        difference = hm - hf
        max_error = max(max_error, float(np.max(np.abs(difference))))
        max_wet = max(max_wet, float(np.max(np.abs(hf))))
        error_sq += float(np.sum(np.abs(difference) ** 2))
        ref_sq += float(np.sum(np.abs(hf) ** 2))
    print(
        json.dumps(
            {
                "example": case["name"],
                "frequency_points": len(ws),
                "frequency_max_abs_error": max_error,
                "frequency_normalized_max_error": max_error / max_wet,
                "frequency_relative_l2": float(np.sqrt(error_sq / ref_sq)),
                "frequency_wet_peak": max_wet,
            }
        ),
        flush=True,
    )
    assert np.max(np.abs(error)) < 1e-10 and relative < 1e-9
    assert max_error / max_wet < 1e-9
    mimo_build = bench.pyFDN.FDNBuild(
        A=build.A,
        B=np.eye(n),
        C=np.eye(n),
        D=np.zeros((n, n)),
        delays=build.delays,
        fs=build.fs,
        post_delay=build.post_delay,
    )
    mimo_model = bench.pyFDN.build_to_flamo(mimo_build, dtype=bench.torch.float64)
    mimo_decomposition = bench.pyFDN.flamo_decompose_for_pr(mimo_model)
    mimo_residues, mimo_direct, _, _ = bench._dss_to_res_flamo(
        poles, loop, mimo_decomposition
    )
    mimo_length = 4096
    mimo_ir = bench.pyFDN.pr_to_impz(
        mimo_residues, poles, mimo_direct, paired, mimo_length, mode="lowMemory"
    )
    mimo_reference = bench.pyFDN.build_to_impz(mimo_build, mimo_length)
    mimo_relative = float(
        np.linalg.norm(mimo_ir - mimo_reference) / np.linalg.norm(mimo_reference)
    )
    print(
        json.dumps(
            {
                "example": case["name"],
                "mimo_ir_samples": mimo_length,
                "mimo_ir_max_error": float(np.max(np.abs(mimo_ir - mimo_reference))),
                "mimo_ir_relative_l2": mimo_relative,
            }
        ),
        flush=True,
    )
    assert mimo_relative < 1e-9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-absorption",
        action="store_true",
        help="Also test the first-order IIR absorption example",
    )
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        cases = bench.cases()
    for case in cases:
        if "build" not in case or case["build"].post_delay is None:
            continue
        if args.all_absorption or case["name"] == "example_pole_boundaries.py":
            run_case(case)


if __name__ == "__main__":
    main()
