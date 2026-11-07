"""Omit the unresolved eight-root cluster in the two-tap absorption example.

Run with example dependencies from an editable checkout::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_two_tap_approximation.py

This is approximate modal analysis, not a complete decomposition or a public
API. The known common filter zero supplies a group factor, removed using FFT
samples. The LF kernel and original-model residue extractor are unchanged.
Assertions check numerical accuracy, not timing. The cutoff is case-specific.
"""

import contextlib
import json
import sys
from time import perf_counter

import benchmark_modal_examples as bench
import numpy as np


def main():
    with contextlib.redirect_stdout(sys.stderr):
        setup = bench.setup_cell("example_pole_boundaries.py", "absorption")
    build = bench.pyFDN.FDNBuild(
        A=setup["feedback_matrix"],
        B=setup["input_gain"],
        C=setup["output_gain"],
        D=setup["direct"],
        delays=setup["delays"],
        fs=setup["fs"],
        post_delay=setup["absorption"],
    )
    model = bench.pyFDN.build_to_flamo(build, dtype=bench.torch.float64)
    n = len(build.delays)
    degree = int(sum(build.delays))
    zero = -0.3 / 0.65
    inner = 0.9
    assert np.all(build.post_delay[0, 0] == 0.65) and np.all(
        build.post_delay[0, 1] == 0.3
    )
    assert np.all(build.post_delay[0, 2] == 0) and np.all(build.post_delay[0, 3] == 1)
    assert np.all(build.post_delay[0, 4:] == 0)
    # Matrix Rouche bound: inside this circle are precisely the eight zero modes.
    bound = inner ** (int(min(build.delays)) + 1) / (
        (0.65 * inner - 0.3) * np.linalg.svd(build.A, compute_uv=False)[-1]
    )
    assert bound < 1
    print(json.dumps({"inner_radius": inner, "boundary_norm_bound": bound}), flush=True)
    timings = []
    for _ in range(3):
        start = perf_counter()
        coefficients, decomposition = bench.model_polynomial(model, degree + n)
        nfft = 1 << (degree + n).bit_length()
        w = np.exp(-2j * np.pi * np.arange(nfft // 2 + 1) / nfft)
        quotient = np.fft.irfft(
            np.fft.rfft(coefficients, nfft) / (1 - zero * w) ** n, n=nfft
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
        forward = w[:, None] ** build.delays * (0.65 + 0.3 * w[:, None])
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


if __name__ == "__main__":
    main()
