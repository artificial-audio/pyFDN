"""Assess partial LF results in every previously failing Kronecker case.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 from an editable checkout::

    python tests/benchmark_problematic_cases.py

This diagnostic inspects the LF kernel's final candidates after its expected
verification exception. It is not a production fallback: it checks whether
omitting unresolved modes would pass response tests. No roots are supplied
by the reference solver. Lossless frequency responses are checked outside
the unit circle (exponential window), since unit-circle poles preclude a
finite uniform frequency-response error there.
"""

import contextlib
import json
import sys
from pathlib import Path

import benchmark_modal_examples as bench
import numpy as np
from scipy.optimize import linear_sum_assignment

from pyFDN.auxiliary.math import general_char_poly_fft
from pyFDN.translate.dss_to_pr import _dss_to_res


def partial_candidates(coefficients):
    captured = {}
    previous_trace = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is bench.lindsey_fox_roots.__code__:
            frame.f_trace_lines = False
            if event == "exception" and isinstance(arg[1], RuntimeError):
                captured.update(frame.f_locals)
            return trace
        return None

    sys.settrace(trace)
    try:
        return bench.lindsey_fox_roots(coefficients[::-1], 1.0, 1.0)
    except RuntimeError:
        return captured["roots"].copy()
    finally:
        sys.settrace(previous_trace)


def run_case(case):
    p = general_char_poly_fft(case["delays"], case["A"])
    roots = partial_candidates(p)
    reference = np.roots(p)
    row, col = linear_sum_assignment(np.abs(reference[:, None] - roots))
    assert np.max(np.abs(reference[row] - roots[col])) < 1e-7
    missing = np.delete(reference, row)
    roots = bench._conjugate_symmetrize(roots)
    poles, pairs, _ = bench.reduce_conjugate_pairs(roots, strict=True)
    residues, direct, _, _ = _dss_to_res(
        poles, case["delays"], case["A"], case["B"], case["C"], case["D"]
    )
    length = 48000
    ir = bench.pyFDN.pr_to_impz(
        residues, poles, direct, pairs, length, mode="lowMemory"
    )
    ref = bench.pyFDN.dss_to_impz(
        *[case[key] for key in ["delays", "A", "B", "C", "D"]], length
    )
    ir_relative = float(np.linalg.norm(ir - ref) / np.linalg.norm(ref))
    n = len(case["delays"])
    mimo_length = 4096
    r_mimo, d_mimo, _, _ = _dss_to_res(
        poles, case["delays"], case["A"], np.eye(n), np.eye(n), np.zeros((n, n))
    )
    ir_mimo = bench.pyFDN.pr_to_impz(
        r_mimo, poles, d_mimo, pairs, mimo_length, mode="lowMemory"
    )
    ref_mimo = bench.pyFDN.dss_to_impz(
        case["delays"], case["A"], np.eye(n), np.eye(n), np.zeros((n, n)), mimo_length
    )
    mimo_relative = float(np.linalg.norm(ir_mimo - ref_mimo) / np.linalg.norm(ref_mimo))
    expanded_p, expanded_r = [], []
    for pole, residue, pair in zip(poles, residues[:, 0, 0], pairs, strict=True):
        expanded_p.append(pole)
        expanded_r.append(residue)
        if pair:
            expanded_p.append(pole.conjugate())
            expanded_r.append(residue.conjugate())
    expanded_p, expanded_r = np.array(expanded_p), np.array(expanded_r)
    radius = float(np.exp(1 / 4800))
    ws = np.exp(-1j * np.linspace(0, np.pi, 32769)) / radius
    error_sq = ref_sq = max_error = 0.0
    for start in range(0, len(ws), 256):
        w = ws[start : start + 256]
        forward = w[:, None] ** case["delays"]
        matrices = np.eye(n) - forward[:, :, None] * case["A"]
        hf = (case["C"] @ np.linalg.solve(matrices, forward[:, :, None] * case["B"]))[
            ..., 0, 0
        ]
        hm = w * ((1 / (1 - w[:, None] * expanded_p)) @ expanded_r)
        error_sq += float(np.sum(np.abs(hm - hf) ** 2))
        ref_sq += float(np.sum(np.abs(hf) ** 2))
        max_error = max(max_error, float(np.max(np.abs(hm - hf))))
    print(
        json.dumps(
            {
                "example": case["name"],
                "expected_poles": len(p) - 1,
                "retained_poles": int(np.sum(1 + pairs)),
                "omitted_poles": len(missing),
                "missing_poles": [[float(z.real), float(z.imag)] for z in missing],
                "retained_pole_error": float(
                    np.max(np.abs(reference[row] - roots[col]))
                ),
                "ir_samples": length,
                "ir_max_error": float(np.max(np.abs(ir - ref))),
                "ir_relative_l2": ir_relative,
                "mimo_ir_samples": mimo_length,
                "mimo_ir_max_error": float(np.max(np.abs(ir_mimo - ref_mimo))),
                "mimo_ir_relative_l2": mimo_relative,
                "frequency_points": len(ws),
                "frequency_radius_z": radius,
                "windowed_frequency_relative_l2": float(np.sqrt(error_sq / ref_sq)),
                "windowed_frequency_max_abs_error": max_error,
                "acceptable_original_io": ir_relative < 1e-9,
                "acceptable_all_ports": mimo_relative < 1e-9,
            }
        ),
        flush=True,
    )


def main():
    results_path = (
        Path(__file__).resolve().parents[1]
        / "docs/benchmarks/lindsey_fox_modal_examples.jsonl"
    )
    failed = {
        record["example"]
        for record in map(json.loads, results_path.read_text().splitlines())
        if record["backend"] == "lindsey_fox" and record["status"] != "ok"
    }
    with contextlib.redirect_stdout(sys.stderr):
        cases = bench.cases()
    for case in cases:
        if case["name"] in failed:
            run_case(case)


if __name__ == "__main__":
    main()
