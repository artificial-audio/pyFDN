"""Diagnostic: extend only LF's four-grid cap to ten, keeping complex128.

Run from the checkout root with BLAS threads limited to one::

    python tests/benchmark_kronecker_resolution.py

The AST assertion guards the single benchmark-only solver change. Reference
roots are used only for validation; they never seed LF or matrix refinement.
SIGALRM limits root search to 60 seconds per case on Unix.
"""

import ast
import contextlib
import inspect
import json
import signal
import sys
from time import perf_counter

import benchmark_modal_examples as bench
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pyFDN.auxiliary.math import general_char_poly_fft
from pyFDN.translate.dss_to_flamo import dss_to_flamo
from pyFDN.translate.dss_to_pr import _dss_to_res

module = sys.modules[bench.lindsey_fox_roots.__module__]
tree = ast.parse(inspect.getsource(bench.lindsey_fox_roots))
changes = 0
for node in ast.walk(tree):
    if (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "attempt"
    ):
        assert isinstance(node.iter, ast.Call) and node.iter.args[0].value == 4
        node.iter.args[0].value = 10
        changes += 1
assert changes == 1
namespace = dict(vars(module))
exec(compile(tree, "<LF-resolution-experiment>", "exec"), namespace)
solve = namespace["lindsey_fox_roots"]
failed = {
    row["example"]
    for row in map(
        json.loads, open("docs/benchmarks/lindsey_fox_problematic_partial_modes.jsonl")
    )
}
with contextlib.redirect_stdout(sys.stderr):
    cases = bench.cases()


def timeout(*_):
    raise TimeoutError("60 second limit")


signal.signal(signal.SIGALRM, timeout)
for case in cases:
    if case["name"] not in failed:
        continue
    p = general_char_poly_fft(case["delays"], case["A"])
    state = {}

    def trace(frame, event, arg, state=state):
        if frame.f_code is solve.__code__:
            frame.f_trace_lines = False
            if event == "return":
                state.update(
                    attempt=frame.f_locals.get("attempt"),
                    nfft=frame.f_locals.get("nfft"),
                )
            return trace
        return None

    signal.alarm(60)
    start = perf_counter()
    sys.settrace(trace)
    try:
        roots = solve(p[::-1], 1.0, 1.0)
        elapsed = perf_counter() - start
        sys.settrace(None)
        signal.alarm(0)
        reference = np.roots(p)
        row, col = linear_sum_assignment(np.abs(reference[:, None] - roots))
        roots = bench._conjugate_symmetrize(roots)
        poles, pairs, _ = bench.reduce_conjugate_pairs(roots, strict=True)
        residues, direct, _, _ = _dss_to_res(
            poles, case["delays"], case["A"], case["B"], case["C"], case["D"]
        )
        ir = bench.pyFDN.pr_to_impz(
            residues, poles, direct, pairs, 48000, mode="lowMemory"
        )
        ref = bench.pyFDN.dss_to_impz(
            *[case[k] for k in ["delays", "A", "B", "C", "D"]], 48000
        )
        model = dss_to_flamo(
            A=case["A"],
            B=case["B"],
            C=case["C"],
            D=case["D"],
            delays=case["delays"],
            fs=48000,
            nfft=2048,
            shell=False,
            dtype=torch.float64,
            device="cpu",
        )
        decomposition = bench.pyFDN.flamo_decompose_for_pr(model)
        loop = bench._FDNLoopFlamo(decomposition.recursion_module)
        refined = np.array(
            [bench._refine_pole_via_svd(complex(z), loop) for z in roots]
        )
        refined = bench._conjugate_symmetrize(refined)
        rp, paired, _ = bench.reduce_conjugate_pairs(refined, strict=True)
        rr, dd, _, _ = _dss_to_res(
            rp, case["delays"], case["A"], case["B"], case["C"], case["D"]
        )
        refined_ir = bench.pyFDN.pr_to_impz(rr, rp, dd, paired, 48000, mode="lowMemory")
        print(
            json.dumps(
                {
                    "matrix_refined_ir_relative_l2": float(
                        np.linalg.norm(refined_ir - ref) / np.linalg.norm(ref)
                    ),
                    "matrix_refined_ir_max_error": float(
                        np.max(np.abs(refined_ir - ref))
                    ),
                    "example": case["name"],
                    "status": "ok",
                    "dtype": str(roots.dtype),
                    "grid_attempts": state["attempt"] + 1,
                    "nfft": state["nfft"],
                    "fft_spacing_hz": 48000 / state["nfft"],
                    "root_search_seconds": elapsed,
                    "poles": int(np.sum(1 + pairs)),
                    "matched_pole_error": float(
                        np.max(np.abs(reference[row] - roots[col]))
                    ),
                    "ir_max_error": float(np.max(np.abs(ir - ref))),
                    "ir_relative_l2": float(
                        np.linalg.norm(ir - ref) / np.linalg.norm(ref)
                    ),
                }
            ),
            flush=True,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "example": case["name"],
                    "status": type(error).__name__,
                    "error": str(error),
                    "elapsed": perf_counter() - start,
                    **state,
                }
            ),
            flush=True,
        )
    finally:
        sys.settrace(None)
        signal.alarm(0)
