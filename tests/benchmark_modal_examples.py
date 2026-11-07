"""Benchmark the actual modal-example setups without changing their solvers.

Run from an editable checkout with the examples dependencies installed::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_modal_examples.py

Setup cells are read from the notebooks; plots and audio cells are not run.
FLAMO cases use an experimental adapter below, not a new public solver API.
It clears known SOS denominators before Fourier coefficient recovery, then
uses the unchanged Lindsey-Fox kernel and FLAMO's existing SVD/residue helpers.
Each backend has a 60-second limit per repetition. JSON lines include failures,
missing pole counts, and independent time-domain IR errors as well as timings.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import signal
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import pyFDN
from pyFDN.auxiliary.flamo import to_numpy
from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.poles import reduce_conjugate_pairs
from pyFDN.translate.flamo_to_pr import (
    _conjugate_symmetrize,
    _count_loop_filter_poles,
    _dss_to_res_flamo,
    _FDNLoopFlamo,
    _iter_leaf_modules,
    _refine_pole_via_svd,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def setup_cell(filename: str, output: str) -> dict[str, Any]:
    """Execute only the original setup cell returning the requested variable."""
    tree = ast.parse((EXAMPLES / filename).read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        returns = [item for item in node.body if isinstance(item, ast.Return)]
        if not returns or not isinstance(returns[-1].value, ast.Tuple):
            continue
        names = [
            item.id for item in returns[-1].value.elts if isinstance(item, ast.Name)
        ]
        if output not in names:
            continue
        node.decorator_list = []
        namespace = {"np": np, "pyFDN": pyFDN, "torch": torch}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"),
            namespace,
        )
        values = namespace[node.name](
            **{arg.arg: namespace[arg.arg] for arg in node.args.args}
        )
        return dict(zip(names, values, strict=True))
    raise ValueError(f"No setup returning {output} in {filename}")


def model_polynomial(model: Any, degree: int) -> tuple[np.ndarray, Any]:
    """Polynomial numerator of det(P(w)), with loop SOS denominators cleared.

    Benchmark-only: finite FIR/SOS loops in the selected examples, a known
    polynomial degree, and constant external input/output/direct gains.
    """
    decomposition = pyFDN.flamo_decompose_for_pr(model)
    recursion = decomposition.recursion_module
    denominators = []
    for branch in (recursion.feedforward, recursion.feedback):
        for leaf in _iter_leaf_modules(branch):
            if getattr(leaf, "n_sections", None) is not None:
                sos = to_numpy(leaf.map(leaf.param)).reshape(leaf.n_sections, 6, -1)
                denominators.extend(sos[:, 3:, :].transpose(0, 2, 1).reshape(-1, 3))
    nfft = 1 << degree.bit_length()
    samples = np.empty(nfft // 2 + 1, dtype=complex)
    with torch.no_grad():
        for k in range(samples.size):
            w = np.exp(-2j * np.pi * k / nfft)
            matrix = recursion.probe_recursion_w(
                torch.tensor(w, dtype=torch.complex128)
            )
            denominator = np.prod(
                [np.polynomial.polynomial.polyval(w, a) for a in denominators]
            )
            samples[k] = np.linalg.det(to_numpy(matrix)) * denominator
    coefficients = np.fft.irfft(samples, n=nfft)[: degree + 1]
    coefficients[0] = float(np.prod([a[0] for a in denominators]))
    return coefficients, decomposition


def model_lindsey_fox(model: Any, degree: int, lossless: bool) -> tuple:
    coefficients, decomposition = model_polynomial(model, degree)
    if lossless:
        low = high = 1.0
    else:
        # Reciprocal Fujiwara lower bound; the selected loops are stable.
        ascending = coefficients[::-1]
        powers = np.arange(1, degree + 1)
        high = 1.0
        low = 1 / (2 * np.max(np.abs(ascending[1:] / ascending[0]) ** (1 / powers)))
    poles = lindsey_fox_roots(
        coefficients[::-1], float(low), float(high), max_attempts=4, max_seconds=60
    )
    loop = _FDNLoopFlamo(decomposition.recursion_module)
    poles = np.array([_refine_pole_via_svd(complex(z), loop) for z in poles])
    poles = _conjugate_symmetrize(poles)
    poles, paired, unpaired = reduce_conjugate_pairs(poles, strict=True)
    try:
        residues, direct, _, _ = _dss_to_res_flamo(poles, loop, decomposition)
    except np.linalg.LinAlgError as error:
        expanded = int(np.sum(1 + paired))
        raise RuntimeError(
            f"FLAMO residue extraction failed after LF found {expanded}/{degree} poles: {error}"
        ) from error
    return residues, poles, direct, paired, {"nonPairedPoles": unpaired}


def static_lindsey_fox(delays, a, b, c, d):
    """Original four-grid baseline, without fallback or matrix refinement."""
    from pyFDN.auxiliary.math import general_char_poly_fft
    from pyFDN.translate.dss_to_pr import _dss_to_res

    singular = np.linalg.svd(a, compute_uv=False)
    bounds = singular[[0, -1], None] ** (1 / np.array([min(delays), max(delays)]))
    coefficients = general_char_poly_fft(delays, a)
    poles = lindsey_fox_roots(
        coefficients[::-1],
        float(bounds.min()),
        float(bounds.max()),
        max_attempts=4,
        max_seconds=60,
    )
    poles, pairs, unpaired = reduce_conjugate_pairs(poles, strict=True)
    residues, direct, _, _ = _dss_to_res(poles, delays, a, b, c, d)
    return residues, poles, direct, pairs, {"nonPairedPoles": unpaired}


def cases() -> list[dict[str, Any]]:
    result = []
    for filename, output in [
        ("example_dss_to_pr_direct.py", "A"),
        ("example_fdn_eigenvectors.py", "A"),
        ("example_random_fdn_statistics.py", "feedback_matrix"),
        ("example_spread_fdn_poles.py", "types"),
    ]:
        setup = setup_cell(filename, output)
        matrices = setup.get("feedback_matrix", setup.get("A"))
        if not isinstance(matrices, dict):
            matrices = {"default": matrices}
        for variant, a in matrices.items():
            result.append(
                {
                    "name": f"{filename}/{variant}",
                    "delays": setup["delays"],
                    "A": a,
                    "B": setup.get("input_gain", setup.get("b")),
                    "C": setup.get("output_gain", setup.get("c")),
                    "D": setup.get("direct", setup.get("d")),
                }
            )
    # Figure 6's exact 21 frozen snapshots, plus the unmodulated matrix once.
    delays = np.array([53, 61, 71, 79, 89, 101, 113, 127])
    triangle = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * np.arange(21) / 21))
    angles = [pyFDN.kronecker_angles(8)] + [
        pyFDN.kronecker_angles(8, theta2=np.pi / 4 + np.pi * weight)
        for weight in triangle
    ]
    for index, angle in enumerate(angles):
        result.append(
            {
                "name": f"example_kronecker_matrix.py/{index:02d}",
                "delays": delays,
                "A": pyFDN.kronecker_matrix(angle),
                "B": np.ones((8, 1)),
                "C": np.ones((1, 8)),
                "D": np.zeros((1, 1)),
            }
        )
    for filename, output in [
        ("example_paraunitary_fdn.py", "feedback_matrix"),
        ("example_pole_boundaries.py", "absorption"),
        ("example_dss_to_pr_flamo.py", "model"),
    ]:
        setup = setup_cell(filename, output)
        if "build" in setup:
            build = setup["build"]
            model = setup["model"]
        else:
            build = pyFDN.FDNBuild(
                A=setup["feedback_matrix"],
                B=setup["input_gain"],
                C=setup["output_gain"],
                D=setup["direct"],
                delays=setup["delays"],
                fs=setup["fs"],
                post_delay=setup.get("absorption"),
            )
            model = pyFDN.build_to_flamo(build, dtype=torch.float64, device="cpu")
        decomposition = pyFDN.flamo_decompose_for_pr(model)
        lossless = build.A.ndim == 3
        if lossless:
            gcp = pyFDN.general_char_poly(build.delays, build.A)
            degree = int(np.flatnonzero(np.abs(gcp) > 1e-12 * np.max(np.abs(gcp)))[-1])
        else:
            degree = int(build.delays.sum()) + _count_loop_filter_poles(
                decomposition.recursion_module
            )
        result.append(
            {
                "name": filename,
                "model": model,
                "build": build,
                "degree": degree,
                "lossless": lossless,
            }
        )
    return result


def benchmark(case: dict[str, Any], repeat: int, seconds: int) -> None:
    reference = None
    if "model" in case:
        degree = case["degree"]

        def render(length):
            return pyFDN.build_to_impz(case["build"], length)

        solvers = {
            "eai": lambda: pyFDN.flamo_to_pr(
                case["model"],
                num_poles=degree,
                maximum_iterations=80,
                refinement_tol=1e-10,
            ),
            "lindsey_fox_adapter": lambda: model_lindsey_fox(
                case["model"], degree, case["lossless"]
            ),
        }
    else:
        degree = int(case["delays"].sum())
        args = [case[key] for key in ("delays", "A", "B", "C", "D")]

        def render(length):
            return pyFDN.dss_to_impz(*args, length)

        solvers = {
            mode: (
                lambda mode=mode: static_lindsey_fox(*args)
                if mode == "lindsey_fox"
                else pyFDN.dss_to_pr(*args, mode=mode)
            )
            for mode in ("eig", "eai", "lindsey_fox")
        }
        if "kronecker" in case["name"]:
            solvers.pop("eig")
            solvers = {"roots": lambda: pyFDN.dss_to_pr(*args, mode="roots"), **solvers}
    length = min(20000, max(2048, 2 * degree))
    ir_time = render(length)
    for name, solve in solvers.items():
        record = {"example": case["name"], "backend": name, "degree": degree}
        start = perf_counter()
        try:
            timings = []
            for _ in range(repeat):
                signal.alarm(seconds)
                start = perf_counter()
                result = solve()
                timings.append(perf_counter() - start)
                signal.alarm(0)
            residues, poles, direct, pairs, _ = result
            expanded = int(np.sum(1 + pairs))
            ir = pyFDN.pr_to_impz(
                residues, poles, direct, pairs, length, mode="lowMemory"
            )
            record.update(
                seconds=float(np.median(timings)),
                poles=expanded,
                ir_error=float(np.max(np.abs(ir - ir_time))),
            )
            if (
                reference is None
                and not name.startswith("lindsey_fox")
                and expanded == degree
                and record["ir_error"] < 1e-7
            ):
                reference = poles
            if reference is not None and reference.size == poles.size:
                row, col = linear_sum_assignment(np.abs(reference[:, None] - poles))
                record["pole_error"] = float(
                    np.max(np.abs(reference[row] - poles[col]))
                )
            record["status"] = (
                "ok"
                if expanded == degree
                and record["ir_error"] < 1e-7
                and record.get("pole_error", 0) < 1e-7
                else "accuracy_failure"
            )
        except (RuntimeError, ValueError, TimeoutError, np.linalg.LinAlgError) as error:
            record.update(
                status=type(error).__name__,
                error=str(error),
                elapsed=perf_counter() - start,
            )
        finally:
            signal.alarm(0)
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--example", help="Only names containing this substring")
    args = parser.parse_args()
    if args.repeat < 1 or args.seconds < 1:
        parser.error("repeat and seconds must be positive")

    def timeout_handler(*_args: Any) -> None:
        raise TimeoutError("per-repetition time limit")

    signal.signal(signal.SIGALRM, timeout_handler)
    # Warm lazy imports/model construction on a separate small system.
    pyFDN.dss_to_pr(
        [3, 4], [[0.4, 0.1], [-0.2, 0.5]], [[1], [0]], [[1, 0]], [[0]], mode="eai"
    )
    with contextlib.redirect_stdout(sys.stderr):
        example_cases = cases()
    for case in example_cases:
        if args.example is None or args.example in case["name"]:
            benchmark(case, args.repeat, args.seconds)


if __name__ == "__main__":
    main()
