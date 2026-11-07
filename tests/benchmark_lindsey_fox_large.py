"""Bounded, isolated scaling trials up to 100,000 poles / 128 delay lines.

Run from the checkout root with BLAS threads limited to one::

    python tests/benchmark_lindsey_fox_large.py > large-modal.jsonl

Dense seeded orthogonal matrices use exactly the requested total delay.
Coefficient checks use independent off-grid determinants. Separate sparse
polynomials have analytically known simple roots. Timings are single trials.
Full decomposition is attempted only below conservative memory/order limits;
skipped stages and timeouts are results, never claimed successful solves.
SIGALRM deadlines require Unix; each case also has a parent-process timeout.
"""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np

from pyFDN.auxiliary.lindsey_fox import lindsey_fox_roots
from pyFDN.auxiliary.math import general_char_poly_fft
from pyFDN.translate.dss_to_impz import dss_to_impz
from pyFDN.translate.dss_to_pr import dss_to_pr
from pyFDN.translate.pr_to_impz import pr_to_impz


def array_sizes(order, lines):
    return {
        "pairing_cost_bytes": 8 * order**2,
        "pairing_complex_temporary_bytes": 16 * order**2,
        "pairing_combined_bytes": 24 * order**2,
        "expanded_all_port_residues_bytes": 16 * order * lines**2,
    }


def dense_case(order, lines):
    assert order >= lines > 0
    rng = np.random.default_rng(73128 + lines)
    weights = rng.uniform(0.6, 1.4, lines)
    delays = 1 + np.floor((order - lines) * weights / weights.sum()).astype(int)
    delays[: order - int(delays.sum())] += 1
    assert int(delays.sum()) == order and np.all(delays > 0)
    q, r = np.linalg.qr(rng.normal(size=(lines, lines)))
    a = 0.9 * q @ np.diag(np.sign(np.diag(r)))
    return delays, a


def worker(args):
    base = {
        "order": args.order,
        "lines": args.matrix_lines,
        "case": "sparse_oracle" if args.oracle else "dense_orthogonal",
    }
    interrupted = {}

    def timeout(_signum, frame):
        interrupted["interrupted_function"] = frame.f_code.co_name
        while frame is not None:
            if frame.f_code.co_name == "lindsey_fox_roots":
                interrupted.update(
                    nfft=frame.f_locals.get("nfft"),
                    candidates=int(getattr(frame.f_locals.get("roots"), "size", 0)),
                )
                break
            frame = frame.f_back
        raise TimeoutError("stage wall-clock deadline")

    signal.signal(signal.SIGALRM, timeout)

    def emit(stage, **data):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        print(
            json.dumps(
                {
                    **base,
                    "stage": stage,
                    "process_peak_rss_mib": rss_bytes / 2**20,
                    **data,
                }
            ),
            flush=True,
        )

    def run(stage, seconds, operation):
        interrupted.clear()
        start = perf_counter()
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            details = operation()
            emit(stage, status="ok", seconds=perf_counter() - start, **details)
        except Exception as error:
            emit(
                stage,
                status=type(error).__name__,
                seconds=perf_counter() - start,
                error=str(error),
                **interrupted,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    cache = {}
    if args.oracle:
        coefficients = np.zeros(args.order + 1)
        coefficients[[0, -1]] = [-0.9, 1]
        radius = 0.9 ** (1 / args.order)
        expected = radius * np.exp(2j * np.pi * np.arange(args.order) / args.order)
        cache["coefficients"] = coefficients
        low = high = radius
    else:
        delays, a = dense_case(args.order, args.matrix_lines)
        singular = np.linalg.svd(a, compute_uv=False)
        bounds = singular[[0, -1], None] ** (
            1 / delays[[np.argmin(delays), np.argmax(delays)]]
        )
        low, high = float(bounds.min()), float(bounds.max())
        emit(
            "allocation_estimates",
            status="estimate",
            **array_sizes(args.order, args.matrix_lines),
        )

        def coefficients_stage():
            start = perf_counter()
            p = general_char_poly_fft(delays, a)
            build_seconds = perf_counter() - start
            cache["coefficients"] = p[::-1]
            rng = np.random.default_rng(91)
            w = np.exp(2j * np.pi * rng.uniform(0, 1, 32))
            expected_values = np.array(
                [
                    np.linalg.det(
                        np.eye(args.matrix_lines) - np.diag(point**delays) @ a
                    )
                    for point in w
                ]
            )
            actual = np.polynomial.polynomial.polyval(w, p)
            error = float(
                np.max(np.abs(expected_values - actual))
                / np.max(np.abs(expected_values))
            )
            assert len(p) == args.order + 1 and np.all(np.isfinite(p)) and error < 1e-8
            return {
                "construction_seconds": build_seconds,
                "off_grid_normalized_max_error": error,
                "coefficients": len(p),
                "coefficient_fft_size": 1 << args.order.bit_length(),
            }

        run("coefficients", 120, coefficients_stage)

        def svd_stage():
            for angle in np.linspace(0.1, 3.0, 16):
                np.linalg.svd(np.diag(np.exp(1j * angle * delays)) - a)
            return {
                "matrix_probes": 16,
                "measurement": "SVD throughput only; not pole refinement or an accuracy check",
            }

        run("svd_probes", 15, svd_stage)

    def root_stage():
        roots = lindsey_fox_roots(
            cache["coefficients"], low, high, max_seconds=args.root_seconds
        )
        assert roots.size == args.order
        details = {"poles": int(roots.size)}
        if args.oracle:
            # Angular ordering matches the regular polygon without an O(M^2) assignment.
            phase = np.mod(np.angle(roots), 2 * np.pi)
            index = np.rint(phase * args.order / (2 * np.pi)).astype(int) % args.order
            assert np.unique(index).size == args.order
            error = float(np.max(np.abs(roots - expected[index])))
            assert error < 1e-10
            details["analytic_max_pole_error"] = error
        else:
            # Independent original-matrix check; no companion solver is built.
            qualities = []
            for index in np.linspace(0, roots.size - 1, min(32, roots.size)).astype(
                int
            ):
                s = np.linalg.svd(np.diag(roots[index] ** delays) - a, compute_uv=False)
                qualities.append(float(s[-1] / s[0]))
            assert max(qualities) < 1e-8
            details["sampled_matrix_relative_singular_value"] = max(qualities)
            details["matrix_checked_poles"] = len(qualities)
        return details

    if "coefficients" in cache:
        run("root_search", args.root_timeout, root_stage)
    if args.oracle:
        return
    sizes = array_sizes(args.order, args.matrix_lines)
    if (
        args.order > 1000
        or max(
            sizes["pairing_combined_bytes"], sizes["expanded_all_port_residues_bytes"]
        )
        > 512 * 2**20
    ):
        emit(
            "decomposition",
            status="not_attempted",
            reason="order/memory preflight; no EAI or Hungarian assignment is allocated",
            **sizes,
        )
        return

    def decomposition_stage():
        b = np.ones((args.matrix_lines, 1)) / np.sqrt(args.matrix_lines)
        c = b.T
        result = dss_to_pr(delays, a, b, c, [[0]], mode="lindsey_fox")
        reference = dss_to_impz(delays, a, b, c, [[0]], 4096)
        ir = pr_to_impz(*result[:4], 4096, mode="lowMemory")
        error = float(np.linalg.norm(ir - reference) / np.linalg.norm(reference))
        assert result[4]["complete"] and error < 1e-8
        return {
            "solver": result[4]["solver"],
            "poles": result[4]["returnedPoles"],
            "ir_relative_l2": error,
            "responseValidation": result[4]["responseValidation"],
        }

    run("decomposition", 30, decomposition_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--lines", nargs="+", type=int, default=[4, 16, 64, 128])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--order", type=int, default=1000)
    parser.add_argument("--matrix-lines", type=int, default=4)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--root-seconds", type=float, default=3)
    parser.add_argument("--root-timeout", type=float, default=15)
    args = parser.parse_args()
    if (
        min(args.orders) < 1
        or min(args.lines) < 1
        or args.root_seconds <= 0
        or args.root_timeout <= 0
    ):
        parser.error("orders, lines, and deadlines must be positive")
    if args.worker:
        worker(args)
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for order in args.orders:
        for lines in [*args.lines, None]:
            command = [sys.executable, __file__, "--worker", "--order", str(order)]
            command += [
                "--root-seconds",
                str(args.root_seconds),
                "--root-timeout",
                str(args.root_timeout),
            ]
            command += ["--oracle"] if lines is None else ["--matrix-lines", str(lines)]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=180, env=env
                )
                print(result.stdout, end="", flush=True)
                if result.returncode:
                    print(
                        json.dumps(
                            {
                                "order": order,
                                "lines": lines,
                                "stage": "worker",
                                "status": "failed",
                                "stderr": result.stderr[-2000:],
                            }
                        ),
                        flush=True,
                    )
            except subprocess.TimeoutExpired as error:
                if error.stdout:
                    print(error.stdout.decode(), end="", flush=True)
                print(
                    json.dumps(
                        {
                            "order": order,
                            "lines": lines,
                            "stage": "worker",
                            "status": "parent_timeout",
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
