"""Load the repository's modal examples for the public-API benchmark.

Only setup cells run; plotting, audio, and decomposition cells are excluded.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pyFDN
from pyFDN.translate.flamo_to_pr import _count_loop_filter_poles

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
        namespace: dict[str, Any] = {"np": np, "pyFDN": pyFDN, "torch": torch}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"),
            namespace,
        )
        values = namespace[node.name](
            **{arg.arg: namespace[arg.arg] for arg in node.args.args}
        )
        return dict(zip(names, values, strict=True))
    raise ValueError(f"No setup returning {output} in {filename}")


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
        dss: pyFDN.DSS
        if "dss" in setup:
            dss, model = setup["dss"], setup["model"]
            post_delay, fir = setup["filters"]["post_delay"], None
        else:
            matrix = np.asarray(setup["feedback_matrix"])
            # a filter (FIR) feedback matrix is a graph operator on the feedback
            # path, with the identity as the DSS feedback matrix
            fir = matrix if matrix.ndim == 3 else None
            n = matrix.shape[0]
            dss = pyFDN.DSS(
                setup["delays"],
                np.eye(n) if fir is not None else matrix,
                setup["input_gain"],
                setup["output_gain"],
                setup["direct"],
            )
            post_delay = setup.get("absorption")
            model = pyFDN.dss_to_flamo(
                *dss,
                setup["fs"],
                dtype=torch.float64,
                device="cpu",
                post_delay=post_delay,
                post_matrix=None
                if fir is None
                else pyFDN.fir_matrix_module(
                    fir, 2**16, dtype=torch.float64, device="cpu"
                ),
            )
        graph = pyFDN.dss_to_td(
            *dss,
            post_delay=post_delay,
            post_matrix=None if fir is None else pyFDN.td.MatrixFIR(fir),
        )
        decomposition = pyFDN.flamo_decompose_for_pr(model)
        lossless = fir is not None
        if fir is not None:
            gcp = pyFDN.general_char_poly(dss.delays, fir)
            degree = int(np.flatnonzero(np.abs(gcp) > 1e-12 * np.max(np.abs(gcp)))[-1])
        else:
            degree = int(dss.delays.sum()) + _count_loop_filter_poles(
                decomposition.recursion_module
            )
        result.append(
            {
                "name": filename,
                "model": model,
                "td": graph,
                "absorption": post_delay is not None,
                "degree": degree,
                "lossless": lossless,
            }
        )
    return result
