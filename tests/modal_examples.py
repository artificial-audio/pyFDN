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
