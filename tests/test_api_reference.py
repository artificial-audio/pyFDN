"""Guard the API reference against silent drift.

``docs/api_reference.rst`` is a hand-curated, categorised list of the *headline*
public surface (categories mirror the package's submodule structure). This test
makes drift impossible: every name in ``pyFDN.__all__`` must be either

  * documented in the reference, or
  * listed below as intentionally undocumented (low-level plumbing exported for
    advanced use, or submodule aliases that aren't standalone functions).

Adding a new public export therefore forces a deliberate choice between the two,
instead of the export silently never appearing in the docs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyFDN
import pyFDN.eq

_REFERENCE = Path(__file__).resolve().parent.parent / "docs" / "api_reference.rst"

REMOVED_EQ_API = {
    "absorption_geq",
    "bandpass_filter",
    "design_geq",
    "first_order_absorption",
    "first_order_shelf_sos",
    "first_order_shelving_eq",
    "geq_sos",
    "one_pole_absorption",
    "one_pole_sos",
    "shelf_crossover_omega",
    "shelving_filter",
}

# Exported from ``pyFDN`` but intentionally NOT in the API reference.
#   - low-level linear-algebra / completion helpers (advanced / plumbing)
#   - submodule aliases exposed for namespaced access (e.g. ``pyFDN.allpass.*``)
# To promote any of these to the headline docs, move it into a category in
# ``docs/api_reference.rst`` and delete it from this set.
INTENTIONALLY_UNDOCUMENTED = {
    # generate.allpass_FDN completion plumbing
    "apply_diagonal_similarity",
    "block_matrix",
    "check_completion",
    "diag_inv_sqrt",
    "diag_sqrt",
    "diagonal_similarity_from_abs2_lyapunov",
    "eig_sqrt_psd",
    "hermitize",
    "map_back_from_similarity",
    "orth_error",
    "sqrtm_psd",
    # misc internal helper
    "is_almost_zero",
    # loss taxonomy base classes (subclass to write a loss; not used directly)
    "ResponseLoss",
    "ParameterLoss",
    # excitation plumbing (train_fdn builds this for you)
    "impulse_excitation",
    # low-level FLAMO graph/recursion manipulation (advanced / plumbing)
    "flamo_delay_feedback_matrix",
    "swap_flamo_recursion_paths",
    # submodule aliases (namespaced access, not standalone functions)
    "allpass",
    "allpass_completion",
    "td",
}


def test_pre_refactor_eq_api_is_removed() -> None:
    for name in REMOVED_EQ_API:
        assert not hasattr(pyFDN, name)
        assert not hasattr(pyFDN.eq, name)


def _documented_names() -> set[str]:
    """Names referenced as autosummary entries in the API reference.

    Matches standalone indented ``pyFDN.<name>`` lines, so inline code examples
    such as ``pyFDN.random_orthogonal(4)`` are not counted.
    """
    text = _REFERENCE.read_text(encoding="utf-8")
    return set(re.findall(r"^\s+pyFDN\.(\w+)\s*$", text, flags=re.MULTILINE))


def test_reference_only_lists_real_exports() -> None:
    documented = _documented_names()
    exported = set(pyFDN.__all__)
    extra = sorted(documented - exported)
    assert not extra, f"API reference lists names not in pyFDN.__all__: {extra}"


def test_documented_and_plumbing_are_disjoint() -> None:
    overlap = sorted(_documented_names() & INTENTIONALLY_UNDOCUMENTED)
    assert not overlap, f"names both documented and marked plumbing: {overlap}"


def test_every_export_is_classified() -> None:
    documented = _documented_names()
    exported = set(pyFDN.__all__)
    unclassified = sorted(exported - documented - INTENTIONALLY_UNDOCUMENTED)
    assert not unclassified, (
        "These exports are neither documented in docs/api_reference.rst nor "
        "marked INTENTIONALLY_UNDOCUMENTED. Classify each as headline (add it to "
        f"a category in the reference) or plumbing (add it to the set): {unclassified}"
    )


def test_plumbing_set_stays_in_sync_with_exports() -> None:
    stale = sorted(INTENTIONALLY_UNDOCUMENTED - set(pyFDN.__all__))
    assert not stale, (
        f"INTENTIONALLY_UNDOCUMENTED lists names no longer exported by pyFDN: {stale}"
    )
