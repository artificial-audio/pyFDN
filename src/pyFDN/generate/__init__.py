"""Matrix and structure generator submodules for pyFDN."""

from . import construct_cascaded_paraunitary_matrix
from . import construct_velvet_feedback_matrix
from . import is_almost_zero
from . import random_matrix_shift
from . import random_orthogonal
from . import shift_matrix
from . import shift_matrix_distribute

# Keep submodule names unshadowed for call sites that import/patch
# ``pyFDN.generate.<module>.<symbol>`` paths.
__all__ = [
    "construct_cascaded_paraunitary_matrix",
    "construct_velvet_feedback_matrix",
    "is_almost_zero",
    "random_matrix_shift",
    "random_orthogonal",
    "shift_matrix",
    "shift_matrix_distribute",
]
