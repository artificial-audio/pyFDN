"""Top-level package for pyFDN."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ._public_api import EXPORT_MAP, EXPORTS

__author__ = """Kalyan Pandey"""
__email__ = "pandey.kalyan416@gmail.com"
__version__ = "0.1.0"

__all__ = EXPORTS


def __getattr__(name: str):
    module_name = EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module 'pyFDN' has no attribute '{name}'")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .auxiliary.acoustics import (
        absorption_filters,
        absorption_to_t60,
        one_pole_absorption,
        rt60_to_slope,
        slope_to_rt60,
    )
    from .auxiliary.delay import matrix_delay_approximation, mgrpdelay, ms2smp
    from .auxiliary.filters import TFMatrix, ZFIR, ZFilter, ZScalar, ZSOS, ZTF
    from .auxiliary.math import (
        det_polynomial,
        matrix_convolution,
        matrix_polyder,
        matrix_polyval,
        negpolyder,
        outer_sum_approximation,
        poly_degree,
        polyder_rational,
        polydiag,
    )
    from .auxiliary.utils import (
        db2mag,
        ensure_3d,
        hertz2unit,
        is_bounding_curve,
        last_nonzero_indices,
        mag2db,
        pole_boundaries,
    )
    from .dsp.dfiltmatrix import DFiltMatrix
    from .dsp.feedback_delay import FeedbackDelay
    from .dsp.filter_matrix import FilterMatrix, IIRFilterState, SOSFilterState
    from .generate.construct_cascaded_paraunitary_matrix import (
        construct_cascaded_paraunitary_matrix,
    )
    from .generate.construct_velvet_feedback_matrix import (
        construct_velvet_feedback_matrix,
    )
    from .generate.is_almost_zero import is_almost_zero
    from .generate.random_matrix_shift import random_matrix_shift
    from .generate.random_orthogonal import random_orthogonal
    from .generate.shift_matrix import shift_matrix
    from .generate.shift_matrix_distribute import shift_matrix_distribute
    from .process import process_fdn
    from .recursive.biquads import Biquads
    from .recursive.core import RecursionCore
    from .recursive.delay_lines import Delay, DelayRead, DelayWrite
    from .recursive.feedback_mix import FeedbackMix
    from .recursive.input_tap import InputTap
    from .recursive.output_tap import OutputTap
    from .recursive.stage import Stage
    from .translate.dss2impz import dss2impz
    from .translate.dss2ss import dss2ss
