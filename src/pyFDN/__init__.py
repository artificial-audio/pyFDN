"""Top-level package for pyFDN."""

from importlib import import_module

__author__ = "Facundo Franchino"
__version__ = "0.1.0"

__all__ = [
    # dsp
    "FeedbackDelay",
    "FIRMatrixFilter",
    "SOSFilterBank",
    # acoustics
    "absorption_filters",
    "echo_density",
    "estimate_rt_bands",
    "absorption_to_rt",
    "edc",
    "one_pole_absorption",
    "rt_to_gain_per_sample",
    "rt_to_slope",
    "slope_to_rt",
    "sos_gain_per_sample_curves",
    # delay utilities
    "matrix_delay_approximation",
    "mgrpdelay",
    "ms_to_smp",
    # matrix generators
    "allpass_in_fdn",
    "anderson_matrix",
    "complete_orthogonal",
    "construct_cascaded_paraunitary_matrix",
    "construct_paraunitary_from_elementals",
    "construct_velvet_feedback_matrix",
    "degree_one_lossless",
    "fdn_matrix_gallery",
    "fdn_system_gallery",
    "filter_matrix_gallery",
    "FDNSystem",
    "householder_matrix",
    "is_almost_zero",
    "nearest_orthogonal",
    "nearest_sign_agnostic_orthogonal",
    "random_matrix_shift",
    "random_orthogonal",
    "schroeder_reverberator",
    "shift_matrix",
    "shift_matrix_distribute",
    "tiny_rotation_matrix",
    "vanilla_FDN",
    # graphicEQ
    "absorption_geq",
    "bandpass_filter",
    "design_geq",
    "graphic_eq",
    "probe_sos",
    "shelving_filter",
    # polynomial and matrix maths
    "det_polynomial",
    "general_char_poly",
    "interpolate_orthogonal",
    "is_orthogonal",
    "is_unilossless",
    "matrix_convolution",
    "matrix_polyder",
    "matrix_polyval",
    "matrix_sqrt",
    "negpolyder",
    "outer_sum_approximation",
    "poly_degree",
    "polyder_rational",
    "polydiag",
    # general utilities
    "db_to_lin",
    "db_to_sq",
    "ensure_3d",
    "hertz_to_unit",
    "hertz_to_rad",
    "rad_to_hertz",
    "is_bounding_curve",
    "last_nonzero_indices",
    "lin_to_db",
    "sq_to_db",
    "mulaw_decode",
    "mulaw_encode",
    "peak_normalize",
    "pole_boundaries",
    "skew",
    # state-space translators
    "dss_to_flamo",
    "dss_to_impz",
    "dss_to_pr_direct",
    "flamo_to_pr",
    "flamo_decompose_for_pr",
    "flamo_extract_pr_decomposition",
    "FlamoDecompositionForPR",
    "dss_to_pr_flamo",
    "dss_to_ss",
    "dss_to_tf",
    "impz_to_res",
    "mtf_to_impz",
    "pr_to_impz",
    # fdn processing
    "process_fdn",
    # plotting
    "plot_impulse_response_matrix",
    "plot_matrix",
    "plot_system_matrix",
    "plot_spectrogram",
    "downsample_minmax",
    "downsample_plotly_trace",
    "downsampled_scatter",
    # FLAMO graph
    "flamo_model_to_nodes",
    "flamo_nodes_flat",
    "draw_flamo_graph",
    # SDN (scattering delay network)
    "SDN",
    # allpass FDN
    "allpass",
    "allpass_completion",
    "apply_diagonal_similarity",
    "block_matrix",
    "check_completion",
    "complete_fdn",
    "complete_full_mimo_halmos",
    "complete_general_mimo_svd",
    "diagonal_similarity_from_abs2_lyapunov",
    "diag_inv_sqrt",
    "diag_sqrt",
    "eig_sqrt_psd",
    "hermitize",
    "homogeneous_allpass_fdn",
    "map_back_from_similarity",
    "rand_admissible_homogeneous_allpass",
    "orth_error",
    "sqrtm_psd",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
]

# acoustics and absorption
from .auxiliary.acoustics import (
    absorption_filters,
    absorption_to_rt,
    echo_density,
    edc,
    estimate_rt_bands,
    one_pole_absorption,
    rt_to_gain_per_sample,
    rt_to_slope,
    slope_to_rt,
    sos_gain_per_sample_curves,
)
from .auxiliary.allpass import (
    is_allpass,
    is_paraunitary,
    is_uniallpass,
    nested_allpass,
    poletti_allpass,
    series_allpass,
)

# delay utilities
from .auxiliary.delay import matrix_delay_approximation, mgrpdelay, ms_to_smp
from .auxiliary.flamo_graph import (
    draw_flamo_graph,
    flamo_model_to_nodes,
    flamo_nodes_flat,
)

# polynomial and matrix maths
from .auxiliary.math import (
    det_polynomial,
    general_char_poly,
    interpolate_orthogonal,
    is_orthogonal,
    is_unilossless,
    matrix_convolution,
    matrix_polyder,
    matrix_polyval,
    matrix_sqrt,
    negpolyder,
    outer_sum_approximation,
    poly_degree,
    polyder_rational,
    polydiag,
)

# plotting
from .auxiliary.plot import (
    downsample_minmax,
    downsample_plotly_trace,
    downsampled_scatter,
    plot_impulse_response_matrix,
    plot_matrix,
    plot_spectrogram,
    plot_system_matrix,
)

# tiny rotation matrix
from .auxiliary.tiny_rotation_matrix import tiny_rotation_matrix

# general utilities
from .auxiliary.utils import (
    db_to_lin,
    db_to_sq,
    ensure_3d,
    hertz_to_rad,
    hertz_to_unit,
    is_bounding_curve,
    last_nonzero_indices,
    lin_to_db,
    mulaw_decode,
    mulaw_encode,
    peak_normalize,
    pole_boundaries,
    rad_to_hertz,
    skew,
    sq_to_db,
)

# dsp components
from .dsp.dfilt_matrix import FIRMatrixFilter
from .dsp.feedback_delay import FeedbackDelay
from .dsp.sos_filter_bank import SOSFilterBank
from .generate.allpass_FDN import allpass_completion
from .generate.allpass_FDN.allpass_completion import (
    apply_diagonal_similarity,
    block_matrix,
    check_completion,
    complete_fdn,
    complete_full_mimo_halmos,
    complete_general_mimo_svd,
    diag_inv_sqrt,
    diag_sqrt,
    diagonal_similarity_from_abs2_lyapunov,
    eig_sqrt_psd,
    hermitize,
    map_back_from_similarity,
    orth_error,
    sqrtm_psd,
)
from .generate.allpass_FDN.homogeneous_allpass_fdn import homogeneous_allpass_fdn
from .generate.allpass_FDN.rand_admissible_homogeneous_allpass import (
    rand_admissible_homogeneous_allpass,
)
from .generate.allpass_in_fdn import allpass_in_fdn
from .generate.anderson_matrix import anderson_matrix
from .generate.complete_orthogonal import complete_orthogonal
from .generate.construct_cascaded_paraunitary_matrix import (
    construct_cascaded_paraunitary_matrix,
)
from .generate.construct_paraunitary_from_elementals import (
    construct_paraunitary_from_elementals,
)
from .generate.construct_velvet_feedback_matrix import construct_velvet_feedback_matrix
from .generate.degree_one_lossless import degree_one_lossless
from .generate.fdn_matrix_gallery import (
    FDNSystem,
    fdn_matrix_gallery,
    fdn_system_gallery,
    filter_matrix_gallery,
)
from .generate.householder_matrix import householder_matrix
from .generate.is_almost_zero import is_almost_zero
from .generate.nearest_orthogonal import nearest_orthogonal
from .generate.nearest_sign_agnostic_orthogonal import nearest_sign_agnostic_orthogonal
from .generate.random_matrix_shift import random_matrix_shift

# matrix generators
from .generate.random_orthogonal import random_orthogonal
from .generate.schroeder_reverberator import schroeder_reverberator
from .generate.SDN import SDN
from .generate.shift_matrix import shift_matrix
from .generate.shift_matrix_distribute import shift_matrix_distribute
from .generate.vanilla_FDN import vanilla_FDN
from .graphicEQ import (
    absorption_geq,
    bandpass_filter,
    design_geq,
    graphic_eq,
    probe_sos,
    shelving_filter,
)

# fdn processing
from .process import process_fdn

# state-space translators
from .translate.dss_to_flamo import dss_to_flamo
from .translate.dss_to_impz import dss_to_impz
from .translate.dss_to_pr_direct import dss_to_pr_direct
from .translate.dss_to_pr_flamo import (
    FlamoDecompositionForPR,
    dss_to_pr_flamo,
    flamo_decompose_for_pr,
    flamo_extract_pr_decomposition,
    flamo_to_pr,
)
from .translate.dss_to_ss import dss_to_ss
from .translate.dss_to_tf import dss_to_tf
from .translate.impz_to_res import impz_to_res
from .translate.mtf_to_impz import mtf_to_impz
from .translate.pr_to_impz import pr_to_impz

# Expose allpass submodule for pyFDN.allpass.is_uniallpass etc.
allpass = import_module(".auxiliary.allpass", __name__)
