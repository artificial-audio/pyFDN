"""Top-level package for pyFDN."""

from importlib import import_module

__author__ = "Facundo Franchino"
__version__ = "0.4.2"

__all__ = [
    # acoustics
    "echo_density",
    "estimate_initial_level_bands",
    "estimate_rt_bands",
    "edc",
    "octave_band_filterbank",
    "octave_bands",
    "rt_to_gain_per_sample",
    "rt_to_slope",
    "slope_amplitude_to_level",
    "slope_to_rt",
    "sos_gain_per_sample_curves",
    # delay utilities
    "matrix_delay_approximation",
    "mgrpdelay",
    "ms_to_smp",
    "flamo_delay_feedback_matrix",
    "swap_flamo_recursion_paths",
    "flamo_time_response",
    "flamo_freq_response",
    "flamo_process",
    # building a FLAMO FDN graph from numpy values
    "assemble_fdn_core",
    "wrap_fdn_shell",
    "gain_module",
    "delay_module",
    "matrix_module",
    "fir_matrix_module",
    "sos_filter_module",
    "hook_module",
    "AttenuationFilter",
    "OutputEQ",
    "audio_metadata",
    "available_audio",
    "load_audio",
    # packaged references and presets
    "paper_link",
    "paper_reference",
    "available_fdn_presets",
    "get_fdn_preset",
    "load_fdn_preset",
    "FDNPreset",
    "fdn_preset_from_dict",
    "fdn_preset_to_dict",
    "save_fdn_preset",
    "fdn_build_from_dict",
    "fdn_build_to_dict",
    "load_fdn_build",
    "save_fdn_build",
    # matrix generators
    "allpass_in_fdn",
    "anderson_matrix",
    "complete_orthogonal",
    "construct_cascaded_paraunitary_matrix",
    "construct_paraunitary_from_elementals",
    "construct_velvet_feedback_matrix",
    "degree_one_lossless",
    "fdn_matrix_gallery",
    "fdn_build_gallery",
    "fdn_system_gallery",
    "filter_matrix_gallery",
    "FDNBuild",
    "FDNSystem",
    "householder_matrix",
    "is_almost_zero",
    "nearest_orthogonal",
    "nearest_sign_agnostic_orthogonal",
    "random_matrix_shift",
    "random_orthogonal",
    "rotation_matrix_from_angles",
    "sample_delay_lengths",
    "schroeder_reverberator",
    "shift_matrix",
    "shift_matrix_distribute",
    "tiny_rotation_matrix",
    # eq
    "EQDesign",
    "decay_to_first_order_shelf",
    "decay_to_geq",
    "decay_to_one_pole",
    "first_order_shelf_biquad",
    "gain_to_bounded_geq",
    "gain_to_first_order_shelf",
    "gain_to_geq",
    "gain_to_one_pole",
    "geq_design_matrix",
    "highshelf_biquad",
    "lowshelf_biquad",
    "one_pole_biquad",
    "peaking_biquad",
    "probe_sos",
    # polynomial and matrix maths
    "adj_poly",
    "adjugate",
    "det_polynomial",
    "general_char_poly",
    "interpolate_orthogonal",
    "is_orthogonal",
    "is_unilossless",
    "loop_tf",
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
    "fade_out",
    "hertz_to_unit",
    "hertz_to_rad",
    "rad_to_hertz",
    "is_bounding_curve",
    "last_nonzero_indices",
    "lin_to_db",
    "max_corr",
    "sq_to_db",
    "mulaw_decode",
    "mulaw_encode",
    "peak_normalize",
    "pole_boundaries",
    "skew",
    # state-space translators
    "build_to_impz",
    "build_to_flamo",
    "dss_to_flamo",
    "dss_to_impz",
    "dss_to_pr",
    "flamo_to_pr",
    "flamo_decompose_for_pr",
    "flamo_extract_pr_decomposition",
    "FlamoDecompositionForPR",
    "dss_to_ss",
    "dss_to_tf",
    "impz_to_res",
    "mtf_to_impz",
    "pr_to_impz",
    # fdn processing
    "process_fdn",
    # training
    "build_fdn",
    "trainable_from_build",
    "trainable_from_preset",
    "LOSSLESS_ALIAS_DECAY_DB",
    "build_set_decay",
    "Trainable",
    "train_fdn",
    "TrainLog",
    # training: what a loss sees
    "Response",
    "model_response",
    "impulse_excitation",
    "param",
    "params",
    "ParamRef",
    # training: losses
    "Loss",
    "ResponseLoss",
    "ParameterLoss",
    "FlatMagnitude",
    "AsymmetricFlatMagnitude",
    "FlatSpectrogram",
    "MatchMagnitude",
    "MatchSpectrogram",
    "MatchMelSpectrogram",
    "MatchImpulseResponse",
    "MatchEnergyDecay",
    "MatchCumulativeEnergy",
    "MatchDC",
    "MatchESR",
    "MatchLogCosh",
    "MatchSDSDR",
    "MatchSISDR",
    "MatchSNR",
    "Energy",
    "Sparsity",
    "L1",
    "L2",
    # features
    "mimo_rir_eigenvalues_per_frequency",
    # plotting
    "animate",
    "plot_db_per_sample",
    "plot_edc",
    "plot_fdn_parameter",
    "plot_FDN_build",
    "plot_impulse_response",
    "plot_impulse_response_matrix",
    "plot_matrix",
    "plot_matrix_grid",
    "plot_system_matrix",
    "plot_spectrogram",
    "downsample_minmax",
    "downsample_plotly_trace",
    "downsampled_scatter",
    # notebook display (marimo)
    "labeled_audio",
    # FLAMO graph
    "flamo_model_to_nodes",
    "flamo_nodes_flat",
    "plot_flamo_graph",
    "extract_build",
    # time-domain graph engine
    "td",
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
    echo_density,
    edc,
    estimate_initial_level_bands,
    estimate_rt_bands,
    octave_band_filterbank,
    octave_bands,
    rt_to_gain_per_sample,
    rt_to_slope,
    slope_amplitude_to_level,
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
from .auxiliary.audio import audio_metadata, available_audio, load_audio

# delay utilities
from .auxiliary.delay import (
    flamo_delay_feedback_matrix,
    matrix_delay_approximation,
    mgrpdelay,
    ms_to_smp,
    swap_flamo_recursion_paths,
)
from .auxiliary.flamo import (
    assemble_fdn_core,
    delay_module,
    fir_matrix_module,
    flamo_freq_response,
    flamo_process,
    flamo_time_response,
    gain_module,
    hook_module,
    matrix_module,
    sos_filter_module,
    wrap_fdn_shell,
)
from .auxiliary.flamo_graph import (
    extract_build,
    flamo_model_to_nodes,
    flamo_nodes_flat,
    plot_flamo_graph,
)
from .auxiliary.marimo_utils import labeled_audio

# polynomial and matrix maths
from .auxiliary.math import (
    adj_poly,
    adjugate,
    det_polynomial,
    general_char_poly,
    interpolate_orthogonal,
    is_orthogonal,
    is_unilossless,
    loop_tf,
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
    animate,
    downsample_minmax,
    downsample_plotly_trace,
    downsampled_scatter,
    plot_db_per_sample,
    plot_edc,
    plot_FDN_build,
    plot_fdn_parameter,
    plot_impulse_response,
    plot_impulse_response_matrix,
    plot_matrix,
    plot_matrix_grid,
    plot_spectrogram,
    plot_system_matrix,
)

# tiny rotation matrix
from .auxiliary.tiny_rotation_matrix import (
    rotation_matrix_from_angles,
    tiny_rotation_matrix,
)

# general utilities
from .auxiliary.utils import (
    db_to_lin,
    db_to_sq,
    ensure_3d,
    fade_out,
    hertz_to_rad,
    hertz_to_unit,
    is_bounding_curve,
    last_nonzero_indices,
    lin_to_db,
    max_corr,
    mulaw_decode,
    mulaw_encode,
    peak_normalize,
    pole_boundaries,
    rad_to_hertz,
    skew,
    sq_to_db,
)
from .build import (
    FDNBuild,
    fdn_build_from_dict,
    fdn_build_to_dict,
    load_fdn_build,
    save_fdn_build,
)
from .eq import (
    EQDesign,
    decay_to_first_order_shelf,
    decay_to_geq,
    decay_to_one_pole,
    first_order_shelf_biquad,
    gain_to_bounded_geq,
    gain_to_first_order_shelf,
    gain_to_geq,
    gain_to_one_pole,
    geq_design_matrix,
    highshelf_biquad,
    lowshelf_biquad,
    one_pole_biquad,
    peaking_biquad,
    probe_sos,
)
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
from .generate.fdn_build_gallery import fdn_build_gallery
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
from .generate.sample_delay_lengths import sample_delay_lengths
from .generate.schroeder_reverberator import schroeder_reverberator
from .generate.SDN import SDN
from .generate.shift_matrix import shift_matrix
from .generate.shift_matrix_distribute import shift_matrix_distribute
from .preset import (
    FDNPreset,
    available_fdn_presets,
    fdn_preset_from_dict,
    fdn_preset_to_dict,
    get_fdn_preset,
    load_fdn_preset,
    save_fdn_preset,
)

# fdn processing
from .process import process_fdn
from .references import paper_link, paper_reference

# training (torch/flamo are imported lazily inside these)
from .train import (
    L1,
    L2,
    LOSSLESS_ALIAS_DECAY_DB,
    AsymmetricFlatMagnitude,
    AttenuationFilter,
    Energy,
    FlatMagnitude,
    FlatSpectrogram,
    Loss,
    MatchCumulativeEnergy,
    MatchEnergyDecay,
    MatchImpulseResponse,
    MatchMagnitude,
    MatchMelSpectrogram,
    MatchSpectrogram,
    MatchDC,
    MatchESR,
    MatchLogCosh,
    MatchSDSDR,
    MatchSISDR,
    MatchSNR,
    OutputEQ,
    ParameterLoss,
    ParamRef,
    Response,
    ResponseLoss,
    Sparsity,
    Trainable,
    TrainLog,
    build_fdn,
    build_set_decay,
    impulse_excitation,
    model_response,
    param,
    params,
    train_fdn,
    trainable_from_build,
    trainable_from_preset,
)
# features
from .train import (
    mimo_rir_eigenvalues_per_frequency,
    )

# state-space translators
from .translate.dss_to_flamo import build_to_flamo, dss_to_flamo
from .translate.dss_to_impz import build_to_impz, dss_to_impz
from .translate.dss_to_pr import dss_to_pr
from .translate.dss_to_ss import dss_to_ss
from .translate.dss_to_tf import dss_to_tf
from .translate.flamo_to_pr import (
    FlamoDecompositionForPR,
    flamo_decompose_for_pr,
    flamo_extract_pr_decomposition,
    flamo_to_pr,
)
from .translate.impz_to_res import impz_to_res
from .translate.mtf_to_impz import mtf_to_impz
from .translate.pr_to_impz import pr_to_impz

# Expose allpass submodule for pyFDN.allpass.is_uniallpass etc.
allpass = import_module(".auxiliary.allpass", __name__)

# Time-domain graph engine (pyFDN.td operators and connectors).
from . import td  # noqa: E402
