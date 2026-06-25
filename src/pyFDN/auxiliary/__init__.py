"""Auxiliary modules (utils, acoustics, allpass, flamo wrappers, etc.)."""

from .allpass import (
    is_allpass,
    is_paraunitary,
    is_uniallpass,
    nested_allpass,
    poletti_allpass,
    series_allpass,
)
from .audio import load_audio
from .flamo import (
    assemble_fdn_core,
    delay_module,
    fir_matrix_module,
    flamo_freq_response,
    flamo_process,
    flamo_time_response,
    gain_module,
    matrix_module,
    output_layer,
    sos_filter_module,
    wrap_fdn_shell,
)
from .flamo_graph import (
    extract_build,
    flamo_model_to_nodes,
    flamo_nodes_flat,
    plot_flamo_graph,
)
from .poles import reduce_conjugate_pairs
from .utils import skew

__all__ = [
    "skew",
    "gain_module",
    "delay_module",
    "fir_matrix_module",
    "sos_filter_module",
    "matrix_module",
    "assemble_fdn_core",
    "wrap_fdn_shell",
    "output_layer",
    "flamo_time_response",
    "flamo_freq_response",
    "flamo_process",
    "load_audio",
    "poletti_allpass",
    "series_allpass",
    "nested_allpass",
    "is_uniallpass",
    "is_allpass",
    "is_paraunitary",
    "flamo_model_to_nodes",
    "extract_build",
    "flamo_nodes_flat",
    "plot_flamo_graph",
    "reduce_conjugate_pairs",
]
