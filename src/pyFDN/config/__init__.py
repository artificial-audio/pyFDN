"""pyFDN declarative graph-IR configuration package.

Single canonical IR rooted at `Shell`. Closed structural composites
(`Shell`, `Series`, `Parallel`, `Recursion`); open terminal hierarchy
rooted at `Module`, with subclasses registered via `@register_module`.
"""
from pyFDN.config.builders import (
    # Composites
    Shell,
    Series,
    Parallel,
    Recursion,
    # Terminal base + concrete
    Module,
    Gain,
    Matrix,
    Delay,
    Filter,
    Biquad,
    OnePole,
    GEQ,
    # Matrix-type taxonomy
    MATRIX_TYPES,
    # Union
    GraphElement,
    # Registry
    register_module,
    get_module_class,
    registered_modules,
    # Matrix builders
    hadamard_matrix,
    householder_matrix,
    random_orthogonal_matrix,
    diagonal_matrix,
    velvet_feedback_matrix,
    scattering_matrix,
    # Filter / delay / tap builders
    one_pole_absorption_filter,
    random_delays,
    prime_delays,
    unity_input_tap,
    unity_output_tap,
    # High-level builder
    build_vanilla_config,
    # Error type
    ValidationError,
)
from pyFDN.config.validate import IOSpec, infer_io, validate
from pyFDN.config.to_json import (
    config_to_dict,
    config_to_json,
    config_to_json_file,
)
from pyFDN.config.from_json import (
    dict_to_config,
    json_to_config,
    json_file_to_config,
)


__all__ = [
    # Composites
    "Shell",
    "Series",
    "Parallel",
    "Recursion",
    # Terminals
    "Module",
    "Gain",
    "Matrix",
    "Delay",
    "Filter",
    "Biquad",
    "OnePole",
    "GEQ",
    "MATRIX_TYPES",
    "GraphElement",
    # Registry
    "register_module",
    "get_module_class",
    "registered_modules",
    # Builders
    "hadamard_matrix",
    "householder_matrix",
    "random_orthogonal_matrix",
    "diagonal_matrix",
    "velvet_feedback_matrix",
    "scattering_matrix",
    "one_pole_absorption_filter",
    "random_delays",
    "prime_delays",
    "unity_input_tap",
    "unity_output_tap",
    "build_vanilla_config",
    # Validation
    "ValidationError",
    "IOSpec",
    "infer_io",
    "validate",
    # JSON (stubbed)
    "config_to_dict",
    "config_to_json",
    "config_to_json_file",
    "dict_to_config",
    "json_to_config",
    "json_file_to_config",
]
