from __future__ import annotations
import warnings
from typing import Any, Dict, Set
import numpy as np

# Optional deps
try:
    import scipy.io as sio
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import scipy.sparse as sp
    _HAVE_SPARSE = True
except Exception:
    _HAVE_SPARSE = False

def _is_matv73(path: str) -> bool:
    """Detect v7.3 (HDF5) MAT files via magic bytes."""
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89HDF\r\n\x1a\n"

# -------------------- v5/v7 helpers (SciPy) --------------------

_KNOWN_NONCLASS_TYPES = {
    "double","single",
    "int8","uint8","int16","uint16","int32","uint32","int64","uint64",
    "logical","char","cell","struct","sparse"
}

def _scan_v5v7_classdef_vars(path: str) -> Set[str]:
    """Use scipy.io.whosmat to find variables that are custom MATLAB classes."""
    class_vars: Set[str] = set()
    try:
        entries = sio.whosmat(path)  # [(name, shape, class)]
    except Exception:
        return class_vars
    for name, _shape, cls in entries:
        # MATLAB reports builtins as above; anything else is a class name
        if cls not in _KNOWN_NONCLASS_TYPES:
            class_vars.add(name)
    return class_vars

def _convert_cell_array_inplace(arr: np.ndarray) -> np.ndarray:
    """
    MATLAB cell arrays arrive as object ndarrays. Recursively convert elements
    but keep ndarray (dtype=object) and original shape.
    """
    it = np.nditer(arr, flags=["refs_ok", "multi_index"], op_flags=["readwrite"])
    for x in it:
        x[...] = _convert_elem(x.item())
    return arr

def _matstruct_to_dict(mobj: Any) -> Dict[str, Any]:
    """Convert scipy.io 'mat_struct' to a nested Python dict (values converted)."""
    d: Dict[str, Any] = {}
    for fn in getattr(mobj, "_fieldnames", []):
        d[fn] = _convert_elem(getattr(mobj, fn))
    return d

def _char_to_str(a: np.ndarray) -> str | list[str]:
    """1-D char → str; 2-D char matrix → list[str]."""
    if not isinstance(a, np.ndarray):
        return str(a)
    if a.dtype.kind in ("U", "S"):
        if a.ndim == 1:
            return "".join(a.tolist())
        if a.ndim == 2:
            return ["".join(row.tolist()) for row in a]
    return str(a)

def _is_mat_struct(elem: Any) -> bool:
    if not _HAVE_SCIPY:
        return False
    mat_struct_type = getattr(sio.matlab.mio5_params, "mat_struct", None)
    return mat_struct_type is not None and isinstance(elem, mat_struct_type)

def _convert_struct_ndarray(arr: np.ndarray) -> Any:
    """
    Convert an ndarray of mat_structs to dicts.
    If shape == (1,1), unwrap to single dict.
    Otherwise, return ndarray(dtype=object) of dicts preserving shape.
    """
    out = np.empty(arr.shape, dtype=object)
    it = np.nditer(arr, flags=["multi_index", "refs_ok"])
    for x in it:
        out[it.multi_index] = _matstruct_to_dict(x.item())
    if arr.shape == (1, 1):
        return out.item()
    return out

def _convert_elem(elem: Any) -> Any:
    """
    Recursive converter for SciPy loadmat outputs, with the following rules:
    - cell arrays: keep as np.ndarray (dtype=object), elements converted
    - structs: dict or ndarray of dicts
    - char arrays: str / list[str]
    - sparse: keep scipy.sparse
    - scalars: unbox 0‑d numpy arrays to Python scalars
    - numeric/logical arrays: keep as np.ndarray
    """
    # MATLAB struct
    if _HAVE_SCIPY:
        if _is_mat_struct(elem):
            return _matstruct_to_dict(elem)

    # cell arrays or struct arrays (object ndarrays)
    if isinstance(elem, np.ndarray) and elem.dtype == object:
        # Peek first element to check if struct array
        if elem.size > 0 and _is_mat_struct(elem.flat[0]):
            return _convert_struct_ndarray(elem)
        else:
            return _convert_cell_array_inplace(elem)

    # char arrays
    if isinstance(elem, np.ndarray) and elem.dtype.kind in ("U", "S"):
        return _char_to_str(elem)

    # sparse
    if _HAVE_SPARSE and sp.issparse(elem):
        return elem

    # scalars
    if isinstance(elem, np.ndarray) and elem.ndim == 0:
        return elem.item()

    # numeric/logical 1×1 arrays → Python scalars
    if (
        isinstance(elem, np.ndarray)
        and elem.dtype != object
        and elem.dtype.kind not in ("U", "S")  # not char arrays
        and elem.shape == (1, 1)
    ):
        return elem[0, 0]

    # everything else: leave numpy arrays (including 0-d scalars) as-is
    return elem

def _clean_top_level(d: Dict[str, Any]) -> Dict[str, Any]:
    drop = {"__header__", "__version__", "__globals__"}
    return {k: v for k, v in d.items() if k not in drop}

# -------------------- v7.3 helpers (HDF5) --------------------

def _scan_v73_classdef_vars_with_h5py(path: str) -> Set[str]:
    """
    Inspect top-level HDF5 nodes and record those with MATLAB_class == 'object'.
    This requires h5py; if unavailable we can't reliably detect custom classes.
    """
    class_vars: Set[str] = set()
    try:
        import h5py  # type: ignore
    except Exception:
        warnings.warn(
            "Detected v7.3 MAT but 'h5py' is not installed; "
            "cannot pre-detect classdef objects. They may still load oddly.",
            RuntimeWarning,
        )
        return class_vars

    with h5py.File(path, "r") as f:
        for name, node in f.items():
            # MATLAB puts attributes on datasets/groups
            try:
                mcls = node.attrs.get("MATLAB_class", None)
                if isinstance(mcls, bytes):
                    mcls = mcls.decode("utf-8", "ignore")
                if mcls == "object":
                    class_vars.add(name)
            except Exception:
                pass
    return class_vars

# -------------------- Public API --------------------

def load_mat_workspace(
    path: str,
    *,
    simplify_chars: bool = True,
    allow_v73_with_mat73: bool = True,
) -> Dict[str, Any]:
    """
    Load a MATLAB .mat workspace dump into Python/Numpy while:
      • Skipping MATLAB classdef objects (warns and omits those variables)
      • Scalars become Python numbers (0-d arrays are unboxed)
      • Keeping cell arrays as numpy object ndarrays (elements converted)
      • Singleton dimensions are always preserved (loadmat(..., squeeze_me=False))

    Parameters
    ----------
    path : str
        Path to .mat file.
    simplify_chars : bool
        Convert char arrays to str / list[str].
    allow_v73_with_mat73 : bool
        For v7.3: use 'mat73' if available; otherwise raise a clear error.

    Returns
    -------
    dict[str, Any]
        Mapping variable name -> converted value (no classdef objects included).
    """
    if _is_matv73(path):
        # Pre-scan for classdef variables
        class_vars = _scan_v73_classdef_vars_with_h5py(path)

        if not allow_v73_with_mat73:
            raise RuntimeError(
                "This is a MATLAB v7.3 (HDF5) file. Enable allow_v73_with_mat73=True "
                "and install 'mat73', or re-save as v7 in MATLAB."
            )
        try:
            import mat73  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Detected v7.3 MAT file but 'mat73' is not installed. "
                "Install it (pip install mat73) or re-save as v7."
            ) from e

        raw = mat73.loadmat(path)  # dict of variables

        # Drop classdef vars and warn
        for v in sorted(class_vars):
            if v in raw:
                warnings.warn(f"Skipping MATLAB class object '{v}' (classdef).", RuntimeWarning)
                raw.pop(v, None)

        if not simplify_chars:
            return raw

        # Normalize char arrays inside the structure returned by mat73 and unbox 0-d scalars
        def fix_nodes(x):
            if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
                return _char_to_str(x)
            if isinstance(x, np.ndarray) and x.ndim == 0:
                return x.item()
            if (
                isinstance(x, np.ndarray)
                and x.dtype != object
                and x.dtype.kind not in ("U", "S")  # not char arrays
                and x.shape == (1, 1)
            ):
                return x[0, 0]
            if isinstance(x, dict):
                return {k: fix_nodes(v) for k, v in x.items()}
            if isinstance(x, list):
                return [fix_nodes(v) for v in x]
            return x

        return {k: fix_nodes(v) for k, v in raw.items()}

    # ----- v5/v7 path (SciPy) -----
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy is required to read non-v7.3 MAT files. Please install scipy.")

    # Identify classdef variables up front
    class_vars = _scan_v5v7_classdef_vars(path)

    # Load (squeeze_me=False to preserve singleton dimensions)
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    data = _clean_top_level(data)

    out: Dict[str, Any] = {}
    for k, v in data.items():
        if k in class_vars:
            warnings.warn(f"Skipping MATLAB class object '{k}' (classdef).", RuntimeWarning)
            continue

        cv = _convert_elem(v)

        # Optionally normalize top-level char arrays
        if simplify_chars and isinstance(cv, np.ndarray) and cv.dtype.kind in ("U", "S"):
            cv = _char_to_str(cv)


        out[k] = cv

    return out