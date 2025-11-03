from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass
from numpy.typing import ArrayLike
from pyFDN.auxiliary.ztf import ZTF
from pyFDN.auxiliary.zscalar import ZScalar

class IIRFilterState:
    """Streaming Direct Form I filter section."""

    def __init__(self, b: ArrayLike, a: ArrayLike) -> None:
        b_arr = np.atleast_1d(np.asarray(b, dtype=float))
        a_arr = np.atleast_1d(np.asarray(a, dtype=float))
        if a_arr.size == 0:
            raise ValueError("Denominator must include at least one coefficient")
        if np.isclose(a_arr[0], 0.0):
            raise ValueError("Leading denominator coefficient must be non-zero")

        if not np.isclose(a_arr[0], 1.0):
            b_arr = b_arr / a_arr[0]
            a_arr = a_arr / a_arr[0]

        self.b = b_arr
        self.a = a_arr
        self.nb = self.b.size
        self.na = self.a.size
        self.x_state = np.zeros(max(self.nb - 1, 0), dtype=float)
        self.y_state = np.zeros(max(self.na - 1, 0), dtype=float)

    def process(self, block: ArrayLike) -> np.ndarray:
        x = np.asarray(block, dtype=float).reshape(-1)
        y = np.zeros_like(x)

        for idx, sample in enumerate(x):
            acc = self.b[0] * sample
            if self.nb > 1:
                acc += float(np.dot(self.b[1:], self.x_state))
            if self.na > 1:
                acc -= float(np.dot(self.a[1:], self.y_state))
            y[idx] = acc

            if self.nb > 1:
                if self.nb > 2:
                    self.x_state[1:] = self.x_state[:-1]
                self.x_state[0] = sample
            if self.na > 1:
                if self.na > 2:
                    self.y_state[1:] = self.y_state[:-1]
                self.y_state[0] = acc

        return y


@dataclass
class FilterMatrix:
    """Matrix of filters mirroring MATLAB ``dfiltMatrix`` semantics."""

    kind: str
    is_diagonal: bool
    output_channels: int
    input_channels: int
    dtype: np.dtype
    matrix: Optional[np.ndarray] = None
    filters: Optional[list[list[IIRFilterState]]] = None

    @classmethod
    def from_data(
        cls,
        data: ArrayLike | ZTF | "FilterMatrix",
        *,
        is_diagonal: Optional[bool] = None,
    ) -> "FilterMatrix":
        if data is None:
            raise ValueError("Filter data must not be None")

        if isinstance(data, FilterMatrix):
            return data

        if isinstance(data, ZTF):
            diag = data.is_diagonal if is_diagonal is None else is_diagonal
            def _real_coefficients(arr: np.ndarray, label: str) -> np.ndarray:
                if np.iscomplexobj(arr):
                    if np.allclose(arr.imag, 0.0, atol=1e-12):
                        return arr.real.astype(float)
                    raise ValueError(f"{label} contains complex coefficients which are unsupported")
                return np.asarray(arr, dtype=float)

            num = _real_coefficients(data.numerator, "Numerator")
            den = _real_coefficients(data.denominator, "Denominator")
            n_rows, n_cols, _ = num.shape
            if diag and n_cols != 1:
                raise ValueError("Diagonal ZTF must have second dimension equal to 1")
            col_iter = 1 if diag else n_cols
            filters: list[list[IIRFilterState]] = []
            for row in range(n_rows):
                row_filters: list[IIRFilterState] = []
                for col in range(col_iter):
                    row_filters.append(IIRFilterState(num[row, col, :], den[row, col, :]))
                filters.append(row_filters)
            return cls(
                kind="iir",
                is_diagonal=diag,
                output_channels=n_rows,
                input_channels=n_rows if diag else n_cols,
                dtype=float,
                filters=filters,
            )

        if isinstance(data, ZScalar):
            diag = data.is_diagonal if is_diagonal is None else is_diagonal
            raw_matrix = data.at(1.0)
            if np.iscomplexobj(raw_matrix):
                if np.allclose(np.imag(raw_matrix), 0.0, atol=1e-12):
                    raw_matrix = np.real(raw_matrix)
                else:
                    raise ValueError("Static matrix contains complex entries which are unsupported")
            matrix = np.asarray(raw_matrix, dtype=float)
            if diag:
                diag_vals = np.diag(matrix) if matrix.ndim == 2 else matrix.reshape(-1)
                condensed = diag_vals.reshape(-1, 1)
                return cls(
                    kind="static",
                    is_diagonal=True,
                    output_channels=condensed.shape[0],
                    input_channels=condensed.shape[0],
                    dtype=condensed.dtype,
                    matrix=condensed,
                )

            return cls(
                kind="static",
                is_diagonal=False,
                output_channels=matrix.shape[0],
                input_channels=matrix.shape[1],
                dtype=matrix.dtype,
                matrix=matrix,
            )

        arr = np.asarray(data, dtype=float)
        diag = bool(is_diagonal) if is_diagonal is not None else False
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim == 2:
            n_rows, n_cols = arr.shape
            if diag:
                if n_cols not in (1, n_rows):
                    raise ValueError("Diagonal static matrix must have shape (N, 1) or (N, N)")
                input_channels = n_rows
            else:
                input_channels = n_cols
            return cls(
                kind="static",
                is_diagonal=diag,
                output_channels=n_rows,
                input_channels=input_channels,
                dtype=arr.dtype,
                matrix=arr,
            )

        if arr.ndim == 3:
            n_rows, n_cols, _ = arr.shape
            col_iter = 1 if diag else n_cols
            filters = []
            for row in range(n_rows):
                row_filters = []
                for col in range(col_iter):
                    row_filters.append(IIRFilterState(arr[row, col, :], [1.0]))
                filters.append(row_filters)
            return cls(
                kind="iir",
                is_diagonal=diag,
                output_channels=n_rows,
                input_channels=n_rows if diag else n_cols,
                dtype=float,
                filters=filters,
            )

        raise ValueError("Unsupported filter data dimensionality")

    def filter(self, block: ArrayLike) -> np.ndarray:
        block_arr = np.asarray(block, dtype=float)
        if block_arr.ndim != 2:
            raise ValueError("Filter input must be 2-D")
        if block_arr.shape[1] != self.input_channels:
            raise ValueError("Input channel mismatch for filter matrix")

        if self.kind == "static":
            if self.is_diagonal:
                gains = (
                    np.diag(self.matrix)
                    if self.matrix.shape[1] == self.output_channels
                    else self.matrix[:, 0]
                )
                return block_arr * gains
            return block_arr @ self.matrix.T

        if self.kind == "iir":
            out = np.zeros((block_arr.shape[0], self.output_channels), dtype=float)
            if self.is_diagonal:
                for row in range(self.output_channels):
                    out[:, row] = self.filters[row][0].process(block_arr[:, row])
            else:
                for row in range(self.output_channels):
                    acc = np.zeros(block_arr.shape[0], dtype=float)
                    for col in range(self.input_channels):
                        acc += self.filters[row][col].process(block_arr[:, col])
                    out[:, row] = acc
            return out

        raise ValueError("Filter type not supported")
