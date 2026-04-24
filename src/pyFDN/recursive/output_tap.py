"""Output summation stage."""

from __future__ import annotations

import torch

from .stage import Stage


class OutputTap(Stage):
    """
    Produces final output by summing weighted line signals.

    This stage:
    - Reads ctx["lines"]
    - Optionally reads ctx["x"] for direct path
    - Applies output matrix C (and optional direct matrix D)
    - Writes ctx["y"] as final output
    - Is purely feedforward (no state)

    Operation: ctx["y"] = ctx["lines"] @ C.T [+ ctx["x"] @ D.T]
    """

    def __init__(
        self,
        output_matrix: torch.Tensor | None = None,
        direct_matrix: torch.Tensor | None = None,
        num_lines: int = 4,
        num_outputs: int = 1,
    ):
        """
        Initialize output summation stage.

        Args:
            output_matrix: Output gain matrix C of shape [N_out, N]
                         If None, creates averaging matrix (all lines contribute equally)
            direct_matrix: Direct-path matrix D of shape [N_out, N_in]
                         If None, no direct path is used
            num_lines: Number of feedback lines (N), used if output_matrix is None
            num_outputs: Number of output channels (N_out), used if matrices are None
        """
        super().__init__(state_keys=set())  # Stateless

        if output_matrix is None:
            # Default: average all lines equally
            self.C = torch.ones(num_outputs, num_lines, dtype=torch.float32) / num_lines
            self.num_outputs = num_outputs
            self.num_lines = num_lines
        else:
            self.C = output_matrix.float()
            self.num_outputs, self.num_lines = self.C.shape

        if self.C.dim() != 2:
            raise ValueError(
                f"Output matrix must be 2D [N_out, N], got shape {self.C.shape}"
            )

        # Direct path is optional
        if direct_matrix is None:
            self.D = None
        else:
            self.D = direct_matrix.float()
            self.num_inputs = self.D.shape[1]
            if self.D.shape[0] != self.num_outputs:
                raise ValueError(
                    f"Direct matrix must have {self.num_outputs} output channels, got shape {self.D.shape}"
                )
            if self.D.shape[1] != self.num_inputs:
                raise ValueError(
                    f"Direct matrix must have {self.num_inputs} input channels, got shape {self.D.shape}"
                )
            if self.D.dim() != 2:
                raise ValueError(
                    f"Direct matrix must be 2D [N_out, N_in], got shape {self.D.shape}"
                )

    def init_state(
        self,
        batch_size: int,
        block_size: int | torch.device | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """No state needed - purely feedforward."""
        if device is None and isinstance(block_size, torch.device):
            device = block_size
        if device is None:
            device = torch.device("cpu")
        # Move matrices to device
        self.C = self.C.to(device)
        if self.D is not None:
            self.D = self.D.to(device)
        return {}

    def step_block(
        self,
        lines: torch.Tensor | None,
        state_t: dict[str, torch.Tensor],
        next_state: dict[str, torch.Tensor],
        block_size: int,
        x_block: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute output as weighted sum of lines (and optional direct path).

        Returns the unchanged `lines` tensor together with the output block.
        """
        if lines is None:
            raise RuntimeError("OutputTap requires `lines` to be set")

        # Apply output matrix using einsum: [B, N, T] @ [N, N_out] -> [B, N_out, T]
        # einsum('bnt,no->bot', lines, self.C.T) computes lines @ C.T efficiently without transposing
        y = torch.einsum("bnt,no->bot", lines, self.C.T)  # [B, N_out, T]

        # Add direct path if present
        if self.D is not None:
            if x_block is None:
                raise RuntimeError(
                    "OutputTap with direct path requires external input `x_block`"
                )
            x = x_block  # [B, N_in, T]
            # Apply direct matrix using einsum: [B, N_in, T] @ [N_in, N_out] -> [B, N_out, T]
            direct = torch.einsum("bnt,no->bot", x, self.D.T)  # [B, N_out, T]
            y = y + direct

        return lines, y
