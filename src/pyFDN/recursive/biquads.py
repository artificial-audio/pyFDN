"""Parallel biquad filter bank stage."""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .stage import Stage


def _biquad_section_vectorized(
    y: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    a1n: torch.Tensor,
    a2n: torch.Tensor,
    b0n: torch.Tensor,
    b1n: torch.Tensor,
    b2n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized single biquad section (DF2T) over the time dimension.

    Recurrence: s_{n+1} = A @ s_n + c * x[n] with s_n = [z1_n, z2_n]^T.
    Output: y[n] = b0n*x[n] + z1_n.
    Uses matrix powers via eigendecomposition and conv1d so there is no Python
    loop over time. Falls back to per-sample step when diagonalization fails.

    Args:
        y: Input block [B, N, T]
        z1, z2: Initial state [B, N]
        a1n, a2n, b0n, b1n, b2n: Normalized coefficients [N]

    Returns:
        output: [B, N, T]
        z1_final, z2_final: [B, N]
    """
    B, N, T = y.shape
    device = y.device
    dtype = y.dtype

    # DF2T state recurrence: A = [[-a1n, 1], [-a2n, 0]], c = [b1n - a1n*b0n, b2n - a2n*b0n]
    # A and c are per-line; build (N, 2, 2) and (N, 2)
    A = torch.stack(
        [
            torch.stack([-a1n, torch.ones(N, device=device, dtype=dtype)], dim=1),
            torch.stack([-a2n, torch.zeros(N, device=device, dtype=dtype)], dim=1),
        ],
        dim=1,
    )
    c = torch.stack(
        [b1n - a1n * b0n, b2n - a2n * b0n],
        dim=1,
    )
    s0 = torch.stack([z1, z2], dim=-1)

    # Matrix powers via eigendecomposition: A = V D V^{-1} => A^j = V D^j V^{-1}
    # eig returns complex; for real coefficients and input the result is real
    try:
        eigenvalues, V = torch.linalg.eig(A)
    except Exception:
        return _biquad_section_sequential(y, z1, z2, a1n, a2n, b0n, b1n, b2n)

    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    A = A.to(cdtype)
    c = c.to(cdtype)
    s0 = s0.to(cdtype)
    y_c = y.to(cdtype)
    eigenvalues = eigenvalues.to(cdtype)
    V = V.to(cdtype)

    V_inv = torch.linalg.inv(V)
    D = eigenvalues

    # D^j for j = 1..T: shape (N, 2, T)
    j_arange = torch.arange(1, T + 1, device=device, dtype=D.real.dtype)
    D_pow = torch.pow(D.unsqueeze(-1), j_arange.unsqueeze(0).unsqueeze(0))

    # A^j = V @ diag(D^j) @ V_inv => (N, 2, 2, T). diag_D is (N, T, 2, 2)
    diag_D = torch.diag_embed(D_pow.permute(0, 2, 1))
    # einsum "nitl" gives (N, 2, T, 2); permute to (N, 2, 2, T) so A_powers[n,:,:,t] = A^{t+1}
    A_powers = torch.einsum("nij,ntjk,nkl->nitl", V, diag_D, V_inv).permute(0, 1, 3, 2).contiguous()

    # Homogeneous part: s_{t+1} = A^{t+1} s_0 => (B, N, T, 2)
    A_powers_nt = A_powers.permute(0, 3, 1, 2)
    hom = torch.einsum("ntij,bnj->bnti", A_powers_nt, s0)

    # Convolution part: sum_{k=0}^{t} A^{t-k} c x[k]. Impulse response h_j = A^j c, j=0..T-1
    # A^0 = I, A^j for j>=1 from A_powers (A_powers[:,:,:,j-1] = A^j). So h_0 = c, h_j = A_powers[:,:,:,j-1] @ c.
    # Build h_j (N, 2, T): h_j[:,:,0] = c, h_j[:,:,j] = A_powers[:,:,:,j-1] @ c for j=1..T-1
    h_j = torch.empty(N, 2, T, device=device, dtype=cdtype)
    h_j[:, :, 0] = c
    if T > 1:
        # A_powers is (N, 2, 2, T) with A_powers[:,:,:,t] = A^{t+1}. So A^j at index j-1.
        h_j[:, :, 1:] = torch.einsum("nijt,nj->nit", A_powers[:, :, :, 0 : T - 1], c)
    # Convolution: out[t] = sum_{k=0}^{t} x[k] h[t-k] with h[j] = A^j c.
    # Pad x on the left with T-1 zeros; kernel = flip(h); then conv1d gives out[t] = sum_j x[j] h[t-j].
    h_n0 = h_j[:, 0, :]  # (N, T), h_n0[n,j] = A^j c for state 0
    h_n1 = h_j[:, 1, :]
    x_padded = F.pad(y_c.real.reshape(1, B * N, T), (T - 1, 0), value=0.0)  # (1, B*N, 2T-1)
    kernel0 = h_n0.unsqueeze(0).expand(B, N, T).reshape(B * N, 1, T).flip(-1).real
    kernel1 = h_n1.unsqueeze(0).expand(B, N, T).reshape(B * N, 1, T).flip(-1).real
    conv0 = F.conv1d(x_padded, kernel0, groups=B * N)[0, :, :T]  # (B*N, T)
    conv1 = F.conv1d(x_padded, kernel1, groups=B * N)[0, :, :T]
    conv_out = torch.stack([conv0, conv1], dim=-1).reshape(B, N, T, 2)

    s_all = (hom + conv_out).real  # s_all[:,:,t,:] = s_{t+1}
    # Output: y[t] = b0n*x[t] + s_t[0], so we need s_t for t=0..T-1 (s_0 given, s_1..s_{T-1} from s_all)
    s0_z1 = (s0.real[:, :, 0:1] if s0.is_complex() else s0[:, :, 0:1])  # (B, N, 1)
    s_for_y = torch.cat([s0_z1, s_all[:, :, :-1, 0]], dim=2)  # (B, N, T)
    y_out = (b0n.unsqueeze(0).unsqueeze(-1) * y + s_for_y).to(dtype)

    z1_final = s_all[:, :, -1, 0]
    z2_final = s_all[:, :, -1, 1]
    return y_out, z1_final, z2_final


def _biquad_section_sequential(
    y: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    a1n: torch.Tensor,
    a2n: torch.Tensor,
    b0n: torch.Tensor,
    b1n: torch.Tensor,
    b2n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample loop fallback for one biquad section."""
    B, N, T = y.shape
    output = torch.zeros_like(y)
    for t in range(T):
        x_n = y[:, :, t]
        y_n = b0n.unsqueeze(0) * x_n + z1
        z1 = b1n.unsqueeze(0) * x_n - a1n.unsqueeze(0) * y_n + z2
        z2 = b2n.unsqueeze(0) * x_n - a2n.unsqueeze(0) * y_n
        output[:, :, t] = y_n
    return output, z1, z2


class Biquads(Stage):
    """
    Parallel bank of biquad IIR filters applied to feedback lines.

    This stage:
    - Operates on the `lines` tensor (feedback-line signals for the current block)
    - Applies biquad filtering to each line independently
    - Maintains IIR filter state across blocks
    - Returns the filtered lines for the next stage

    Filter structure: Transposed Direct Form II biquad
        a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

    State per line: [z1, z2] (two delay elements for DF2T)
    """

    def __init__(
        self,
        num_lines: int = 4,
        biquad_coeffs: Optional[torch.Tensor] = None,
        *,
        use_vectorized: bool = False,
    ):
        """
        Initialize parallel biquad filter bank.

        Args:
            num_lines: Number of filter lines (N)
            biquad_coeffs: Filter coefficients of shape [N, 6] or [N, num_sections, 6]
                          where each row is [a0, a1, a2, b0, b1, b2]
                          If None, creates simple one-pole lowpass filters
            use_vectorized: If True, use block vectorized path (eig + conv, no Python
                            loop over time). If False (default), use per-sample
                            sequential loop. Both are numerically equivalent.
        """
        super().__init__(state_keys={"biquad_state"})
        self.num_lines = num_lines
        self.use_vectorized = use_vectorized
        
        # Initialize filter coefficients
        if biquad_coeffs is None:
            # Default: simple one-pole lowpass (y[n] = 0.9*y[n-1] + 0.1*x[n])
            # As biquad: a0=1.0, a1=-0.9, a2=0, b0=0.1, b1=0, b2=0
            self.coeffs = torch.tensor(
                [[1.0, -0.9, 0.0, 0.1, 0.0, 0.0]],
                dtype=torch.float32
            ).repeat(num_lines, 1)  # [N, 6]
            # Add section dimension to match expected 3D shape [N, num_sections, 6]
            self.coeffs = self.coeffs.unsqueeze(1)  # [N, 1, 6]
            self.num_sections = 1
        else:
            self.coeffs = biquad_coeffs.float()
            if self.coeffs.dim() == 2:
                # [N, 6] -> add section dimension
                self.coeffs = self.coeffs.unsqueeze(1)  # [N, 1, 6]
            if self.coeffs.shape[-1] != 6:
                raise ValueError(
                    f"Biquad coefficients must have 6 values [a0, a1, a2, b0, b1, b2], "
                    f"got {self.coeffs.shape[-1]} values"
                )
            self.num_sections = self.coeffs.shape[1]
    
    def init_state(self, batch_size: int, block_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize biquad filter states.
        
        State shape: [B, N, num_sections, 2] for DF2T states [z1, z2]
        """
        # Move coefficients to device
        self.coeffs = self.coeffs.to(device)
        
        return {
            "biquad_state": torch.zeros(
                batch_size, self.num_lines, self.num_sections, 2,
                device=device, dtype=torch.float32
            )
        }
    
    def step_block(
        self,
        lines: Optional[torch.Tensor],
        state_t: Dict[str, torch.Tensor],
        next_state: Dict[str, torch.Tensor],
        block_size: int,
        x_block: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply biquad filtering to the feedback-line tensor.

        Reads `lines` [B, N, T], updates biquad state in `next_state`, and
        returns the filtered lines for the next stage.
        """
        if lines is None:
            raise RuntimeError("Biquads requires `lines` to be set")

        x = lines  # [B, N, T]
        filter_state = state_t["biquad_state"].clone()  # [B, N, num_sections, 2]
        
        B, N, T = x.shape
        
        # Process each section (cascaded biquads).
        # use_vectorized: block path (eig + conv). Otherwise: per-sample loop.
        y = x.clone()

        for section_idx in range(self.num_sections):
            # Get coefficients for this section: [a0, a1, a2, b0, b1, b2]
            a0, a1, a2, b0, b1, b2 = self.coeffs[:, section_idx].unbind(dim=1)  # Each: [N]

            # Get DF2T state for this section: [B, N, 2] -> [z1, z2]
            state = filter_state[:, :, section_idx, :]
            z1 = state[:, :, 0]  # [B, N]
            z2 = state[:, :, 1]  # [B, N]

            # Normalize coefficients by a0 for stable DF2T update
            inv_a0 = 1.0 / a0  # [N]
            b0n = b0 * inv_a0
            b1n = b1 * inv_a0
            b2n = b2 * inv_a0
            a1n = a1 * inv_a0
            a2n = a2 * inv_a0

            if self.use_vectorized:
                output, z1_next, z2_next = _biquad_section_vectorized(
                    y, z1, z2, a1n, a2n, b0n, b1n, b2n
                )
            else:
                output, z1_next, z2_next = _biquad_section_sequential(
                    y, z1, z2, a1n, a2n, b0n, b1n, b2n
                )

            filter_state[:, :, section_idx, 0] = z1_next
            filter_state[:, :, section_idx, 1] = z2_next
            y = output
        
        # Save updated state
        next_state["biquad_state"] = filter_state

        return y, None
