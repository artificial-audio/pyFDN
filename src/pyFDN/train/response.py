"""What the trainer sees: the FDN's impulse response.

:class:`Response` is the single object every loss is written against. It holds
the impulse response in the **time domain**; the frequency-domain views
(:attr:`~Response.spectrum`, :attr:`~Response.magnitude`) are derived from it and
cached, so several spectral losses in one objective share a single FFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Response:
    """The impulse response of an FDN, as a loss sees it.

    Attributes
    ----------
    h : torch.Tensor
        Impulse response of shape ``(n_samples, n_out, n_in)`` -- the same
        convention as :func:`pyFDN.build_to_impz`. This is the response itself:
        any anti-aliasing envelope the model was built with has already been
        removed by the shell's output layer, so ``h`` is accurate to
        ``alias_decay_db`` (see :func:`pyFDN.trainable_from_build`) and needs no
        further correction. Differentiable during training.
    fs : float
        Sample rate in Hz.
    """

    h: torch.Tensor
    fs: float

    @property
    def n_samples(self) -> int:
        return int(self.h.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.h.shape[1])

    @property
    def n_in(self) -> int:
        return int(self.h.shape[2])

    @cached_property
    def spectrum(self) -> torch.Tensor:
        """``rfft(h)`` over time -- shape ``(n_samples // 2 + 1, n_out, n_in)``.

        The DFT of the *truncated* impulse response, i.e. of ``h`` under a
        rectangular window of ``n_samples``. Computed once per response and
        shared by every loss that asks for it.
        """
        import torch

        return torch.fft.rfft(self.h, dim=0)

    @cached_property
    def magnitude(self) -> torch.Tensor:
        """``|spectrum|``, shape ``(n_samples // 2 + 1, n_out, n_in)``."""
        return self.spectrum.abs()

    def flamo_layout(self) -> torch.Tensor:
        """``h`` permuted to FLAMO's ``(batch, n_samples, n_out)`` layout.

        FLAMO's loss functions take a batched time signal whose batch axis is
        the excited input, which is exactly ``h``'s input axis moved to front.
        """
        return self.h.permute(2, 0, 1)


def impulse_excitation(
    n_in: int, nfft: int, device: Any = None, dtype: Any = None
) -> torch.Tensor:
    """The excitation that makes a model's output the full IR matrix.

    One Dirac per input channel, each on its own batch row, so a model's output
    is ``(n_in, nfft, n_out)`` -- the transfer matrix ``H[out, in]``, which
    :func:`model_response` permutes into ``(nfft, n_out, n_in)``. For a
    single-input FDN this is one impulse and one batch row.
    """
    import torch

    torch_dtype = torch.float32 if dtype is None else dtype
    x = torch.zeros((n_in, nfft, n_in), device=device, dtype=torch_dtype)
    for i in range(n_in):
        x[i, 0, i] = 1.0
    return x


def require_time_output(model: Any) -> None:
    """Raise unless ``model``'s output layer returns the impulse response.

    Every pyFDN shell does, because :func:`pyFDN.wrap_fdn_shell` builds the
    inverse-FFT layer that matches the core's ``alias_decay_db`` -- the domain
    is a property of the model, not something a caller sets afterwards. A
    ``Shell`` assembled by hand can still hand back a frequency-domain tensor,
    which every loss would silently read as a time signal; hence this check.
    """
    from flamo.processor import dsp

    layer = model.get_outputLayer()
    if not isinstance(layer, dsp.iFFT | dsp.iFFTAntiAlias):
        raise ValueError(
            f"model's output layer is {type(layer).__name__}, so it does not "
            "return an impulse response. Build the shell with "
            "pyFDN.wrap_fdn_shell (or pyFDN.build_fdn), which pairs the FFT "
            "input layer with the iFFT the core's alias_decay_db calls for."
        )


def model_response(model: Any, excitation: torch.Tensor | None = None) -> Response:
    """Run ``model`` on an impulse and wrap the result in a :class:`Response`.

    Pass ``excitation`` to reuse a tensor across steps;
    :func:`impulse_excitation` builds one.

    Gradients flow through the returned response, so this is what the trainer
    calls each step -- and, detached, what you can call to inspect a model.
    """
    require_time_output(model)
    n_in, n_out = int(model.input_channels), int(model.output_channels)
    if excitation is None:
        excitation = impulse_excitation(
            n_in, int(model.nfft), dtype=getattr(model, "dtype", None)
        )
    y = model(excitation)  # (n_in, nfft, n_out)
    if y.shape[0] != n_in or y.shape[2] != n_out:
        raise ValueError(
            f"model output {tuple(y.shape)} does not match its (n_in, nfft, "
            f"n_out) = ({n_in}, {int(model.nfft)}, {n_out})"
        )
    return Response(h=y.permute(1, 2, 0), fs=model_fs(model))


def model_fs(model: Any) -> float:
    """Sample rate read from the model's delay module."""
    from pyFDN.auxiliary.flamo_graph import flamo_model_to_nodes, flamo_nodes_flat

    for node in flamo_nodes_flat(flamo_model_to_nodes(model)):
        if node["type"] == "Leaf" and hasattr(node["module"], "fs"):
            return float(node["module"].fs)
    raise ValueError("model has no module exposing fs; check build again.")
