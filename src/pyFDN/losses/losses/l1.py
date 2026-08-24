import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """
    L1 Loss / Mean Absolute Error (MAE).

    Computes the absolute element-wise difference:
        loss = |input - target|

    Args:
        reduction (str, optional): 'mean' (default), 'sum', or 'none'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Invalid reduction mode '{reduction}'. Choose from ['mean', 'sum', 'none']."
            )
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Predicted tensor of arbitrary shape.
            target (torch.Tensor): Target tensor of matching shape.

        Returns:
            torch.Tensor: Computed L1 loss.
        """
        if input.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: input shape {tuple(input.shape)} "
                f"does not match target shape {tuple(target.shape)}."
            )

        loss = torch.abs(input - target)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss