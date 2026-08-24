import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) / L2 Loss.

    Computes the squared error between predictions and targets:
        loss = (input - target) ** 2

    Args:
        reduction (str, optional): Reduction method to apply to the output.
            Options: 'mean' (default), 'sum', 'none'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Invalid reduction mode '{reduction}'. Choose from ['mean', 'sum', 'none']."
            )
        self.reduction = reduction

    def forward(
            self, 
            input: torch.Tensor, 
            target: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Predicted tensor of arbitrary shape.
            target (torch.Tensor): Ground-truth target tensor of matching shape.

        Returns:
            torch.Tensor: Reduced scalar loss by default ('mean'), or tensor based on reduction mode.
        """
        if input.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: input shape {tuple(input.shape)} "
                f"does not match target shape {tuple(target.shape)}."
            )

        loss = (input - target) ** 2

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss # none