import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Initialize the MSELoss module."""
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) loss between two tensors.
        Args:
            pred (torch.Tensor): First tensor.
            target (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: MSE loss.
        """
        if mass is not None:
            pred = pred * mass
            target = target * mass
        return nn.functional.mse_loss(pred, target, reduction="mean")
