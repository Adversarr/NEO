import torch
from torch import nn


class L1Loss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Initialize the L1Loss module."""
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute the Mean Absolute Error (L1) loss between two tensors.
        Args:
            pred (torch.Tensor): First tensor.
            targ (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: L1 loss.
        """
        if mass is not None:
            pred = pred * mass
            target = target * mass

        return nn.functional.l1_loss(pred, target, reduction="mean")
