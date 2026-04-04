import torch
from torch import nn


class RelL1Loss(nn.Module):
    def __init__(self, channel_wise: bool = False, eps: float = 1e-6) -> None:
        """Initialize the Relative L1 Loss module.

        Args:
            channel_wise (bool): If True, compute relative L1 loss per channel.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.channel_wise = channel_wise
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute the Relative Mean Absolute Error (L1) loss between two tensors.

        Args:
            pred (torch.Tensor): Prediction tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Relative L1 loss.
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        if mass is not None:
            pred = pred * mass
            target = target * mass

        if self.channel_wise:  # apply to last dimension
            # Calculate L1 for each channel separately
            norm_channels = torch.mean(torch.abs(target), dim=1)  # B, C
            l1_per_channel = torch.mean(torch.abs(pred - target), dim=1)  # B, C
            return torch.mean(l1_per_channel / (norm_channels + self.eps))
        else:
            # Calculate L1 for the entire tensor
            pred = pred.flatten(-2)
            target = target.flatten(-2)
            l1 = torch.mean(torch.abs(pred - target), dim=-1)
            norm = torch.mean(torch.abs(target), dim=-1)
            return torch.mean(l1 / (norm + self.eps))
