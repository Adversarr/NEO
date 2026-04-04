import torch
from torch import nn


class RootRelMSELoss(nn.Module):
    def __init__(self, channel_wise: bool = False, eps: float = 1e-6) -> None:
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
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        if mass is not None:
            # Support (B, N) or (B, N, 1) mass by normalizing to (B, N, 1)
            if mass.dim() == 2:
                mass = mass.unsqueeze(-1)
            pred = pred * mass
            target = target * mass

        diff = pred - target
        if self.channel_wise:  # apply to last dimension.
            # Calculate MSE for each channel separately
            norm_channels = torch.linalg.norm(target, dim=1)  # B, C
            root_mse_per_channel = torch.linalg.norm(diff, dim=1)  # B, C
            rel_root_mse = root_mse_per_channel / (norm_channels + self.eps)
            return torch.mean(rel_root_mse)
        else:
            # Calculate Root Relative MSE for the entire tensor with safe denominator
            pred = pred.flatten(-2)
            target = target.flatten(-2)
            mse = torch.mean((pred - target) ** 2, dim=-1)  # [B]
            norm = torch.mean(target ** 2, dim=-1)          # [B]
            rel = mse / norm.clamp_min(self.eps)
            return torch.mean(torch.sqrt(rel))
