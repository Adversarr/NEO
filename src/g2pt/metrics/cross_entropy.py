import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLossForSegmentation(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mass: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Cross Entropy loss between two tensors.
        Args:
            pred (torch.Tensor): First tensor.
            target (torch.Tensor): Second tensor.
            mass (torch.Tensor | None, optional): Mass tensor. Defaults to None.
        Returns:
            torch.Tensor: Cross Entropy loss.
        """
        eps = 1e-12
        B, N, C = pred.shape
        assert target.shape == (B, N)

        logits = pred.reshape(-1, C)
        labels = target.reshape(-1)
        # segmentation task requires lower label smoothing
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        ce_flat = F.cross_entropy(logits, labels, reduction="none", weight=class_weights)
        ce = ce_flat.view(B, N)
        if mass is not None:
            w = mass.view(B, N)
            batch_loss = (ce * w).sum(dim=1) / (w.sum(dim=1) + eps)
            loss = batch_loss.mean()
        else:
            loss = ce.mean()
        return loss


class CrossEntropyLossForClassification(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the cross entropy loss between two one-hot tensors.
        Args:
            pred (torch.Tensor): First tensor.
            target (torch.Tensor): Second tensor.
        Returns:
            torch.Tensor: Cross entropy loss.
        """
        assert pred.ndim == 2
        if class_weights is not None:
            class_weights = class_weights.to(pred.device)
        return F.cross_entropy(pred, target, weight=class_weights)