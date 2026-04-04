from typing import Tuple, Type
from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
import numpy as np

class BottleneckAnalyzer(Callback):
    """
    Analyzes gradients and parameters directly before the optimizer step (no hooks needed),
    used to diagnose bottleneck layers.

    Monitored metrics:
    1. Grad Norm: gradient magnitude, reflecting the learning pressure of each layer.
    2. Stable Rank (SVD): rank of the gradient matrix, reflecting the richness of the
       learning signal (capacity utilization).
    """
    
    def __init__(
        self, 
        log_every_n_steps: int = 500, 
        analyze_rank: bool = True,
        layer_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d, nn.Conv1d)
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.analyze_rank = analyze_rank
        self.layer_types = layer_types

    def _compute_stable_rank(self, tensor: torch.Tensor) -> float:
        """
        Compute the stable rank: sum(sigma^2) / max(sigma)^2.
        More continuous than the numerical rank; reflects the effective dimensionality of the matrix.
        """
        # Flatten: for Conv2d (Out, In, K, K) -> (Out, In*K*K)
        if tensor.dim() > 2:
            tensor = tensor.flatten(1)
        
        if tensor.dim() < 2 or tensor.numel() == 0:
            return 0.0

        # Cast to float32 to avoid half-precision overflow; detach to avoid memory leaks
        tensor = tensor.detach().float()

        try:
            # Compute singular values only (faster than full SVD)
            s = torch.linalg.svdvals(tensor)
            max_s = s[0] if len(s) > 0 else 0
            if max_s == 0:
                return 0.0
            
            # Stable Rank computation
            rank = (s ** 2).sum() / (max_s ** 2)
            return rank.item()
        except Exception:
            return 0.0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Rate control: SVD is expensive; avoid running every step
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        metrics = {}
        
        # Iterate over all submodules
        for name, module in pl_module.named_modules():
            if isinstance(module, self.layer_types):
                # Only analyze layers that have parameters with gradients
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                if module.weight.grad is None:
                    continue

                grad = module.weight.grad
                weight = module.weight
                
                # 1. Basic metric: gradient norm
                # A large norm means the layer is undergoing significant updates
                g_norm = torch.linalg.norm(grad).item()
                metrics[f"analysis/{name}/grad_norm"] = g_norm

                # 2. Advanced metric: stable rank of the gradient matrix (Capacity Utilization)
                if self.analyze_rank:
                    # Only compute rank for layers with enough parameters; skip tiny layers
                    if weight.numel() > 512:
                        s_rank = self._compute_stable_rank(grad)

                        # Rank ratio: Stable Rank / physical minimum dimension
                        # Physical dim: Linear(Out, In) -> min(Out, In)
                        # Conv(Out, In, K, K) -> min(Out, In*K*K)
                        min_dim = min(grad.shape[0], np.prod(grad.shape[1:]))
                        rank_ratio = s_rank / (min_dim + 1e-6)
                        
                        metrics[f"analysis/{name}/grad_rank"] = s_rank
                        metrics[f"analysis/{name}/rank_ratio"] = rank_ratio

        # Log all metrics using pl_module.log_dict
        # on_step=True logs per step automatically; prog_bar=False keeps the progress bar clean
        pl_module.log_dict(metrics, prog_bar=False)
