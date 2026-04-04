import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step

class CosineAnnealingWithWarmupLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.
    
    LR increases linearly from base_lr * warmup_start_factor to base_lr during warmup_steps,
    then decreases from base_lr to base_lr * final_lr_factor using cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for warmup.
        total_steps (int): Total number of steps.
        warmup_start_factor (float): Starting LR factor for warmup. Default: 0.0.
        final_lr_factor (float): Final LR factor after total_steps. Default: 0.0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        total_steps: int = 0,
        warmup_start_factor: float = 0.0,
        final_lr_factor: float = 0.0,
        last_epoch: int = -1,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an instance of Optimizer, got {type(optimizer).__name__}.")
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_factor = warmup_start_factor
        self.final_lr_factor = final_lr_factor
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)
        
        current_step = self.last_epoch
        lrs = []
        
        for base_lr in self.base_lrs:
            if current_step < self.warmup_steps:
                # Linear warmup
                if self.warmup_steps == 0:
                    factor = 1.0
                else:
                    progress = current_step / self.warmup_steps
                    factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            elif current_step >= self.total_steps:
                # Post total_steps: final_lr
                factor = self.final_lr_factor
            else:
                # Cosine annealing
                progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cosine_out = 0.5 * (1 + math.cos(math.pi * progress))
                factor = self.final_lr_factor + (1.0 - self.final_lr_factor) * cosine_out
                
            lrs.append(base_lr * factor)
            
        return lrs
