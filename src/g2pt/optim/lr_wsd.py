# warmup-stable-decay

import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step


class WarmupStableDecayLR(LRScheduler):
    """
    Learning rate scheduler with three phases:
    1. Warmup: LR increases linearly from base_lr * warmup_start_factor to base_lr * warmup_end_factor.
    2. Stable: LR remains at base_lr * warmup_end_factor.
    3. Decay: LR decreases from base_lr * warmup_end_factor to base_lr * decay_end_factor.
    After total_steps, the learning rate stays at base_lr * decay_end_factor.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for warmup (non-negative). Default: 0 (no warmup).
        stable_steps (int): Number of steps for stable phase (non-negative). Default: 0 (no stable).
        decay_steps (int): Number of steps for decay (non-negative). Default: 0 (no decay).
        warmup_start_factor (float): Starting LR factor for warmup, relative to base_lr.
            Must satisfy 0 <= warmup_start_factor <= warmup_end_factor. Default: 0.0.
        warmup_end_factor (float): Ending LR factor for warmup (and level for stable). Default: 1.0.
        decay_end_factor (float): Final LR factor after decay. Must satisfy
            0 <= decay_end_factor <= warmup_end_factor. Default: 0.0.
        last_epoch (int): The index of last epoch. Default: -1.
        decay_sched (str): Decay schedule, 'linear' or 'cosine'. Default: 'cosine'.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        stable_steps: int = 0,
        decay_steps: int = 0,
        warmup_start_factor: float = 0.0,
        warmup_end_factor: float = 1.0,
        decay_end_factor: float = 0.0,
        last_epoch: int = -1,
        decay_sched: str = "cosine",  # options: 'linear', 'cosine'
    ):
        # Parameter validation
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an instance of Optimizer, got {type(optimizer).__name__}.")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be non-negative, got {stable_steps}.")
        if decay_steps < 0:
            raise ValueError(f"decay_steps must be non-negative, got {decay_steps}.")
        if not (0.0 <= warmup_start_factor <= warmup_end_factor):
            raise ValueError(
                f"warmup_start_factor ({warmup_start_factor}) must be in [0.0, warmup_end_factor ({warmup_end_factor})]."
            )
        if not (0.0 <= decay_end_factor <= warmup_end_factor):
            raise ValueError(
                f"decay_end_factor ({decay_end_factor}) must be in [0.0, warmup_end_factor ({warmup_end_factor})]."
            )
        if decay_sched not in ["linear", "cosine", "exp"]:
            raise ValueError(f"decay_sched must be 'linear', 'cosine' or 'exp', got {decay_sched}.")

        self.decay_sched = decay_sched
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.warmup_start_factor = warmup_start_factor
        self.warmup_end_factor = warmup_end_factor
        self.decay_end_factor = decay_end_factor
        self.total_steps = warmup_steps + stable_steps + decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rates (called at each step)."""
        _warn_get_lr_called_within_step(self)

        current_step = self.last_epoch
        lrs = []

        for base_lr in self.base_lrs:
            if current_step < self.warmup_steps:
                # Warmup phase: linear increase
                if self.warmup_steps == 0:
                    factor = self.warmup_end_factor
                else:
                    progress = current_step / self.warmup_steps
                    factor = self.warmup_start_factor + (self.warmup_end_factor - self.warmup_start_factor) * progress

            elif current_step < self.warmup_steps + self.stable_steps:
                # Stable phase: constant LR
                factor = self.warmup_end_factor

            elif current_step < self.total_steps:
                # Decay phase
                if self.decay_steps == 0:
                    factor = self.warmup_end_factor
                else:
                    decay_step = current_step - (self.warmup_steps + self.stable_steps)
                    progress = decay_step / self.decay_steps
                    if self.decay_sched == "linear":
                        # Linear decay
                        factor = self.warmup_end_factor - (self.warmup_end_factor - self.decay_end_factor) * progress
                    elif self.decay_sched == "cosine":
                        # Cosine decay
                        factor = (
                            self.decay_end_factor
                            + 0.5 * (self.warmup_end_factor - self.decay_end_factor)
                            * (1 + math.cos(math.pi * progress))
                        )
                    elif self.decay_sched == "exp":
                        # Exponential decay
                        factor = self.warmup_end_factor * (self.decay_end_factor / self.warmup_end_factor) ** progress
            else:
                # After total_steps: keep final factor
                factor = self.decay_end_factor

            lrs.append(base_lr * factor)

        return lrs

    def _get_closed_form_lr(self):
        """Optional fast closed-form LR computation."""
        current_step = self.last_epoch
        return [
            base_lr * self._get_factor(current_step)
            for base_lr in self.base_lrs
        ]

    def _get_factor(self, current_step):
        """Helper to compute the LR factor at a given step."""
        if current_step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.warmup_end_factor
            progress = current_step / self.warmup_steps
            return self.warmup_start_factor + (self.warmup_end_factor - self.warmup_start_factor) * progress

        elif current_step < self.warmup_steps + self.stable_steps:
            return self.warmup_end_factor

        elif current_step < self.total_steps:
            if self.decay_steps == 0:
                return self.warmup_end_factor
            decay_step = current_step - (self.warmup_steps + self.stable_steps)
            progress = decay_step / self.decay_steps
            return self.warmup_end_factor - (self.warmup_end_factor - self.decay_end_factor) * progress

        else:
            return self.decay_end_factor
