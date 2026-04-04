import math
from typing import List

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step


class OneCycleWarmupDecayLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        stable_steps: int = 0,
        decay_steps: int = 0,
        warmup_start_factor: float = 0.0,
        warmup_end_factor: float = 1.0,
        decay_end_factor: float = 0.0,
        schedule: str = "CosWithWarmup",
        last_epoch: int = -1,
        inv_sqrt_t0: float = 1.0,
    ):
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

        schedule_norm = schedule.lower()
        alias = {
            "coswithwarmup": "cos",
            "linearwithwarmup": "linear",
            "invsqrtwithwarmup": "inv_sqrt",
            "coslinearenvelope": "cos_linear_envelope",
            "boltonwarmupscheduler": "bolt_on",
        }
        schedule_norm = alias.get(schedule_norm, schedule_norm)
        if schedule_norm not in {"cos", "linear", "inv_sqrt", "cos_linear_envelope", "bolt_on"}:
            raise ValueError(
                "schedule must be one of ['CosWithWarmup','LinearWithWarmup','InvSqrtWithWarmup','CosLinearEnvelope','BoltOnWarmupScheduler']."
            )
        if schedule_norm == "inv_sqrt" and inv_sqrt_t0 <= 0.0:
            raise ValueError(f"inv_sqrt_t0 must be positive, got {inv_sqrt_t0}.")

        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.warmup_start_factor = warmup_start_factor
        self.warmup_end_factor = warmup_end_factor
        self.decay_end_factor = decay_end_factor
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.schedule = schedule_norm
        self.inv_sqrt_t0 = inv_sqrt_t0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        _warn_get_lr_called_within_step(self)
        step = self.last_epoch
        return [base_lr * self._factor(step) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> List[float]:
        step = self.last_epoch
        return [base_lr * self._factor(step) for base_lr in self.base_lrs]

    def _factor(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.warmup_end_factor
            p = step / self.warmup_steps
            return self.warmup_start_factor + (self.warmup_end_factor - self.warmup_start_factor) * p

        if step < self.warmup_steps + self.stable_steps:
            return self.warmup_end_factor

        if step < self.total_steps:
            if self.decay_steps == 0:
                return self.warmup_end_factor
            dstep = step - (self.warmup_steps + self.stable_steps)
            p = dstep / self.decay_steps
            if self.schedule == "linear":
                return self.warmup_end_factor - (self.warmup_end_factor - self.decay_end_factor) * p
            if self.schedule == "cos":
                return self.decay_end_factor + 0.5 * (self.warmup_end_factor - self.decay_end_factor) * (1 + math.cos(math.pi * p))
            if self.schedule == "inv_sqrt":
                num = self.inv_sqrt_t0 + self.warmup_steps
                den = self.inv_sqrt_t0 + step
                val = self.warmup_end_factor * math.sqrt(num / den)
                return max(self.decay_end_factor, val)
            if self.schedule == "cos_linear_envelope":
                env = 1.0 - p
                amp = (self.warmup_end_factor - self.decay_end_factor) * env
                return self.decay_end_factor + 0.5 * amp * (1 + math.cos(math.pi * p))
            if self.schedule == "bolt_on":
                return self.warmup_end_factor
        else:
            if self.schedule == "bolt_on":
                return self.warmup_end_factor
            return self.decay_end_factor