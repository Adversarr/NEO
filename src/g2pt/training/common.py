import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from g2pt.optim.muon import Muon, build_muon_param_groups
from g2pt.optim.lr_wsd import WarmupStableDecayLR
from g2pt.optim.lr_cosine_decay import CosineAnnealingWithWarmupLR
import torch.distributed as dist
from dataclasses import dataclass

from g2pt.neuralop.model import Transolver2Model, Transolver2SolverModel


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.04
    max_lr: float = 5.0e-4


@dataclass
class SchedulerConfig:
    name: str = "wsd"
    warmup: float = 0.1
    stable: float = 0.7
    schedule: str = "exp"
    final_lr: float = 1e-6


def create_optimizer_and_scheduler(
    module,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    total_steps: int,
):
    """
    Create optimizer and scheduler based on configuration.

    Args:
        module: The model module to optimize
        optimizer_config: Dict with optimizer configuration (name, betas, etc.)
        scheduler_config: Dict with scheduler configuration (name, warmup, stable, schedule)
        max_lr: Maximum learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps (epochs)
        world_size: Number of distributed training processes (default: 1)

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_name = optimizer_config.name.lower()
    scheduler_name = scheduler_config.name.lower()

    world_size: int = dist.get_world_size() if dist.is_initialized() else 1
    pg = None
    # Create optimizer
    if optimizer_name == "muon":
        # Split parameters: 2D+ for Muon, others for AdamW
        param_groups = build_muon_param_groups(module)
        optimizer = Muon(
            param_groups,
            lr=optimizer_config.max_lr,
            weight_decay=optimizer_config.weight_decay,
            adamw_betas=optimizer_config.betas,
        )

    elif optimizer_name == "adamw":
        # Use AdamW optimizer
        normal, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('register') or name == 'output_norm':
                print(f"Add {name} to no_decay")
                no_decay.append(param)
            else:
                normal.append(param)
        pg = [
            {"params": normal, "weight_decay": optimizer_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = AdamW(
            pg,
            lr=optimizer_config.max_lr,
            betas=optimizer_config.betas,
            weight_decay=optimizer_config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create scheduler
    if scheduler_name == "wsd":
        # Warmup-Stable-Decay scheduler
        warmup_ratio = scheduler_config.warmup
        stable_ratio = scheduler_config.stable
        decay_schedule = scheduler_config.schedule

        warmup_steps = int(total_steps * warmup_ratio)
        stable_steps = int(total_steps * stable_ratio)
        decay_steps = total_steps - warmup_steps - stable_steps

        scheduler = WarmupStableDecayLR(
            optimizer,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            warmup_start_factor=1e-3,
            warmup_end_factor=1.0,
            decay_end_factor=0.01,
            decay_sched=decay_schedule,
        )

    elif scheduler_name == "onecycle":
        # OneCycleLR scheduler
        pct_start = scheduler_config.warmup
        if pg is not None:
            max_lr = [p['lr'] if 'lr' in p else optimizer_config.max_lr for p in pg]
        else:
            max_lr = optimizer_config.max_lr
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            cycle_momentum=optimizer_name != "muon",
        )
    elif scheduler_name == "exp":
        ratio = scheduler_config.final_lr / optimizer_config.max_lr
        scheduler = ExponentialLR(
            optimizer,
            gamma=ratio ** (1 / total_steps),
        )
    elif scheduler_name == "cosine":
        # Cosine annealing with warmup
        warmup_steps = getattr(scheduler_config, "warmup_steps", 0)
        # If warmup_steps is not provided but warmup ratio is
        if warmup_steps == 0 and scheduler_config.warmup > 0:
            warmup_steps = int(total_steps * scheduler_config.warmup)
        
        # Use provided total_steps or fallback to argument
        sched_total_steps = getattr(scheduler_config, "total_steps", 0)
        if sched_total_steps <= 0:
            sched_total_steps = total_steps
        
        # Calculate final_lr_factor relative to max_lr
        final_lr_factor = scheduler_config.final_lr / optimizer_config.max_lr
        
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=sched_total_steps,
            final_lr_factor=final_lr_factor,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return optimizer, scheduler

def load_ckpt(ckpt_path: str, strict: bool = True):
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)

def extract_state_dict(ckpt_path: str):
    obj = load_ckpt(ckpt_path)
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

def filter_state_dict_by_prefix(state_dict: dict, prefix: str):
    out = {}
    pl = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[pl:]] = v
    return out

def load_partial_state_dict_strict(module, ckpt_path: str, *, key_prefix: str, strict=True):
    sd = extract_state_dict(ckpt_path)
    sub_sd = filter_state_dict_by_prefix(sd, key_prefix)
    missing, unexpected = module.load_state_dict(sub_sd, strict=False)
    if missing or unexpected:
        if strict:
            raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        else:
            print(f"missing={missing}, unexpected={unexpected}")
    return list(sub_sd.keys())

def load_params_state_dict_strict(modules_map: dict[str, object], ckpt_path: str):
    sd = extract_state_dict(ckpt_path)
    for prefix, module in modules_map.items():
        sub_sd = filter_state_dict_by_prefix(sd, prefix)
        missing, unexpected = module.load_state_dict(sub_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict load failed for prefix {prefix}: missing={missing}, unexpected={unexpected}")
    return True


def accuracy(y_pred, y_true):
    """
    Compute accuracy for segmentation task.

    Args:
        y_pred: Predicted logits (B, N, C)
        y_true: Ground truth labels (B, N)

    Returns:
        float: Accuracy score
    """
    y_pred = y_pred.argmax(dim=-1)
    correct = (y_pred == y_true).float()
    return correct.mean().item()
