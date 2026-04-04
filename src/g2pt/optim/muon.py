"""
Muon optimizer with an integrated AdamW backend in a single class.

Muon is an optimizer that uses orthogonalization via Newton-Schulz iterations to
provide more stable training for large language models. This implementation combines
Muon for 2D weight matrices with AdamW for other parameters in a unified optimizer.

Basic Usage:
    # Split parameters into Muon and AdamW groups
    param_groups = build_muon_param_groups(model)

    # Create optimizer with default settings
    optimizer = Muon(param_groups)

Advanced Usage:
    # Manually specify parameter groups with custom settings
    optimizer = Muon([
        {
            'params': model.linear_layers,
            'use_muon': True,  # Use Muon for 2D weight matrices
            'lr': 3e-4,
            'weight_decay': 0.1,
            'momentum': 0.95,
            'ns_steps': 5,
            'muon_scaling': 'moonlight'
        },
        {
            'params': model.embeddings + model.biases,
            'use_muon': False,  # Use AdamW for other parameters
            'lr': 3e-4,
            'weight_decay': 0.1,
            'adamw_betas': (0.9, 0.95),
            'adamw_eps': 1e-8
        }
    ])

Key Features:
    - Orthogonal gradient updates via Newton-Schulz iterations
    - Compatible with distributed training (DDP/FSDP)
    - Two scaling modes: "moonlight" (recommended) and "original"
    - Automatic parameter grouping via build_muon_param_groups()

Note: Muon is designed specifically for 2D weight matrices in hidden layers.
Do not apply it to embeddings, registers, or normalization parameters.

References:
    - https://github.com/MoonshotAI/Moonlight
    - https://github.com/KellerJordan/Muon/
"""

import math
from typing import List, Dict, Any, Union
import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """
    Newton–Schulz iteration to orthogonalize a 2D matrix (zeroth power).
    Quintic polynomial iteration (a, b, c) chosen to maximize slope at zero.

    Args:
        G: 2D tensor (gradient/momentum matrix)
        steps: number of NS iterations (5–6 is typically enough)
        eps: small value for numerical stability

    Returns:
        Orthogonalized update (same shape as G). The computation runs in bfloat16.

    Notes:
        - This implementation follows Keller Jordan's public Muon code.
        - Normalization uses Frobenius norm for stability, matching the reference.
    """
    assert G.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # Ensure spectral norm is at most 1
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure ||X||_F <= 1 (reference uses Frobenius norm)
    # X = X / (X.norm() + eps)
    X.div_(X.norm().clamp_(min=eps))
    for _ in range(steps):
        A = X @ X.T
        # B = b * A + c * (A @ A)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        # X = a * X + B @ X
        X = torch.addmm(X, B, X, beta=a, alpha=1)

    # Restore original orientation
    if G.size(0) > G.size(1):
        X = X.T

    return X


class Muon(Optimizer):
    """
    Muon optimizer with an integrated AdamW backend in a single class.

    Two param-group modes:
      - use_muon=True  : Muon applied to 2D hidden-layer matrices (e.g., Linear weights)
      - use_muon=False : AdamW applied to embeddings, biases, norms, non-2D tensors

    Scaling modes:
      - muon_scaling="moonlight" (recommended): 0.2 * sqrt(max(A, B))
          Aligns update RMS with AdamW, so you can reuse existing AdamW LR/WD (e.g., LR=3e-4).
      - muon_scaling="original": sqrt(max(1, A/B))
          KellerJordan variant; typically needs much larger LR (e.g., ~10x AdamW) and rescaled WD.

    Usage:
        muon_params = [...]
        adamw_params = [...]
        optimizer = Muon(
            [
                dict(params=muon_params, use_muon=True, lr=3e-4, weight_decay=0.1,
                     momentum=0.95, nesterov=True, ns_steps=5, muon_scaling="moonlight"),
                dict(params=adamw_params, use_muon=False, lr=3e-4, weight_decay=0.1,
                     adamw_betas=(0.9, 0.95), adamw_eps=1e-8),
            ]
        )

    Notes:
        - Works with DDP/FSDP out of the box: gradients are synchronized before optimizer.step().
        - Muon only supports 2D parameters by design (hidden dense layers). Do not include
          embeddings, bias, norm gamma in the Muon group.
        - For original scaling, be mindful of A/B dimension orientation. Torch.nn.Linear uses
          (out_features, in_features). Adjust if you use custom ops that invert this orientation.
    """

    def __init__(
        self,
        params: Union[List[Dict[str, Any]], Any],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        muon_scaling: str = "moonlight",  # group-level default; can be overridden per group
    ):
        """
        Args:
            params: List of param_groups; each must include 'use_muon': bool
            lr: Base learning rate (used as default for both modes)
            weight_decay: Decoupled weight decay coefficient
            momentum: Momentum for Muon (0.95 typical)
            nesterov: Whether to use Nesterov momentum in Muon
            ns_steps: Newton–Schulz iterations for orthogonalization
            adamw_betas: Betas for AdamW (beta1, beta2)
            adamw_eps: Epsilon for AdamW
            muon_scaling: Default scaling mode for Muon groups ("moonlight" or "original")
        """
        processed_groups: List[Dict[str, Any]] = []

        for group in params:
            # Handle raw param lists passed directly
            if not isinstance(group, dict):
                raise ValueError("Params must be passed as a list of dicts with 'use_muon' key.")

            use_muon = group.get("use_muon", False)  # Default to False if not specified

            pg = {
                "params": list(group["params"]),
                "use_muon": use_muon,
                "weight_decay": group.get("weight_decay", weight_decay),
                "lr": group.get("lr", lr),
            }
            if len(pg["params"]) == 0:
                # Skip empty groups to avoid errors downstream
                continue
            if use_muon:
                pg["momentum"] = group.get("momentum", momentum)
                pg["nesterov"] = group.get("nesterov", nesterov)
                pg["ns_steps"] = group.get("ns_steps", ns_steps)
                pg["muon_scaling"] = group.get("muon_scaling", muon_scaling)
            else:
                pg["adamw_betas"] = group.get("adamw_betas", adamw_betas)
                pg["adamw_eps"] = group.get("adamw_eps", adamw_eps)
            
            processed_groups.append(pg)
        super().__init__(processed_groups, {})

    @staticmethod
    def _muon_lr_scale(shape: torch.Size, mode: str) -> float:
        """
        Compute the Muon scaling factor based on matrix shape and scaling mode.

        Args:
            shape: 2D parameter shape
            mode: 'moonlight' or 'original'

        Returns:
            Scaling factor to multiply with the group's base LR.

        Notes:
            - Moonlight: 0.2 * sqrt(max(A, B)), symmetric w.r.t. dims.
            - Original:  sqrt(max(1, A/B)), orientation-sensitive (A=row, B=col).
              With torch.nn.Linear (out_features, in_features), this equals dout/din.
        """
        A, B = shape[:2]
        if mode == "moonlight":
            return 0.2 * math.sqrt(max(A, B))
        # "original"
        return math.sqrt(max(1.0, A / B))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one optimization step.

        Returns:
            Loss, if a closure is provided and evaluated.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                # Muon branch
                lr = group["lr"]
                wd = group["weight_decay"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                scaling_mode = group["muon_scaling"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients.")
                    if g.ndim > 2:
                         g = g.view(g.size(0), -1) # Flatten if needed (e.g. Conv2d 1x1 treated as linear)

                    assert g.ndim == 2, "Muon requires 2D gradients"

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                    buf = state["momentum_buffer"]
                    # buf.mul_(momentum).add_(g)
                    buf.lerp_(g, 1 - momentum)
                    update = g.lerp(buf, momentum) if nesterov else buf
                    # Orthogonalize
                    u = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Decoupled weight decay
                    if wd != 0:
                        p.data.mul_(1.0 - lr * wd)
                    # Apply scaled Muon update
                    scaled_lr = lr * self._muon_lr_scale(p.shape, scaling_mode)
                    p.data.add_(u.view_as(p).to(p.dtype), alpha=-scaled_lr)
            else:
                # AdamW branch
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    g = p.grad

                    # State initialization
                    if "step" not in state:
                        state["step"] = 0.0
                        state["exp_avg"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # AdamW updates
                    exp_avg.lerp_(g, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias_corr1 = 1.0 - beta1 ** state["step"]
                    bias_corr2 = 1.0 - beta2 ** state["step"]

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                    if wd != 0:
                        p.data.mul_(1.0 - lr * wd)
                    step_size = lr / bias_corr1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

def build_muon_param_groups(
    model: torch.nn.Module,
    param_names_adam: List[str] = ['embed'],
    param_names_no_decay_adam: List[str] = ['register', 'layer_scale'],
) -> List[Dict[str, Any]]:
    """
    Split model parameters into Muon (2D hidden matrices) and AdamW (others).
    Only classifies parameters; optimizer hyperparameters use Muon.__init__ defaults.

    Args:
        model: torch.nn.Module

    Returns:
        A list with two param groups containing only 'params' and 'use_muon'.
    """
    muon_params = []
    adam_params = []
    no_decay_adams = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        is_force_adam = any([n in lname for n in param_names_adam])
        is_no_decay = any([n in lname for n in param_names_no_decay_adam])
        is_eligible = (
            p.ndim == 2
            and p.size(0) > 32
            and p.size(1) > 32
            and not is_force_adam
            and not is_no_decay
        )

        if is_eligible:
            muon_params.append(p)
        else:
            if is_no_decay:
                no_decay_adams.append(p)
            else: # force adam.
                adam_params.append(p)

    param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
        ),
        dict(
            params=adam_params,
            use_muon=False,
        ),
        dict(
            params=no_decay_adams,
            use_muon=False,
            weight_decay=0.0,
        ),
    ]
    return param_groups
