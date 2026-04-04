from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from g2pt.neuralop.layers.norms import get_normalization
from g2pt.neuralop.layers.softmax import get_softmax
from g2pt.utils.common import roundup

from g2pt.neuralop.layers.attn.mha import MultiHeadAttention

class TransolverSubspaceGenerator(nn.Module):
    def __init__(self, d_model: int, modes: int, norm_type: str = "layernorm", enable_temperature: bool = False,):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        self.softmax = get_softmax()  # Softmax to normalize the subspace generation output
        self.lin = nn.Linear(d_model, modes)  # Simple linear layer for subspace generation
        self.norm = get_normalization(norm_type, d_model)
        self.min_norm = 1e-6
        self.enable_temperature = enable_temperature

        if self.enable_temperature:
            # Ada-Temp mechanism: τ = τ₀ + Linear(xᵢ), we set τ₀ in the last linear bias
            # Pointwise temperature adjustment network
            hid = roundup(int(math.sqrt(d_model)), 16)
            self.temp_proj = nn.Sequential(
                nn.Linear(d_model, hid, bias=False),
                nn.GELU(),
                nn.Linear(hid, 1, bias=True),
            )

        else:
            self.temp_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the subspace generator."""
        self.lin.reset_parameters()
        nn.init.orthogonal_(self.lin.weight) # type: ignore
        if self.enable_temperature:
            # Reset temperature parameters
            self.temp_proj[0].reset_parameters()
            # Re-initialize final layer to near-zero
            nn.init.zeros_(self.temp_proj[-1].weight)
            nn.init.ones_(self.temp_proj[-1].bias)

    def gumbel_softmax(self, logits, tau, hard=False):
        y = logits

        # TODO: we are doing some regression work, add the noise to the logits
        # will help?
        # It introduce very noisy gradients, but you should use it very carefully
        if self.training:
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            y = y + gumbel_noise

        y = y / torch.clamp(tau, min=1e-2, max=3)
        y = self.softmax(y)
        if hard:
            _, y_hard = y.max(dim=-1)
            y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
            y = (y_one_hot - y).detach() + y
        return y


    def forward(self, x: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate the subspace projection for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, nPoints, d_model).

        Returns:
            torch.Tensor: Subspace projection tensor of shape (B, nPoints, modes).
        """
        # Input tensor shape validation
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3D (batch, seq_len, features), got {x.shape}")
        if mass.dim() != 3 or mass.shape[-1] != 1:
            raise ValueError(f"mass must be of shape (B, nPoints, 1), got {mass.shape}")
        if x.shape[0] != mass.shape[0] or x.shape[1] != mass.shape[1]:
            raise ValueError(f"x and mass must have matching batch and sequence dimensions, got x: {x.shape}, mass: {mass.shape}")
        nx = self.norm(x)
        logits = self.lin(nx)

        if self.enable_temperature:
            temp_adjustment = self.temp_proj(nx) # b, npoints, 1
            temperature = F.softplus(temp_adjustment)
            # Use Gumbel-Softmax for differentiable categorical sampling
            # hard=False for training, hard=True for inference (straight-through estimator)
            trial_func = self.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            trial_func = self.softmax(logits)  # b, npoints, modes

        trial_in = trial_func * mass  # b, npoints, modes
        trial_in_t = trial_in.transpose(-1, -2)  # b, modes, npoints
        with torch.autocast(x.device.type, enabled=False):
            # b, modes, 1
            inv_trial_norm = torch.reciprocal(torch.sum(trial_in_t, dim=-1, keepdim=True) + self.min_norm)

        return trial_func, trial_in_t, inv_trial_norm

class TransolverSubspaceGenerator2(nn.Module):
    def __init__(self, d_model: int, modes: int, norm_type: str = "layernorm", enable_temperature: bool = False,):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        self.norm = get_normalization(norm_type, d_model)
        self.body = nn.Sequential(
            nn.Linear(d_model, modes),
            get_softmax(),
            # nn.Linear(d_model, d_model + modes),
            # nn.SiLU(),
            # nn.Linear(d_model + modes, modes),
        )
        self.gating = nn.Sequential(
            nn.Linear(d_model, modes),
            nn.Sigmoid(),
        )
        self.min_norm = 1e-6
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the subspace generator."""
        nn.init.orthogonal_(self.body[0].weight, 1 / math.sqrt(self.modes)) # type: ignore
        self.body[0].bias.data.fill_(0.0) # type: ignore
        nn.init.zeros_(self.gating[0].weight) # type: ignore
        self.gating[0].bias.data.fill_(0.0) # type: ignore

    def forward(self, x: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate the subspace projection for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, nPoints, d_model).

        Returns:
            torch.Tensor: Subspace projection tensor of shape (B, nPoints, modes).
        """
        # Input tensor shape validation
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3D (batch, seq_len, features), got {x.shape}")
        if mass.dim() != 3 or mass.shape[-1] != 1:
            raise ValueError(f"mass must be of shape (B, nPoints, 1), got {mass.shape}")
        if x.shape[0] != mass.shape[0] or x.shape[1] != mass.shape[1]:
            raise ValueError(f"x and mass must have matching batch and sequence dimensions, got x: {x.shape}, mass: {mass.shape}")
        x = self.norm(x)
        trial_func = self.body(x) # b, npoints, modes
        trial_func_gate = self.gating(x) # b, npoints, modes

        trial_in = trial_func * mass  # b, npoints, modes
        trial_in_t = trial_in.transpose(-1, -2)  # b, modes, npoints
        with torch.autocast(x.device.type, enabled=False):
            # b, modes, 1
            inv_trial_norm = torch.reciprocal(torch.sum(trial_in_t, dim=-1, keepdim=True) + self.min_norm)

        trial_func_rec = trial_func * trial_func_gate
        return (
            trial_func_rec, # For reconstruct from patches/modes
            trial_in_t,     # For down projection to patches/modes
            inv_trial_norm, # For normalize the down projection
        )

@dataclass
class TransolverAttentionConfig:
    num_heads: int
    modes: int
    bias: bool
    dim_heads: int | None = None
    enable_rope: bool = True
    qk_norm: bool = False
    norm_type: str = "layernorm"
    subgen_enable_temperature: bool = False
    sdpa_dropout_p: float = 0.0
    rope_min_log_freq: float = -5.0
    rope_max_log_freq: float = 3.0

class TransolverSelfAttention(nn.Module):
    """
    Transolver Attention (Self), similar to ViT design:

    1. Patchify: use subspace projection to simulate patches.
    2. Unpachify: unproject the patches to the original function space.
    """

    def __init__(
        self,
        d_model: int,
        phys_dim: int,
        config: TransolverAttentionConfig,
    ):
        super().__init__()
        num_heads = config.num_heads
        modes = config.modes
        bias = config.bias
        dim_heads = config.dim_heads
        enable_rope = config.enable_rope
        qk_norm = config.qk_norm
        norm_type = config.norm_type
        subgen_enable_temperature = config.subgen_enable_temperature

        self.modes = modes
        dim_heads = dim_heads or (d_model // num_heads)  # Default dimension of each head
        # Use shared subspace generator for patch projection
        self.sub_gen = TransolverSubspaceGenerator(d_model, modes, norm_type, enable_temperature=subgen_enable_temperature)
        self.attn_norm = get_normalization(norm_type, d_model)
        # TODO: Liger-based normalization kernels require CUDA; fallback to PyTorch LayerNorm/RMSNorm if running on CPU.
        self.sdpa = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dim_heads=dim_heads,
            bias=bias,
            norm_type=norm_type,
            enable_rope=enable_rope,
            rope_min_log_freq=config.rope_min_log_freq,
            rope_max_log_freq=config.rope_max_log_freq,
            qk_norm=qk_norm,
            phys_dim=phys_dim,
            dropout=config.sdpa_dropout_p,
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the reduced self-attention layer."""
        # Reset internal linear of subspace generator
        self.sub_gen.reset_parameters()
        self.sdpa.reset_parameters()

    def forward(
        self,
        fx: torch.Tensor,
        x: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the reduced self-attention layer.

        Args:
            fx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            x (torch.Tensor): Input tensor of shape (B, nPoints, phys_dim).
            mass (torch.Tensor | None): Pointwise mass tensor for weighting, if available. (B, nPoints, 1).

        Returns:
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        
        # Input tensor shape validation
        if fx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got fx: {fx.shape}, x: {x.shape}")
        if fx.shape[0] != x.shape[0] or fx.shape[1] != x.shape[1]:
            raise ValueError(f"fx and x must have matching batch and sequence dimensions, got fx: {fx.shape}, x: {x.shape}")
        
        # 1. Apply the subspace generation via shared generator
        # Ensure resolution invariance by always using (mass-)weighted means.
        # If `mass` is not provided, fall back to uniform ones so the projection
        # becomes a true mean over points under uniform sampling.
        #! Confirm fallback semantics depending on dataset: if sampling is non-uniform
        #! but `mass` is not available, uniform mass may not guarantee invariance.
        #! This is the user's duty, but not this class.
        if mass is None:
            mass = torch.ones_like(fx[..., :1])
        # Generator returns trial function, transposed weighted basis, and inverse norm.
        # TODO: Generator clamps using `min_norm`; legacy used `+1e-6`. If strict alignment is required,
        # set `min_norm` accordingly or adjust here.
        trial_func, trial_in_t, inv_trial_norm = self.sub_gen(fx, mass)
        fx_proj = torch.bmm(trial_in_t, fx) * inv_trial_norm  # [B, modes, d_model]
        x_proj = torch.bmm(trial_in_t, x) * inv_trial_norm  # [B, modes, phys_dim]
        qkv = self.attn_norm(fx_proj)  # self attention: qkv is same
        y_proj = self.sdpa(qkv, qkv, qkv, x_proj)  # [B, modes, d_model]

        # 3. reconstruction
        y_recon = torch.bmm(trial_func, y_proj)  # [B, nPoints, d_model]
        return y_recon

class TransolverCrossAttention_ShareKV(nn.Module):
    """
    Transolver Attention (Cross), similar to ViT design:

    1. Patchify: use subspace projection to simulate patches.
    2. Unpachify: unproject the patches to the original function space.
    """

    def __init__(
        self,
        kv_dim: int,
        q_dim: int,
        phys_dim: int,
        out_dim: int,
        config: TransolverAttentionConfig,
    ):
        super().__init__()
        num_heads = config.num_heads
        modes = config.modes
        bias = config.bias
        dim_heads = config.dim_heads
        enable_rope = config.enable_rope
        qk_norm = config.qk_norm
        norm_type = config.norm_type
        subgen_enable_temperature = config.subgen_enable_temperature

        self.modes = modes
        dim_heads = dim_heads or (kv_dim // num_heads)  # Default dimension of each head
        self.sub_q = TransolverSubspaceGenerator(q_dim, modes, norm_type, subgen_enable_temperature)
        self.sub_kv = TransolverSubspaceGenerator(kv_dim, modes, norm_type, subgen_enable_temperature)
        self.norm_q = get_normalization(norm_type, q_dim)
        self.norm_kv = get_normalization(norm_type, kv_dim)
        # TODO: Liger-based normalization kernels require CUDA; fallback to PyTorch LayerNorm/RMSNorm if running on CPU.
        self.sdpa = MultiHeadAttention(
            d_model=kv_dim,
            q_dim=q_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dim_heads=dim_heads,
            bias=bias,
            norm_type=norm_type,
            enable_rope=enable_rope,
            rope_min_log_freq=config.rope_min_log_freq,
            rope_max_log_freq=config.rope_max_log_freq,
            qk_norm=qk_norm,
            phys_dim=phys_dim,
            dropout=config.sdpa_dropout_p,
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the reduced self-attention layer."""
        self.sub_q.reset_parameters()
        self.sub_kv.reset_parameters()
        self.sdpa.reset_parameters()

    def forward(
        self,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        x: torch.Tensor,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply Transolver-style Cross Attention.

        Args:
            qx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            kvx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            x (torch.Tensor): Input tensor of shape (B, nPoints, phys_dim).
            mass (torch.Tensor | None): Pointwise mass tensor for weighting, if available. (B, nPoints, 1).

        Returns:
            torch.Tensor: Transformed tensor with the same shape as input.
        """
        
        # Input tensor shape validation
        if qx.dim() != 3 or kvx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got qx: {qx.shape}, kvx: {kvx.shape}, x: {x.shape}")
        if qx.shape[0] != kvx.shape[0] or qx.shape[0] != x.shape[0]:
            raise ValueError(f"Batch sizes must match, got qx: {qx.shape[0]}, kvx: {kvx.shape[0]}, x: {x.shape[0]}")
        if qx.shape[1] != kvx.shape[1] or qx.shape[1] != x.shape[1]:
            raise ValueError(f"Sequence lengths must match, got qx: {qx.shape[1]}, kvx: {kvx.shape[1]}, x: {x.shape[1]}")

        if mass_q is None:
            mass_q = torch.ones_like(qx[..., :1])
        if mass_kv is None:
            mass_kv = torch.ones_like(kvx[..., :1])
        # Generator returns trial function, transposed weighted basis, and inverse norm.
        _, trial_in_t_q, inv_trial_norm_q = self.sub_q(qx, mass_q)
        trial_func_kv, trial_in_t_kv, inv_trial_norm_kv = self.sub_kv(kvx, mass_kv)

        # 2.1 Project
        #? bmm: sum(trial(x_i) * f(x_i) * mass(x_i)), inv_norm = 1 / sum(trial(x_i) * mass(x_i))
        #? => xx_proj is resolution invariant
        qx_proj = torch.bmm(trial_in_t_q, qx) * inv_trial_norm_q  # [B, modes, d_model]
        kvx_proj = torch.bmm(trial_in_t_kv, kvx) * inv_trial_norm_kv  # [B, modes, d_model]
        x_q_proj = torch.bmm(trial_in_t_q, x) * inv_trial_norm_q  # [B, modes, phys_dim]
        x_k_proj = torch.bmm(trial_in_t_kv, x) * inv_trial_norm_kv  # [B, modes, phys_dim]

        # 2.2 Standard Attention
        q = self.norm_q(qx_proj)  # [B, modes, d_model]
        kv = self.norm_kv(kvx_proj)  # [B, modes, d_model]
        y_proj = self.sdpa(q, kv, kv, x_q_proj, x_k_proj)  # [B, modes, out_dim]

        # 3. reconstruction
        y_recon = torch.bmm(trial_func_kv, y_proj)  # [B, nPoints, out_dim]
        return y_recon
