
from typing import Tuple
import torch
from torch import nn

from g2pt.neuralop.layers.norms import get_normalization

from g2pt.neuralop.layers.attn.mha import MultiHeadAttention
from g2pt.neuralop.layers.attn.transolver import TransolverAttentionConfig, TransolverSubspaceGenerator


class Transolver2SelfAttention(nn.Module):
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
        reg_tok_fx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the reduced self-attention layer.

        Args:
            fx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            x (torch.Tensor): Input tensor of shape (B, nPoints, phys_dim).
            reg_tok_fx (torch.Tensor): Register tokens of shape (B, nRegisterTokens, d_model).
            reg_tok_x (torch.Tensor): Register tokens of shape (B, nRegisterTokens, phys_dim).
            mass (torch.Tensor | None): Pointwise mass tensor for weighting, if available. (B, nPoints, 1).

        Returns:
            torch.Tensor: Transformed fx tensor with the same shape as input.
            torch.Tensor: Register tokens with the same shape as input.
        """
        
        # Input tensor shape validation
        if fx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got fx: {fx.shape}, x: {x.shape}")
        if fx.shape[0] != x.shape[0] or fx.shape[1] != x.shape[1]:
            raise ValueError(f"fx and x must have matching batch and sequence dimensions, got fx: {fx.shape}, x: {x.shape}")
        
        # Validate register token batch and feature alignment
        if reg_tok_fx.dim() != 3 or reg_tok_x.dim() != 3:
            raise ValueError(f"Register tokens must be 3D, got reg_tok_fx: {reg_tok_fx.shape}, reg_tok_x: {reg_tok_x.shape}")
        if reg_tok_fx.shape[0] != fx.shape[0] or reg_tok_x.shape[0] != x.shape[0]:
            raise ValueError("Register tokens batch size must match inputs")
        if reg_tok_fx.shape[-1] != fx.shape[-1] or reg_tok_x.shape[-1] != x.shape[-1]:
            raise ValueError("Register token feature dimensions must match fx/x respectively")

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

        fx_proj_full = torch.cat([fx_proj, reg_tok_fx], dim=1) # [B, modes+nRegisterTokens, d_model]
        x_proj_full = torch.cat([x_proj, reg_tok_x], dim=1) # [B, modes+nRegisterTokens, phys_dim]
        qkv = self.attn_norm(fx_proj_full)  # self attention: qkv is same
        y_proj_full = self.sdpa(qkv, qkv, qkv, x_proj_full)  # [B, modes+nRegisterTokens, d_model]
        y_proj = y_proj_full[:, :self.modes, :]  # [B, modes, d_model]
        reg_token = y_proj_full[:, self.modes:, :]  # [B, nRegisterTokens, d_model]
        
        # 3. reconstruction
        y_recon = torch.bmm(trial_func, y_proj)  # [B, nPoints, d_model]
        return y_recon, reg_token


class Transolver2CrossAttention_ShareKV(nn.Module):
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
        reg_tok_qx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        x_kv: torch.Tensor | None=None,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Transolver-style Cross Attention.

        Args:
            qx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            kvx (torch.Tensor): Input tensor of shape (B, nPoints, d_model).
            x (torch.Tensor): Input tensor of shape (B, nPoints, phys_dim).
            mass (torch.Tensor | None): Pointwise mass tensor for weighting, if available. (B, nPoints, 1).

        Returns:
            torch.Tensor: Transformed qx tensor with the same shape as input.
            torch.Tensor: Register tokens with the same shape as input.
        """
        
        # Input tensor shape validation
        if qx.dim() != 3 or kvx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got qx: {qx.shape}, kvx: {kvx.shape}, x: {x.shape}")
        if qx.shape[0] != kvx.shape[0] or qx.shape[0] != x.shape[0]:
            raise ValueError(f"Batch sizes must match, got qx: {qx.shape[0]}, kvx: {kvx.shape[0]}, x: {x.shape[0]}")
        if qx.shape[1] != x.shape[1]:
            raise ValueError(f"Sequence lengths must match, got qx: {qx.shape[1]}, x: {x.shape[1]}")
        if x_kv is not None and x_kv.shape[1] != kvx.shape[1]:
            raise ValueError(f"Sequence lengths must match, got x_kv: {x_kv.shape[1]}, kvx: {kvx.shape[1]}")

        if mass_q is None:
            mass_q = torch.ones_like(qx[..., :1])
        if mass_kv is None:
            mass_kv = torch.ones_like(kvx[..., :1])
        # Validate register token shapes for cross attention
        if reg_tok_qx.dim() != 3 or reg_tok_x.dim() != 3:
            raise ValueError(f"Register tokens must be 3D, got reg_tok_qx: {reg_tok_qx.shape}, reg_tok_x: {reg_tok_x.shape}")
        if reg_tok_qx.shape[0] != qx.shape[0] or reg_tok_x.shape[0] != x.shape[0]:
            raise ValueError("Register tokens batch size must match qx/x respectively")
        if reg_tok_qx.shape[-1] != qx.shape[-1] or reg_tok_x.shape[-1] != x.shape[-1]:
            raise ValueError("Register token feature dimensions must match qx/x respectively")
        # Generator returns trial function, transposed weighted basis, and inverse norm.
        trial_func_q, trial_in_t_q, inv_trial_norm_q = self.sub_q(qx, mass_q)
        trial_func_kv, trial_in_t_kv, inv_trial_norm_kv = self.sub_kv(kvx, mass_kv)

        # 2.1 Project
        #? bmm: sum(trial(x_i) * f(x_i) * mass(x_i)), inv_norm = 1 / sum(trial(x_i) * mass(x_i))
        #? => xx_proj is resolution invariant
        qx_proj = torch.bmm(trial_in_t_q, qx) * inv_trial_norm_q  # [B, modes, d_model]
        kvx_proj = torch.bmm(trial_in_t_kv, kvx) * inv_trial_norm_kv  # [B, modes, d_model]
        x_q_proj = torch.bmm(trial_in_t_q, x) * inv_trial_norm_q  # [B, modes, phys_dim]
        x_k_input = x_kv if x_kv is not None else x
        x_k_proj = torch.bmm(trial_in_t_kv, x_k_input) * inv_trial_norm_kv  # [B, modes, phys_dim]

        qx_proj_full = torch.cat([qx_proj, reg_tok_qx], dim=1) # [B, modes+nRegisterTokens, d_model]
        x_q_proj_full = torch.cat([x_q_proj, reg_tok_x], dim=1) # [B, modes+nRegisterTokens, phys_dim]

        # 2.2 Standard Attention
        q = self.norm_q(qx_proj_full)  # [B, modes+nRegisterTokens, d_model]
        kv = self.norm_kv(kvx_proj)  # [B, modes, d_model]
        y_proj_full = self.sdpa(q, kv, kv, x_q_proj_full, x_k_proj)  # [B, modes+nRegisterTokens, out_dim]
        y_proj = y_proj_full[:, :self.modes, :]  # [B, modes, out_dim]
        reg_token = y_proj_full[:, self.modes:, :]  # [B, nRegisterTokens, out_dim]
        # 3. reconstruction, unproject use q weights.
        y_recon = torch.bmm(trial_func_q, y_proj)  # [B, nPoints, out_dim]
        return y_recon, reg_token
