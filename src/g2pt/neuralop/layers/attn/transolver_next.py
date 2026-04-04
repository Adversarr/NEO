from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from g2pt.neuralop.layers.mlps import FeedForwardWithGating, FeedForwardWithGatingConfig
from g2pt.neuralop.layers.norms import get_normalization
from g2pt.neuralop.layers.softmax import get_softmax
from g2pt.utils.common import roundup

from g2pt.neuralop.layers.attn.mha import MultiHeadAttention
from g2pt.neuralop.layers.attn.transolver import TransolverSubspaceGenerator

@dataclass
class TransolverNeXtAttentionConfig:
    num_heads: int
    modes: int
    bias: bool
    ffn: FeedForwardWithGatingConfig
    dim_heads: int | None = None
    enable_rope: bool = True
    rope_min_log_freq: float = -5.0
    rope_max_log_freq: float = 3.0
    qk_norm: bool = False
    norm_type: str = "layernorm"
    subgen_enable_temperature: bool = False
    sdpa_dropout_p: float = 0.0
    num_interleaved_layers: int = 0
    enable_post_norm: bool = True

class TransolverNeXtSelfAttention(nn.Module):
    """
    TransolverNeXt Attention (Self), similar to ViT design:

    1. Patchify: use subspace projection to simulate patches.
    2. Unpachify: unproject the patches to the original function space.

    It follows the transolver's design but saves computation of FFN on points by pre-computing the FFN on patches.
    To get better result, you should reduce the outer point-FFN hidden_dimension/mlp_ratio
    """

    def __init__(
        self,
        d_model: int,
        phys_dim: int,
        config: TransolverNeXtAttentionConfig,
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
        num_interleaved_layers = config.num_interleaved_layers

        self.modes = modes
        dim_heads = dim_heads or (d_model // num_heads)  # Default dimension of each head
        # Use shared subspace generator for patch projection
        self.sub_gen = TransolverSubspaceGenerator(
            d_model, modes, norm_type, enable_temperature=subgen_enable_temperature
        )
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
        # Backward Compatibility: enable_post_norm is True by default
        enable_post_norm = config.enable_post_norm if hasattr(config, "enable_post_norm") else True
        self.out_norm = get_normalization(norm_type, d_model) if enable_post_norm else nn.Identity()
        self.inter_attn = nn.ModuleList()
        self.inter_attn_norm = nn.ModuleList()
        self.inter_ffn = nn.ModuleList()
        self.inter_ffn_norm = nn.ModuleList()
        for _ in range(num_interleaved_layers):
            self.inter_attn.append(MultiHeadAttention(
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
            ))
            self.inter_attn_norm.append(get_normalization(norm_type, d_model))
            self.inter_ffn.append(FeedForwardWithGating(d_model, d_model, config.ffn))
            self.inter_ffn_norm.append(get_normalization(norm_type, d_model))


    def reset_parameters(self) -> None:
        """Reset the parameters of the reduced self-attention layer."""
        # Reset internal linear of subspace generator
        self.sub_gen.reset_parameters()
        self.sdpa.reset_parameters()
        for inter_attn in self.inter_attn:
            inter_attn.reset_parameters()
        for inter_ffn in self.inter_ffn:
            inter_ffn.reset_parameters()


    def forward(
        self,
        fx: torch.Tensor,
        x: torch.Tensor,
        reg_tok_fx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply reduced self-attention with register tokens.

        Args:
            fx: Functional features `(B, nPoints, d_model)`.
            x: Physical coordinates `(B, nPoints, phys_dim)`.
            reg_tok_fx: Register tokens in feature space `(B, nRegisters, d_model)`.
            reg_tok_x: Register tokens in physical space `(B, nRegisters, phys_dim)`.
            mass: Optional pointwise mass `(B, nPoints, 1)`.

        Returns:
            Tuple `(y_recon, reg_token)` where `y_recon` has shape `(B, nPoints, d_model)`
            and `reg_token` has shape `(B, nRegisters, d_model)`.
        """
        
        # Input tensor shape validation
        if fx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got fx: {fx.shape}, x: {x.shape}")
        if fx.shape[0] != x.shape[0] or fx.shape[1] != x.shape[1]:
            raise ValueError(f"fx and x must have matching batch and sequence dimensions, got fx: {fx.shape}, x: {x.shape}")
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

        # 2.1 First essential attention with Pre-Norm
        fx_proj_full = torch.cat([fx_proj, reg_tok_fx], dim=1)  # [B, modes+nReg, d_model]
        x_proj_full = torch.cat([x_proj, reg_tok_x], dim=1)  # [B, modes+nReg, phys_dim]

        qkv = self.attn_norm(fx_proj_full)
        y_proj_full = self.sdpa(qkv, qkv, qkv, x_proj_full) + fx_proj_full  # [B, modes+nReg, d_model]

        # 2.2 Apply interleaved modules on the concatenated sequence
        for ffn_norm, ffn, attn_norm, attn in zip(
            self.inter_ffn_norm, self.inter_ffn, self.inter_attn_norm, self.inter_attn
        ):
            y_proj_full = ffn(ffn_norm(y_proj_full)) + y_proj_full
            qkv = attn_norm(y_proj_full)
            y_proj_full = attn(qkv, qkv, qkv, x_proj_full) + y_proj_full  # [B, modes+nReg, d_model]

        # 2.3 Post-Norm and split
        y_proj_full = self.out_norm(y_proj_full)
        y_proj = y_proj_full[:, :self.modes, :]  # [B, modes, d_model]
        reg_token = y_proj_full[:, self.modes:, :]  # [B, nReg, d_model]

        # 3. Reconstruction
        y_recon = torch.bmm(trial_func, y_proj)  # [B, nPoints, d_model]
        return y_recon, reg_token

class TransolverNeXtCrossAttention_ShareKV(nn.Module):
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
        config: TransolverNeXtAttentionConfig,
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
        num_interleaved_layers = config.num_interleaved_layers

        self.modes = modes
        dim_heads = dim_heads or (kv_dim // num_heads)  # Default dimension of each head
        self.sub_q = TransolverSubspaceGenerator(q_dim, modes, norm_type, subgen_enable_temperature)
        self.sub_kv = TransolverSubspaceGenerator(kv_dim, modes, norm_type, subgen_enable_temperature)
        self.norm_q = get_normalization(norm_type, q_dim)
        self.norm_kv = get_normalization(norm_type, kv_dim)
        self.sdpa = MultiHeadAttention(
            d_model=kv_dim,
            q_dim=q_dim,
            out_dim=q_dim,
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
        self.out_norm = get_normalization(norm_type, q_dim) if config.enable_post_norm else nn.Identity()

        
        self.inter_attn = nn.ModuleList()
        self.inter_attn_norm_q = nn.ModuleList()
        self.inter_ffn = nn.ModuleList()
        self.inter_ffn_norm = nn.ModuleList()
        for i in range(num_interleaved_layers):
            self.inter_attn.append(
                MultiHeadAttention(
                    d_model=kv_dim,
                    q_dim=q_dim,
                    out_dim=q_dim,
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
            )
            self.inter_attn_norm_q.append(get_normalization(norm_type, q_dim))
            self.inter_ffn.append(FeedForwardWithGating(q_dim, q_dim, config.ffn))
            self.inter_ffn_norm.append(get_normalization(norm_type, q_dim))

    def reset_parameters(self) -> None:
        """Reset the parameters of the reduced self-attention layer."""
        self.sub_q.reset_parameters()
        self.sub_kv.reset_parameters()
        self.sdpa.reset_parameters()
        for inter_attn, inter_attn_norm, inter_ffn, inter_ffn_norm in zip(
            self.inter_attn, self.inter_attn_norm_q, self.inter_ffn, self.inter_ffn_norm
        ):
            inter_attn.reset_parameters()
            inter_ffn.reset_parameters()

    def forward(
        self,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        x: torch.Tensor,
        reg_tok_qx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Transolver-style cross attention with register tokens.

        Args:
            qx: Query features `(B, nPoints, q_dim)`.
            kvx: Key/Value features `(B, nPoints, kv_dim)`.
            x: Physical coordinates for query `(B, nPoints, phys_dim)`.
            reg_tok_qx: Register tokens in feature space `(B, nRegisters, q_dim)`.
            reg_tok_x: Register tokens in physical space `(B, nRegisters, phys_dim)`.
            x_kv: Optional physical coordinates for kv `(B, nPoints_kv, phys_dim)`.
            mass_q: Optional query mass `(B, nPoints, 1)`.
            mass_kv: Optional kv mass `(B, nPoints_kv, 1)`.

        Returns:
            Tuple `(y_recon, reg_token)` with shapes `(B, nPoints, out_dim)` and `(B, nRegisters, out_dim)`.
        """
        
        # Input tensor shape validation
        if qx.dim() != 3 or kvx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got qx: {qx.shape}, kvx: {kvx.shape}, x: {x.shape}")
        if qx.shape[0] != kvx.shape[0] or qx.shape[0] != x.shape[0]:
            raise ValueError(f"Batch sizes must match, got qx: {qx.shape[0]}, kvx: {kvx.shape[0]}, x: {x.shape[0]}")

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
        qx_proj = torch.bmm(trial_in_t_q, qx) * inv_trial_norm_q  # [B, modes, q_dim]
        kvx_proj = torch.bmm(trial_in_t_kv, kvx) * inv_trial_norm_kv  # [B, modes, kv_dim]
        x_q_proj = torch.bmm(trial_in_t_q, x) * inv_trial_norm_q  # [B, modes, phys_dim]
        x_k_input = x_kv if x_kv is not None else x
        x_k_proj = torch.bmm(trial_in_t_kv, x_k_input) * inv_trial_norm_kv  # [B, modes, phys_dim]

        # 2.2 Standard Attention
        qx_proj_full = torch.cat([qx_proj, reg_tok_qx], dim=1)  # [B, modes+nReg, q_dim]
        x_q_proj_full = torch.cat([x_q_proj, reg_tok_x], dim=1)  # [B, modes+nReg, phys_dim]

        q = self.norm_q(qx_proj_full)
        kv = self.norm_kv(kvx_proj)
        y_proj_full = self.sdpa(q, kv, kv, x_q_proj_full, x_k_proj) + qx_proj_full  # [B, modes+nReg, out_dim]

        # 2.4 Interleaved Attention on concatenated sequence
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.inter_attn, self.inter_attn_norm_q, self.inter_ffn, self.inter_ffn_norm
        ):
            y_proj_full = ffn(ffn_norm(y_proj_full)) + y_proj_full  # [B, modes+nReg, out_dim]
            q = attn_norm(y_proj_full)
            y_proj_full = attn(q, kv, kv, x_q_proj_full, x_k_proj) + y_proj_full  # [B, modes+nReg, out_dim]

        # 2.5 Output Normalization
        y_proj_full = self.out_norm(y_proj_full)
        y_proj = y_proj_full[:, :self.modes, :]  # [B, modes, out_dim]
        reg_token = y_proj_full[:, self.modes:, :]  # [B, nReg, out_dim]
        # 3. Reconstruction
        y_recon = torch.bmm(trial_func_q, y_proj)  # [B, nPoints, out_dim]
        # y_recon = y_recon - y_recon.mean(dim=1, keepdim=True)
        return y_recon, reg_token
