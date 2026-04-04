from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from g2pt.neuralop.layers.mlps import FeedForwardWithGating, FeedForwardWithGatingConfig
from g2pt.neuralop.layers.norms import get_normalization
from g2pt.neuralop.layers.rope import RoPE_M, apply_rotary_pos_emb


@dataclass
class TransolverExpAttentionConfig:
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


class TransolverExpSelfAttention(nn.Module):
    def __init__(self, d_model: int, phys_dim: int, config: TransolverExpAttentionConfig):
        super().__init__()
        num_heads = config.num_heads
        modes = config.modes
        bias = config.bias
        dim_heads = config.dim_heads
        norm_type = config.norm_type

        self.modes = modes
        self.num_heads = num_heads
        self.dim_heads = dim_heads = dim_heads or (d_model // num_heads)  # Default dimension of each head
        self.slice_features = nn.Parameter(torch.randn(num_heads, modes, dim_heads))  # (h, m, d)

        self.enable_rope = config.enable_rope
        if self.enable_rope:
            self.rope = RoPE_M(phys_dim, dim_heads, config.rope_min_log_freq, config.rope_max_log_freq)

        attn_dim = dim_heads * num_heads
        self.to_k = nn.Linear(d_model, attn_dim, bias=bias)
        self.to_v = nn.Linear(d_model, attn_dim, bias=bias)
        self.to_out = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.q_norm = get_normalization(norm_type, dim_heads)
        self.k_norm = get_normalization(norm_type, dim_heads)
        # Channel Mixing 1
        self.channel_norm_1 = get_normalization(norm_type, attn_dim)
        self.channel_mixing_1 = FeedForwardWithGating(attn_dim, attn_dim, config.ffn)

        # slice mixing
        self.sm_norm = get_normalization(norm_type, attn_dim)
        self.sm_to_q = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.sm_to_k = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.sm_to_out = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.sm_to_v = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.sm_q_norm = get_normalization(norm_type, dim_heads)
        self.sm_k_norm = get_normalization(norm_type, dim_heads)

        # Channel Mixing 2
        self.channel_norm_2 = get_normalization(norm_type, attn_dim)
        self.channel_mixing_2 = FeedForwardWithGating(attn_dim, attn_dim, config.ffn)

        # Attn 2
        self.to_q_2 = nn.Linear(d_model, attn_dim, bias=bias)
        self.to_k_2 = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.to_v_2 = nn.Linear(attn_dim, attn_dim, bias=bias)
        self.to_out_2 = nn.Linear(attn_dim, d_model, bias=bias)
        self.q_norm_2 = get_normalization(norm_type, dim_heads)
        self.k_norm_2 = get_normalization(norm_type, dim_heads)
        self.out_slice_norm = get_normalization(norm_type, attn_dim)

        self.scale = dim_heads ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.02
        cutoff = 3.0
        nn.init.trunc_normal_(self.slice_features, std=std, a=-cutoff, b=cutoff)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-cutoff, b=cutoff)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.channel_mixing_1.reset_parameters()
        self.channel_mixing_2.reset_parameters()

    def forward(
        self,
        fx: torch.Tensor,
        x: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reduced self-attention with register tokens.

        Args:
            fx: Functional features `(B, nPoints, d_model)`.
            x: Physical coordinates `(B, nPoints, phys_dim)`.
            mass: Optional pointwise mass `(B, nPoints, 1)`.

        Returns:
            Tuple `(y_recon, reg_token)` where `y_recon` has shape `(B, nPoints, d_model)`
            and `reg_token` has shape `(B, nRegisters, d_model)`.
            lbl: load balancing loss
        """
        
        # Input tensor shape validation
        if fx.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Input tensors must be 3D (batch, seq_len, features), got fx: {fx.shape}, x: {x.shape}")
        if fx.shape[0] != x.shape[0] or fx.shape[1] != x.shape[1]:
            raise ValueError(f"fx and x must have matching batch and sequence dimensions, got fx: {fx.shape}, x: {x.shape}")


        b, n = fx.shape[:2]
        h, d = self.num_heads, self.dim_heads

        # Cross attention 1. Q = slice, KV = points
        k = self.to_k(fx).view(b, n, h, d)
        k = k.permute(0, 2, 1, 3)  # (b h n d)
        v = self.to_v(fx).view(b, n, h, d).permute(0, 2, 1, 3)  # (b h n d)
        q = self.slice_features.view(1, h, self.modes, d).expand(b, -1, -1, -1)  # (b h m d)
        q, k = self.q_norm(q), self.k_norm(k) # (b h m d), (b h n d)
        if self.enable_rope:
            k, cos, sin = self.rope(k, x, return_cos_sin=True) # (b h n d)
        else:
            cos = sin = None

        attn_mask = None
        if mass is not None: # Weighted by mass
            # mass: (B, N, 1) -> log_mass: (B, 1, 1, N) for broadcasting to (B, H, M, N)
            attn_mask = torch.log(mass.clamp(min=1e-6)).view(b, 1, 1, n)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale) # (b h m d)
        o = o.permute(0, 2, 1, 3).contiguous().view(b, self.modes, h * d) # (b m h*d)
        o = self.to_out(o) # (b m attn_dim)

        # Channel Mixing 1
        slice = o + self.channel_mixing_1(self.channel_norm_1(o)) # (b m h*d)

        # Slice Mixing
        sn = self.sm_norm(slice) # (b m h*d)
        q = self.sm_to_q(sn).view(b, self.modes, h, d).permute(0, 2, 1, 3)  # (b h m d)
        k = self.sm_to_k(sn).view(b, self.modes, h, d).permute(0, 2, 1, 3)  # (b h m d)
        v = self.sm_to_v(sn).view(b, self.modes, h, d).permute(0, 2, 1, 3)  # (b h m d)
        q, k = self.sm_q_norm(q), self.sm_k_norm(k) # (b h m d), (b h m d)
        o = F.scaled_dot_product_attention(q, k, v, scale=self.scale) # (b h m d)
        o = o.permute(0, 2, 1, 3).contiguous().view(b, self.modes, h * d) # (b m h*d)
        attn_slice = self.sm_to_out(o) # (b m h*d)
        slice = slice + attn_slice # (b m h*d)

        # Channel Mixing 2
        slice = slice + self.channel_mixing_2(self.channel_norm_2(slice)) # (b m h*d)

        # Cross attention 2. Q = points, KV = slice
        slice = self.out_slice_norm(slice) # (b m h*d)
        q = self.to_q_2(fx).view(b, n, h, d).permute(0, 2, 1, 3)  # (b h n d)
        k = self.to_k_2(slice).view(b, self.modes, h, d).permute(0, 2, 1, 3)  # (b h m d)
        v = self.to_v_2(slice).view(b, self.modes, h, d).permute(0, 2, 1, 3)  # (b h m d)
        q, k = self.q_norm_2(q), self.k_norm_2(k) # (b h n d), (b h m d)
        if cos is not None and sin is not None:
            q = apply_rotary_pos_emb(q, cos, sin) # (b h n d)
        o = F.scaled_dot_product_attention(q, k, v, scale=self.scale) # (b h n d)
        o = o.permute(0, 2, 1, 3).contiguous().view(b, n, h * d) # (b n h*d)
        o = self.to_out_2(o) # (b n d_model)
        return o
