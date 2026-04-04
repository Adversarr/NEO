import torch
from torch import nn
from torch.nn import functional as F

from g2pt.neuralop.layers.norms import get_normalization
from g2pt.neuralop.layers.rope import RoPE_M


class MultiHeadAttention(nn.Module):
    """
    A simple attention mechanism that computes the attention weights and applies them to the input tensor.
    This is a canonical attention layer that can be used in various neural network architectures.

    This class does not apply any masking or additional operations,
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_dim: int | None = None,
        out_dim: int | None = None,
        dim_heads: int | None = None,
        norm_type: str = "layernorm",
        bias: bool = False,
        enable_rope: bool = True,
        rope_min_log_freq: float = -5,
        rope_max_log_freq: float = 3,
        qk_norm: bool = False,
        phys_dim: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.bias = bias
        self.qk_norm = qk_norm
        self.q_dim = q_dim or d_model
        self.out_dim = out_dim or d_model
        self.dropout = dropout

        dim_heads = dim_heads or (d_model // num_heads)  # Default dimension of each head
        self.dim_heads = dim_heads
        if qk_norm:
            self.q_norm: nn.LayerNorm | nn.RMSNorm = get_normalization(norm_type, dim_heads)  # type: ignore
            self.k_norm: nn.LayerNorm | nn.RMSNorm = get_normalization(norm_type, dim_heads)  # type: ignore

        head_total = num_heads * dim_heads
        self.to_q = nn.Linear(self.q_dim, head_total, bias=bias)
        self.to_k = nn.Linear(d_model, head_total, bias=bias)
        self.to_v = nn.Linear(d_model, head_total, bias=bias)
        self.out_proj = nn.Linear(head_total, self.out_dim, bias=bias)
        if enable_rope:
            self.rope = RoPE_M(
                phys_dim=phys_dim, d_model=dim_heads, min_log_freq=rope_min_log_freq, max_log_freq=rope_max_log_freq
            )  # Rotary Position Embedding
        self.enable_rope = enable_rope

    def reset_parameters(self) -> None:
        """Reset the parameters of the canonical self-attention layer."""
        self.to_q.reset_parameters()
        self.to_k.reset_parameters()
        self.to_v.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(
        self,
        qx: torch.Tensor,
        kx: torch.Tensor,
        vx: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input tensor shape validation
        if qx.dim() != 3 or kx.dim() != 3 or vx.dim() != 3:
            raise ValueError(
                f"Input tensors must be 3D (batch, seq_len, features), got qx: {qx.shape}, kx: {kx.shape}, vx: {vx.shape}"
            )
        if qx.shape[0] != kx.shape[0] or qx.shape[0] != vx.shape[0]:
            raise ValueError(f"Batch sizes must match, got qx: {qx.shape[0]}, kx: {kx.shape[0]}, vx: {vx.shape[0]}")

        q = self.to_q(qx)  # B, N, C
        k = self.to_k(kx)  # B, N, C
        v = self.to_v(vx)  # B, N, C

        bs, q_len, _ = qx.shape
        k_len, v_len = kx.shape[1], vx.shape[1]
        v = v.view(bs, v_len, self.num_heads, -1).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q.view(bs, q_len, self.num_heads, -1)).transpose(1, 2)  # B, H, N, PerHead
            k = self.k_norm(k.view(bs, k_len, self.num_heads, -1)).transpose(1, 2)  # B, H, N, PerHead
        else:
            q = q.view(bs, q_len, self.num_heads, -1).transpose(1, 2)
            k = k.view(bs, k_len, self.num_heads, -1).transpose(1, 2)

        if self.enable_rope:
            assert self.rope is not None, "RoPE should be initialized."
            q = self.rope(q, x_q)
            k = self.rope(k, x_k if x_k is not None else x_q)

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)  # B, H, N, PerHead
        return self.out_proj(y.transpose(1, 2).reshape(bs, q_len, -1))  # B, N, C
