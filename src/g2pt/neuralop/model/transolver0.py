"""This is just a mirror implementation of Transolver from thuml. Do not use this in production."""

from dataclasses import dataclass
import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import MultiLayerFeedForward, MultiLayerFeedForwardConfig
import warnings

from g2pt.utils.common import roundup

def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights


def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau

    y = F.softmax(y, dim=-1)

    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num), nn.GELU(), nn.Linear(slice_num, 1), nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = (
            self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )  # B H N C

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Transolver0MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(Transolver0MLP, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class Transolver_plus_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = Transolver0MLP(hidden_dim, roundup(hidden_dim * mlp_ratio, 16), hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=True) + fx
        else:
            fx += self.Attn(self.ln_1(fx))
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

@dataclass
class Transolver0ModelConfig:
    d_model: int
    num_layers: int
    # lifting part
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    # Transolver0 Block
    attn_heads: int
    mlp_ratio: int
    slice_num: int



class Transolver0Model(nn.Module):
    def __init__(self, phys_dim: int, func_dim: int, out_dim: int, config: Transolver0ModelConfig):
        super(Transolver0Model, self).__init__()
        print(config)
        self.out_dim = out_dim
        self.func_dim = func_dim
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = func_dim + self.embed_position.output_channels
        self.config = config
        self.n_hidden = config.d_model
        self.space_dim = phys_dim
        self.lifting = MultiLayerFeedForward(self.total_in_dim, config.d_model, config.lifting)
        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=config.attn_heads, hidden_dim=config.d_model,
                                                      dropout=0,
                                                      act='gelu',
                                                      mlp_ratio=config.mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=config.slice_num,
                                                      last_layer=(_ == config.num_layers - 1))
                                     for _ in range(config.num_layers)])
        self.reset_parameters()
        warnings.warn("Transolver0Model is a mirror of original Transolver model, "
                      "its capacity is much smaller than our optimized version.")

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_lift(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lift the input tensor to a higher-dimensional space."""
        pe = self.embed_position(x)
        out = [pe]
        if fx is not None:
            out.append(fx)
            assert fx.shape[-1] == self.func_dim, "Functional dimension mismatch"
        else:
            assert self.func_dim == 0, "Functional input is required but not provided."

        return self.lifting(torch.cat(out, dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transolver0 does not use mass matrix"""
        y = self.forward_lift(x, fx)

        for i, block in enumerate(self.blocks):
            y = block(y)
        return y