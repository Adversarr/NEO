from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from g2pt.neuralop.layers.attn.transolver_next import (
    TransolverNeXtAttentionConfig,
    TransolverNeXtSelfAttention,
    TransolverNeXtCrossAttention_ShareKV,
)
from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import (
    FeedForwardWithGating,
    FeedForwardWithGatingConfig,
    MultiLayerFeedForward,
    MultiLayerFeedForwardConfig,
)
from g2pt.neuralop.layers.norms import get_normalization


class TransolverNeXtBlock(nn.Module):
    def __init__(
        self,
        # attention
        d_model: int,
        phys_dim: int,
        attn: TransolverNeXtAttentionConfig,
        ffn: FeedForwardWithGatingConfig,
        ffn_norm_type: str,
        layer_scale_init,
    ) -> None:
        super().__init__()
        self.attn = TransolverNeXtSelfAttention(d_model, phys_dim, attn)
        requires_grad_for_layer_scale = layer_scale_init != 0
        value_layer_scale_init = layer_scale_init if layer_scale_init != 0 else 1.0
        self.attn_layer_scale = nn.Parameter(
            torch.ones(1, 1, d_model) * value_layer_scale_init,
            requires_grad=requires_grad_for_layer_scale,
        )
        self.ffn = FeedForwardWithGating(d_model, d_model, ffn)
        self.ffn_ln = get_normalization(ffn_norm_type, d_model)

    def reset_parameters(self):
        """Reset the parameters of the transformer block."""
        self.attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        fx: torch.Tensor,
        reg_tok_fx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update point features and register tokens in a NeXt block."""
        fx_a, reg_tok_fx_a = self.attn(fx, x, reg_tok_fx, reg_tok_x, mass)
        fx = fx + fx_a * self.attn_layer_scale
        reg_tok_fx = reg_tok_fx + reg_tok_fx_a * self.attn_layer_scale

        fx = fx + self.ffn(self.ffn_ln(fx))
        reg_tok_fx = reg_tok_fx + self.ffn(self.ffn_ln(reg_tok_fx))
        return fx, reg_tok_fx

class TransolverNeXtCrossBlock(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        q_dim: int, # also for d_model
        kv_dim: int,
        attn: TransolverNeXtAttentionConfig,
        ffn: FeedForwardWithGatingConfig,
        ffn_norm_type: str,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.attn = TransolverNeXtSelfAttention(d_model=q_dim, phys_dim=phys_dim, config=attn)
        attn_layer_scale_init = layer_scale_init if layer_scale_init != 0 else 1.0
        self.attn_layer_scale = nn.Parameter(
            torch.ones(1, 1, q_dim) * attn_layer_scale_init,
            requires_grad=layer_scale_init != 0,
        )

        self.cross = TransolverNeXtCrossAttention_ShareKV(kv_dim, q_dim, phys_dim, attn)
        self.cross_layer_scale = nn.Parameter(
            torch.ones(1, 1, q_dim) * attn_layer_scale_init,
            requires_grad=layer_scale_init != 0,
        )

        self.ffn = FeedForwardWithGating(q_dim, q_dim, ffn)
        self.ffn_ln = get_normalization(ffn_norm_type, q_dim)

    def reset_parameters(self):
        """Reset the parameters of the transformer block."""
        self.attn.reset_parameters()
        self.cross.reset_parameters()
        self.ffn.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        reg_tok_qx: torch.Tensor,
        reg_tok_x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update query features and register tokens with self+cross attention."""
        y_a, reg_tok_qx_a = self.attn(qx, x, reg_tok_qx, reg_tok_x, mass_q)
        y = qx + y_a * self.attn_layer_scale
        reg_tok_qx = reg_tok_qx + reg_tok_qx_a * self.attn_layer_scale

        y_c, reg_tok_qx_c = self.cross(qx=y, kvx=kvx, x=x, reg_tok_qx=reg_tok_qx, reg_tok_x=reg_tok_x, x_kv=x_kv, mass_q=mass_q, mass_kv=mass_kv)
        y = y + y_c * self.cross_layer_scale
        reg_tok_qx = reg_tok_qx + reg_tok_qx_c * self.cross_layer_scale

        y = y + self.ffn(self.ffn_ln(y))
        reg_tok_qx = reg_tok_qx + self.ffn(self.ffn_ln(reg_tok_qx))
        return y, reg_tok_qx

@dataclass
class TransolverNeXtModelConfig:
    d_model: int
    num_layers: int
    num_registers: int
    # lifting part
    layer_scale_init: float
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    # project part
    project: MultiLayerFeedForwardConfig
    # TransolverNeXt Block
    attn: TransolverNeXtAttentionConfig
    ffn: FeedForwardWithGatingConfig
    ffn_norm_type: str = "layernorm"


class TransolverNeXtModel(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        func_dim: int,
        out_dim: int,
        config: TransolverNeXtModelConfig,
    ) -> None:
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = func_dim + self.embed_position.output_channels
        d_model = config.d_model
        num_layers = config.num_layers
        self.d_model = d_model
        self.func_dim = func_dim
        self.phys_dim = phys_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.lifting = MultiLayerFeedForward(self.total_in_dim, d_model, config.lifting)
        self.project = MultiLayerFeedForward(d_model, out_dim, config.project)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransolverNeXtBlock(
                    d_model=d_model,
                    phys_dim=phys_dim,
                    attn=config.attn,
                    ffn=config.ffn,
                    ffn_norm_type=config.ffn_norm_type,
                    layer_scale_init=config.layer_scale_init,
                )
            )

        self.register_tokens = nn.Parameter(torch.randn(1, config.num_registers, d_model))
        self.register_tokens_x = nn.Parameter(
            torch.randn(1, config.num_registers, phys_dim),
            # TODO: It is a little bit tricky, if RoPE is not enabled, this parameter will not
            # contribute to loss computation, and DDP will fail.
            requires_grad=config.attn.enable_rope,
        )

    def reset_parameters(self):
        """Reset the parameters of the TransolverModel."""
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        return_register_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the TransolverModel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., phys_dim).
            fx (torch.Tensor | None): Functional values at the input points.

        Returns:
            torch.Tensor: Output tensor after applying the operator.
        """
        if mass is None:
            m = torch.ones_like(x[..., :1])
        else:
            m = mass / torch.mean(mass, dim=1, keepdim=True)

        B = x.shape[0]
        regtok_fx = self.register_tokens.expand(B, -1, -1)
        regtok_x = self.register_tokens_x.expand(B, -1, -1).tanh()

        if regtok_fx.shape[-1] != self.d_model:
            raise ValueError(f"Register token feature dim mismatch: expected {self.d_model}, got {regtok_fx.shape[-1]}")
        if regtok_x.shape[-1] != self.phys_dim:
            raise ValueError(f"Register token phys dim mismatch: expected {self.phys_dim}, got {regtok_x.shape[-1]}")

        x_hidden = self.forward_lift(x, fx)
        # print(f"0 x_hidden.norm: {x_hidden.norm().item()} std={x_hidden.std(dim=1).mean().item()}")
        # print(f"0 regtok_fx.norm: {regtok_fx.norm().item()} std={regtok_fx.std(dim=1).mean().item()}")
        for block in self.blocks:
            x_hidden, regtok_fx = block(x, x_hidden, regtok_fx, regtok_x, m)
            # print(f"x_hidden.norm: {x_hidden.norm().item()} std={x_hidden.std(dim=1).mean().item()}")
            # print(f"regtok_fx.norm: {regtok_fx.norm().item()} std={regtok_fx.std(dim=1).mean().item()}")
        y = self.forward_project(x_hidden)
        return y if not return_register_tokens else (y, regtok_fx)

    def forward_lift(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lift the input tensor to a higher-dimensional space."""
        pe = self.embed_position(x)
        out: List[torch.Tensor] = [pe]
        if fx is not None:
            out.append(fx)
            assert fx.shape[-1] == self.func_dim, "Functional dimension mismatch"
        else:
            assert self.func_dim == 0, "Functional input is required but not provided."

        return self.lifting(torch.cat(out, dim=-1))

    def forward_project(
        self,
        out: torch.Tensor,
    ):
        """Output the final result."""
        return self.project(out)


class TransolverNeXtSolverModel(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        q_dim: int,  # p-basis
        kv_dim: int,  # q-basis
        config: TransolverNeXtModelConfig,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = q_dim + self.embed_position.output_channels
        d_model = config.d_model
        num_layers = config.num_layers

        self.d_model = d_model
        self.func_dim = q_dim
        self.phys_dim = phys_dim
        self.num_layers = num_layers

        self.output_norm = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.lifting = MultiLayerFeedForward(self.total_in_dim, d_model, config.lifting)
        out_dim = out_dim or kv_dim
        self.project = MultiLayerFeedForward(d_model, out_dim, config.project)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransolverNeXtCrossBlock(
                    q_dim=d_model,
                    kv_dim=kv_dim,
                    phys_dim=phys_dim,
                    attn=config.attn,
                    ffn=config.ffn,
                    ffn_norm_type=config.ffn_norm_type,
                    layer_scale_init=config.layer_scale_init,
                )
            )

        # register tokens (feature and coordinate)
        self.register_tokens = nn.Parameter(torch.randn(1, config.num_registers, d_model))
        self.register_tokens_x = nn.Parameter(torch.randn(1, config.num_registers, phys_dim))

    def reset_parameters(self):
        """Reset the parameters of the TransolverNeXtSolverModel."""
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()  # type: ignore

    def forward_lift(self, x: torch.Tensor, qx: torch.Tensor) -> torch.Tensor:
        """Lift concatenated positional embedding and q-basis features."""
        pe = self.embed_position(x)
        out: List[torch.Tensor] = [pe, qx]
        return self.lifting(torch.cat(out, dim=-1))  # B, ..., d_model

    def forward_project_pq(
        self,
        kvx: torch.Tensor,
        px: torch.Tensor,
        rhs: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Project p-basis to q-basis coefficients using Petrov–Galerkin formulation."""
        b, n, _ = kvx.shape
        rsqrt_vol = torch.rsqrt(torch.clamp_min(torch.sum(mass, dim=1, keepdim=True), min=1e-6))
        weights = torch.sqrt(mass) * rsqrt_vol
        coeff = torch.bmm((px * weights).mT, (rhs * weights))
        return torch.sigmoid(self.output_norm) * coeff

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        rhs: torch.Tensor | None = None,
        x_kv: torch.Tensor | None = None,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
        return_pq_project: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the TransolverNeXtSolverModel to the input tensor.

        It ensures the output stay in qx space.
        """
        if mass_q is None:
            m = torch.ones_like(qx[..., :1])
        else:
            m = mass_q / torch.mean(mass_q, dim=1, keepdim=True)
        if mass_kv is None:
            m_kv = torch.ones_like(kvx[..., :1])
        else:
            m_kv = mass_kv / torch.mean(mass_kv, dim=1, keepdim=True)
        if x_kv is None:
            x_kv = x

        B = x.shape[0]
        regtok_fx = self.register_tokens.expand(B, -1, -1)
        regtok_x = self.register_tokens_x.expand(B, -1, -1).tanh()

        # validate register token dims
        if regtok_fx.shape[-1] != self.d_model:
            raise ValueError(f"Register token feature dim mismatch: expected {self.d_model}, got {regtok_fx.shape[-1]}")
        if regtok_x.shape[-1] != self.phys_dim:
            raise ValueError(f"Register token phys dim mismatch: expected {self.phys_dim}, got {regtok_x.shape[-1]}")

        p = self.forward_lift(x, qx)
        # p_norm = torch.linalg.norm(p, dim=-2, keepdim=True).mean()
        # p_std = torch.linalg.norm(p, dim=-1, keepdim=True).std()
        # print(f"0 p_norm: {p_norm.item()}, p_std: {p_std.item()}")
        for block in self.blocks:
            p, regtok_fx = block(x, p, kvx, regtok_fx, regtok_x, x_kv, m, m_kv)
            # p_norm = torch.linalg.norm(p, dim=-2, keepdim=True).mean()
            # p_std = torch.linalg.norm(p, dim=-1, keepdim=True).std()
            # print(f"  p_norm: {p_norm.item()}, p_std: {p_std.item()}")

        p_basis = self.project(p)
        if return_pq_project:
            assert rhs is not None, "Project pq requires rhs."
            solution = self.forward_project_pq(kvx, p_basis, rhs, m_kv)
            return solution, p_basis
        else:
            return torch.empty(1), p_basis
