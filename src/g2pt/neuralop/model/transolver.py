from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from g2pt.neuralop.layers.attn.transolver import (
    TransolverAttentionConfig,
    TransolverCrossAttention_ShareKV,
    TransolverSelfAttention,
)
from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import (
    FeedForwardWithGating,
    FeedForwardWithGatingConfig,
    MultiLayerFeedForward,
    MultiLayerFeedForwardConfig,
)
from g2pt.neuralop.layers.norms import get_normalization


class TransolverBlock(nn.Module):
    def __init__(
        self,
        # attention
        d_model: int,
        phys_dim: int,
        attn: TransolverAttentionConfig,
        ffn: FeedForwardWithGatingConfig,
        ffn_norm_type: str,
    ) -> None:
        super().__init__()
        self.attn = TransolverSelfAttention(d_model, phys_dim, attn)
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
        mass: torch.Tensor | None = None
    ) -> torch.Tensor:
        # TransloverSelfAttention has normalization inside
        fx = fx + self.attn(fx, x, mass)  # B, nPoint, C
        fx = fx + self.ffn(self.ffn_ln(fx))  # B, nPoint, C
        return fx

class TransolverCrossBlock(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        q_dim: int, # also for d_model
        kv_dim: int,
        attn: TransolverAttentionConfig,
        ffn: FeedForwardWithGatingConfig,
        ffn_norm_type: str,
    ) -> None:
        super().__init__()
        self.attn = TransolverSelfAttention(d_model=q_dim, phys_dim=phys_dim, config=attn)

        self.cross = TransolverCrossAttention_ShareKV(kv_dim, q_dim, phys_dim, q_dim, attn)
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
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TransloverSelfAttention has normalization inside
        y = qx + self.attn(qx, x, mass_q)  # B, nPoint, C
        y = y + self.cross(qx=y, kvx=kvx, x=x, mass_q=mass_q, mass_kv=mass_kv)  # B, nPoint, C
        y = y + self.ffn(self.ffn_ln(y))  # B, nPoint, C
        return y

@dataclass
class TransolverModelConfig:
    d_model: int
    num_layers: int
    # lifting part
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    # project part
    project: MultiLayerFeedForwardConfig
    # Transolver Block
    attn: TransolverAttentionConfig
    ffn: FeedForwardWithGatingConfig
    ffn_norm_type: str = "layernorm"


class TransolverModel(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        func_dim: int,
        out_dim: int,
        config: TransolverModelConfig,
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
                TransolverBlock(
                    d_model=d_model,
                    phys_dim=phys_dim,
                    attn=config.attn,
                    ffn=config.ffn,
                    ffn_norm_type=config.ffn_norm_type,
                )
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
        mass: torch.Tensor | None = None
    ) -> torch.Tensor:
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

        x_hidden = self.forward_lift(x, fx)  # B, ..., d_model
        for block in self.blocks:
            x_hidden = block(x, x_hidden, m)  # B, nPoint, d_model
        y = self.forward_project(x_hidden)
        return y

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


class TransolverSolverModel(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        q_dim: int,  # p-basis
        kv_dim: int,  # q-basis
        config: TransolverModelConfig,
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
        self.project = MultiLayerFeedForward(d_model, kv_dim, config.project)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransolverCrossBlock(
                    q_dim=d_model,
                    kv_dim=kv_dim,
                    phys_dim=phys_dim,
                    attn=config.attn,
                    ffn=config.ffn,
                    ffn_norm_type=config.ffn_norm_type,
                )
            )

    def reset_parameters(self):
        """Reset the parameters of the TransolverSolverModel."""
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()  # type: ignore

    def forward_lift(self, x, qx):
        pe = self.embed_position(x)
        out: List[torch.Tensor] = [pe, qx]
        return self.lifting(torch.cat(out, dim=-1)) # B, ..., d_model

    def forward_project_pq(self, kvx, px, rhs, mass):
        # Petrov-Galerkin: P-Q basis
        # P-basis: px
        # Q-basis: kvx
        # Output in Q-basis
        b, n, c = kvx.shape
        # rsqrt_vol = 1 / math.sqrt(n) # rely on mean(mass) == 1
        rsqrt_vol = torch.rsqrt(torch.clamp_min(torch.sum(mass, dim = 1, keepdim=True), min=1e-6)) # (b, 1, 1)
        weights = torch.sqrt(mass) * rsqrt_vol # (b, n, 1)
        # coeff = torch.einsum("bni,bnj,bn->bij", px * rsqrt_vol, rhs * rsqrt_vol, mass.squeeze(-1)) # (b, c, nsamples)
        coeff = torch.bmm((px * weights).mT, (rhs * weights)) # (b, c, nsamples)
        return torch.sigmoid(self.output_norm) * coeff   # (b, c, nsamples)

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        rhs: torch.Tensor,
        mass_q: torch.Tensor | None = None,
        mass_kv: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the TransolverSolverModel to the input tensor.

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

        # do not preprocess qx.
        p = self.forward_lift(x, qx)  # B, ..., d_model
        for block in self.blocks:
            # x, qx, kvx, mass_q, mass_kv
            p = block(x, p, kvx, m, m_kv)  # B, nPoint, d_model

        p_basis = self.project(p)
        solution = self.forward_project_pq(kvx, p_basis, rhs, m_kv)
        return solution, p_basis
