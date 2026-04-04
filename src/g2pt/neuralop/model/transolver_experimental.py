from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from g2pt.neuralop.layers.attn.transolver_experimental import (
    TransolverExpAttentionConfig,
    TransolverExpSelfAttention,
)
from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import (
    FeedForwardWithGating,
    FeedForwardWithGatingConfig,
    MultiLayerFeedForward,
    MultiLayerFeedForwardConfig,
)
from g2pt.neuralop.layers.norms import get_normalization


class TransolverExpBlock(nn.Module):
    def __init__(
        self,
        # attention
        d_model: int,
        phys_dim: int,
        attn: TransolverExpAttentionConfig,
        ffn: FeedForwardWithGatingConfig,
        ffn_norm_type: str,
        layer_scale_init,
    ) -> None:
        super().__init__()
        self.attn_norm = get_normalization(attn.norm_type, d_model)
        self.attn = TransolverExpSelfAttention(d_model, phys_dim, attn)
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
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update point features and register tokens in a NeXt block."""
        if self.training:
            fx = fx + self.attn(self.attn_norm(fx), x, mass)
            fx = fx + self.ffn(self.ffn_ln(fx))
        else:
            fx += self.attn(self.attn_norm(fx), x, mass)
            fx += self.ffn(self.ffn_ln(fx))
        return fx

@dataclass
class TransolverExpModelConfig:
    d_model: int
    num_layers: int
    num_registers: int
    # lifting part
    layer_scale_init: float
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    # project part
    project: MultiLayerFeedForwardConfig
    # TransolverExp Block
    attn: TransolverExpAttentionConfig
    ffn: FeedForwardWithGatingConfig
    ffn_norm_type: str = "layernorm"


class TransolverExpModel(nn.Module):
    def __init__(
        self,
        phys_dim: int,
        func_dim: int,
        out_dim: int,
        config: TransolverExpModelConfig,
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
                TransolverExpBlock(
                    d_model=d_model,
                    phys_dim=phys_dim,
                    attn=config.attn,
                    ffn=config.ffn,
                    ffn_norm_type=config.ffn_norm_type,
                    layer_scale_init=config.layer_scale_init,
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
        mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
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

        x_hidden = self.forward_lift(x, fx)
        # print(f"0 x_hidden.norm: {x_hidden.norm().item()} std={x_hidden.std(dim=1).mean().item()}")
        for block in self.blocks:
            x_hidden = block(x, x_hidden, m)
            # print(f"x_hidden.norm: {x_hidden.norm().item()} std={x_hidden.std(dim=1).mean().item()}")
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