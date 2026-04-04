from dataclasses import dataclass
from typing import List

from torch import nn
import torch

from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import MultiLayerFeedForward, MultiLayerFeedForwardConfig


@dataclass
class OnlyPositionalEmbeddingModelConfig:
    d_model: int
    # lifting part
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    # project part
    project: MultiLayerFeedForwardConfig


class OnlyPositionalEmbeddingModel(nn.Module):
    def __init__(self, phys_dim: int, func_dim: int, out_dim: int, config: OnlyPositionalEmbeddingModelConfig):
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = func_dim + self.embed_position.output_channels
        d_model = config.d_model
        self.d_model = d_model
        self.func_dim = func_dim
        self.phys_dim = phys_dim
        self.out_dim = out_dim

        self.lifting = MultiLayerFeedForward(self.total_in_dim, d_model, config.lifting)
        self.project = MultiLayerFeedForward(d_model, out_dim, config.project)

    def reset_parameters(self):
        self.lifting.reset_parameters()
        self.project.reset_parameters()

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

    def forward(
        self, x: torch.Tensor, fx: torch.Tensor | None = None, mass: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply the positional embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., phys_dim).
            fx (torch.Tensor | None): Functional values at the input points.

        Returns:
            torch.Tensor: Output tensor after applying the operator.
        """

        x_hidden = self.forward_lift(x, fx)  # B, ..., d_model
        y = self.forward_project(x_hidden)
        return y
