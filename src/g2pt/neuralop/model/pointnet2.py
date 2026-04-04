from dataclasses import dataclass
import torch
from torch import nn

from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import MultiLayerFeedForward, MultiLayerFeedForwardConfig
from g2pt.neuralop.layers.act import get_activation

from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class SetAbstractionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: float, r: float, act: str):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.local_nn = MLP([in_channels + 3, 64, 64, out_channels], norm='batch_norm', plain_last=False, act=act)
        self.conv = PointNetConv(self.local_nn, add_self_loops=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str):
        super().__init__()
        self.nn = MLP([in_channels + 3, 256, 512, out_channels], norm='batch_norm', plain_last=False, act=act)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), pos.size(1)))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FeaturePropagationModule(nn.Module):
    def __init__(self, k: int, in_channels: int, out_channels: int, act: str):
        super().__init__()
        self.k = k
        self.nn = MLP([in_channels, out_channels, out_channels], norm='batch_norm', plain_last=False, act=act)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        x_skip: torch.Tensor | None,
        pos_skip: torch.Tensor,
        batch_skip: torch.Tensor,
    ):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2Backbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        sa_channels: list[int],
        sa_ratios: list[float],
        sa_radii: list[float],
        fp_k: list[int],
        act: str,
    ):
        super().__init__()
        self.act = act
        self.sa1 = SetAbstractionModule(d_model, sa_channels[0], sa_ratios[0], sa_radii[0], act)
        self.sa2 = SetAbstractionModule(sa_channels[0], sa_channels[1], sa_ratios[1], sa_radii[1], act)
        self.sa3 = GlobalSAModule(sa_channels[1], sa_channels[2], act)

        self.fp3 = FeaturePropagationModule(fp_k[0], sa_channels[2] + sa_channels[1], sa_channels[1], act)
        self.fp2 = FeaturePropagationModule(fp_k[1], sa_channels[1] + sa_channels[0], sa_channels[0], act)
        self.fp1 = FeaturePropagationModule(fp_k[2], sa_channels[0] + d_model, sa_channels[0], act)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        sa3_out = self.sa3(*sa2_out)

        fp3_out = self.fp3(*sa3_out, *sa2_out)
        fp2_out = self.fp2(*fp3_out, *sa1_out)
        x, _, _ = self.fp1(*fp2_out, *sa0_out)
        return x


@dataclass
class PointNet2ModelConfig:
    d_model: int
    embed_position: EmbedPositionConfig
    lifting: MultiLayerFeedForwardConfig
    project: MultiLayerFeedForwardConfig
    act: str = "silu"
    bias: bool = True
    sa_ratios: list[float] = None  # type: ignore
    sa_radii: list[float] = None  # type: ignore
    sa_channels: list[int] = None  # type: ignore
    fp_k: list[int] = None  # type: ignore


class PointNet2Model(nn.Module):
    def __init__(self, phys_dim: int, func_dim: int, out_dim: int, config: PointNet2ModelConfig) -> None:
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = func_dim + self.embed_position.output_channels
        self.func_dim = func_dim
        self.phys_dim = phys_dim
        self.out_dim = out_dim
        self.d_model = config.d_model

        assert config.sa_channels is not None and len(config.sa_channels) == 3
        assert config.sa_ratios is not None and len(config.sa_ratios) == 2
        assert config.sa_radii is not None and len(config.sa_radii) == 2
        assert config.fp_k is not None and len(config.fp_k) == 3
        assert config.sa_channels[0] == self.d_model

        self.lifting = MultiLayerFeedForward(self.total_in_dim, self.d_model, config.lifting)
        self.project = MultiLayerFeedForward(self.d_model, out_dim, config.project)
        self.act = get_activation(config.act)

        self.model = PointNet2Backbone(
            d_model=self.d_model,
            sa_channels=config.sa_channels,
            sa_ratios=config.sa_ratios,
            sa_radii=config.sa_radii,
            fp_k=config.fp_k,
            act=config.act,
        )

    def reset_parameters(self):
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        def reset_if(module):
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.model.apply(reset_if)

    def forward(self, x: torch.Tensor, fx: torch.Tensor | None = None, mass: torch.Tensor | None = None) -> torch.Tensor:
        pe = self.embed_position(x)
        outs = [pe]
        if fx is not None:
            outs.append(fx)
            assert fx.shape[-1] == self.func_dim
        else:
            assert self.func_dim == 0
        x_hidden = self.lifting(torch.cat(outs, dim=-1))

        b, n, c = x_hidden.shape
        x_hidden_flat = x_hidden.view(b * n, c)
        pos_flat = x.view(b * n, self.phys_dim)
        batch = torch.arange(b, device=x.device).repeat_interleave(n)
        y = self.model(x_hidden_flat, pos_flat, batch).reshape(b, n, -1)
        return self.project(y)