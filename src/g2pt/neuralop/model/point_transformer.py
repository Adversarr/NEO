from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from g2pt.neuralop.layers.act import get_activation
from g2pt.neuralop.layers.embed_position import EmbedPositionConfig, get_embed_position
from g2pt.neuralop.layers.mlps import MultiLayerFeedForward, MultiLayerFeedForwardConfig

from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn, knn_graph, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

from g2pt.neuralop.layers.norms import get_normalization

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

def _make_linear_norm_act(in_channels: int, out_channels: int, bias: bool, norm_type: str, act: str):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels, bias=bias),
        get_normalization(norm_type, out_channels),
        get_activation(act),
    )

@dataclass
class PointTransformerConfig:
    """Configuration for PointTransformer models.

    - d_model: hidden feature dimension per point. Can be a list of integers to specify different dimensions for each layer.
    - num_layers: number of point transformer blocks.
    - embed_position: positional embedding configuration.
    - lifting: MLP config to lift input features to `d_model`.
    - project: MLP config to project hidden features to output dimension.
    - k: number of nearest neighbors used in knn graph.
    - pos_mlp_hidden: hidden size of `pos_nn` and `attn_nn` inside PointTransformerConv.

    TODO: Consider supporting alternative neighbor graphs (e.g., radius graph) if needed.
    Note: Currently only `knn_graph` is implemented to mirror example usage.
    """
    d_model: int | list[int]
    num_layers: int
    norm_type: str
    act: str
    bias: bool
    lifting: MultiLayerFeedForwardConfig
    project: MultiLayerFeedForwardConfig
    embed_position: EmbedPositionConfig
    k: int = 16
    pos_mlp_hidden: int = 64


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """

    def __init__(self, in_channels, out_channels, norm_type: str, act: str, bias: bool, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = _make_linear_norm_act(in_channels, out_channels, bias, norm_type, act)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce="max")

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class TransformerBlock(torch.nn.Module):
    """Point Transformer block with linear projections and PointTransformerConv.
    
    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        phys_dim: Coordinate dimension for positional encoding.
        hidden_mlp_dim: Hidden dimension for MLPs in PointTransformerConv.
        norm_type: Type of normalization to use in MLPs.
    """
    def __init__(self, in_channels, out_channels, phys_dim, hidden_mlp_dim, norm_type: str, act: str):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.pos_nn = MLP([phys_dim, hidden_mlp_dim, out_channels], norm=None, plain_last=False, act=act)
        self.attn_nn = MLP([out_channels, hidden_mlp_dim, out_channels], norm=None, plain_last=False, act=act)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)
        self.pre_attn_norm = get_normalization(norm_type, in_channels)
        self.lin_in_act = get_activation(act)
        self.lin_out_act = get_activation(act)

    def forward(self, x, pos, edge_index):
        """Apply the transformer block.
        
        Args:
            x: Input features (N, in_channels).
            pos: Node positions (N, phys_dim).
            edge_index: Graph connectivity (2, E).
            
        Returns:
            Transformed features (N, out_channels).
        """
        x = self.lin_in_act(self.lin_in(x))
        x = self.pre_attn_norm(x)
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out_act(self.lin_out(x))
        return x


class TransitionUp(torch.nn.Module):
    """Reduce features dimensionality and interpolate back to higher
    resolution and cardinality.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool, norm_type: str, act: str):
        super().__init__()
        self.mlp_sub = _make_linear_norm_act(in_channels, out_channels, bias, norm_type, act)
        self.mlp = _make_linear_norm_act(out_channels, out_channels, bias, norm_type, act)
        self.lin_out_act = get_activation(act)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class PointTransformerBackbone(torch.nn.Module):
    def __init__(self, dim_model, phys_dim, hidden_mlp_dim, k, norm_type: str, act: str, bias: bool):
        super().__init__()
        self.k = k

        # Create transformer blocks for the backbone architecture
        def make_block(in_channels, out_channels):
            return TransformerBlock(in_channels, out_channels, phys_dim, hidden_mlp_dim, norm_type, act)

        self.transformer_input = make_block(in_channels=dim_model[0], out_channels=dim_model[0])

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    bias=bias,
                    norm_type=norm_type,
                    act=act,
                    k=self.k,
                )
            )

            self.transformers_down.append(make_block(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(
                    in_channels=dim_model[i + 1],
                    out_channels=dim_model[i],
                    bias=bias,
                    norm_type=norm_type,
                    act=act,
                )
            )

            self.transformers_up.append(make_block(in_channels=dim_model[i], out_channels=dim_model[i]))

        # Summit layers for the highest resolution point cloud
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None, plain_last=False)

        self.transformer_summit = make_block(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

    def reset_parameters(self):
        """Reset parameters of all submodules that have reset_parameters method."""
        def reset_if_provided(module):
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.apply(reset_if_provided)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        out_x = []
        out_pos = []
        out_batch = []

        # First transformer block
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # Backbone down: reduce cardinality and augment dimensionality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # Backbone up: augment cardinality and reduce dimensionality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](
                x=out_x[-i - 2],
                x_sub=x,
                pos=out_pos[-i - 2],
                pos_sub=out_pos[-i - 1],
                batch_sub=out_batch[-i - 1],
                batch=out_batch[-i - 2],
            )

            edge_index = knn_graph(out_pos[-i - 2], k=self.k, batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        return x


class PointTransformerModel(nn.Module):
    """PointTransformer operator model following the Transolver-style scaffolding.

    This model lifts inputs, applies a stack of PointTransformer blocks,
    and projects to the desired output dimension.

    Args:
        phys_dim: coordinate dimension.
        func_dim: input functional feature dimension.
        out_dim: output feature dimension.
        config: PointTransformerConfig.
    """

    def __init__(
        self,
        phys_dim: int,
        func_dim: int,
        out_dim: int,
        config: PointTransformerConfig,
    ) -> None:
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = func_dim + self.embed_position.output_channels

        d_model = config.d_model
        self.func_dim = func_dim
        self.phys_dim = phys_dim
        self.out_dim = out_dim
        self.num_layers = config.num_layers
        if isinstance(d_model, int):
            self.d_model = [d_model] * self.num_layers
        elif isinstance(d_model, list):
            assert len(d_model) == self.num_layers, "d_model must have same length as num_layers"
            self.d_model: list[int] = d_model
        else:
            raise ValueError("d_model must be an int or a list of ints")

        self.lifting = MultiLayerFeedForward(self.total_in_dim, self.d_model[0], config.lifting)
        self.project = MultiLayerFeedForward(self.d_model[0], out_dim, config.project)

        self.model = PointTransformerBackbone(
            dim_model=self.d_model,
            phys_dim=phys_dim,
            hidden_mlp_dim=config.pos_mlp_hidden,
            k=config.k,
            bias=config.bias,
            norm_type=config.norm_type,
            act=config.act,
        )

    def reset_parameters(self):
        """Reset parameters of the model components."""
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        self.model.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        fx: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the PointTransformerModel.

        Args:
            x: coordinates (B, N, phys_dim).
            fx: optional functional features (B, N, func_dim). If None, func_dim must be 0.
            mass: optional per-point weights used for normalization (not used in blocks).

        Returns:
            Output features (B, N, out_dim).
        """
        x_hidden = self.forward_lift(x, fx)
        # (bn, c)
        b, n, c = x_hidden.shape
        x_hidden_flat = x_hidden.view(b * n, c)
        pos_flat = x.view(b * n, self.phys_dim)
        batch = torch.arange(b, device=x.device).repeat_interleave(n) # (B * N,)
        y = self.model(x_hidden_flat, pos_flat, batch).reshape(b, n, -1)
        return self.forward_project(y)

    def forward_lift(self, x: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        """Lift coordinates and features into the model hidden space.

        Concatenates positional embedding and functional features, then applies `lifting`.

        Args:
            x: coordinates (B, N, phys_dim).
            fx: optional functional features (B, N, func_dim).

        Returns:
            Hidden features (B, N, d_model).
        """
        pe = self.embed_position(x)
        out: List[torch.Tensor] = [pe]
        if fx is not None:
            out.append(fx)
            assert fx.shape[-1] == self.func_dim, "Functional dimension mismatch"
        else:
            assert self.func_dim == 0, "Functional input is required but not provided."
        return self.lifting(torch.cat(out, dim=-1))

    def forward_project(self, out: torch.Tensor) -> torch.Tensor:
        """Project hidden features to the final output dimension."""
        return self.project(out)


class PointTransformerSolverModel(nn.Module):
    """Solver model using PointTransformer blocks with Petrov–Galerkin projection.

    Follows the TransolverSolverModel style: lift q-basis, process with blocks,
    project to k-v basis, then compute solution coefficients via weighted inner product.

    Args:
        phys_dim: coordinate dimension.
        q_dim: dimension of p-basis (input basis features).
        kv_dim: dimension of q-basis (output basis features).
        config: PointTransformerConfig.
    """

    def __init__(
        self,
        phys_dim: int,
        q_dim: int,  # p-basis
        kv_dim: int,  # q-basis
        config: PointTransformerConfig,
    ) -> None:
        super().__init__()
        self.embed_position = get_embed_position(phys_dim, config.embed_position)
        self.total_in_dim = q_dim + kv_dim + self.embed_position.output_channels
        self.func_dim = q_dim
        self.phys_dim = phys_dim
        self.num_layers = config.num_layers

        d_model = config.d_model
        self.phys_dim = phys_dim
        if isinstance(d_model, int):
            self.d_model = [d_model] * self.num_layers
        elif isinstance(d_model, list):
            assert len(d_model) == self.num_layers, "d_model must have same length as num_layers"
            self.d_model: list[int] = d_model
        else:
            raise ValueError("d_model must be an int or a list of ints")

        self.lifting = MultiLayerFeedForward(self.total_in_dim, self.d_model[0], config.lifting)
        self.project = MultiLayerFeedForward(self.d_model[0], kv_dim, config.project)
        self.output_norm = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.model = PointTransformerBackbone(
            dim_model=self.d_model,
            phys_dim=phys_dim,
            hidden_mlp_dim=config.pos_mlp_hidden,
            k=config.k,
            bias=config.bias,
            norm_type=config.norm_type,
            act=config.act,
        )


    def reset_parameters(self):
        """Reset the parameters of the solver model."""
        self.lifting.reset_parameters()
        self.project.reset_parameters()
        self.model.reset_parameters()

    def forward_lift(self, x: torch.Tensor, qx: torch.Tensor, kvx: torch.Tensor) -> torch.Tensor:
        """Lift coordinates and p-basis features into hidden space."""
        pe = self.embed_position(x)
        out: List[torch.Tensor] = [pe, qx, kvx]
        return self.lifting(torch.cat(out, dim=-1))

    def forward_project_pq(
        self,
        kvx: torch.Tensor,
        px: torch.Tensor,
        rhs: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Petrov–Galerkin projection coefficients in q-basis.

        Args:
            kvx: q-basis functions evaluated at points (B, N, kv_dim).
            px: p-basis functions evaluated at points (B, N, kv_dim) after projection.
            rhs: right-hand side values at points (B, N, nsamples).
            mass: normalized mass/weights per point (B, N, 1), mean-one per batch.

        Returns:
            Coefficients in q-basis (B, kv_dim, nsamples).

        Note:
            Mirrors Transolver’s projection; `output_norm` scales the coefficients.
        """
        # rely on mean(mass) == 1 via normalization in forward
        rsqrt_vol = torch.rsqrt(torch.clamp_min(torch.sum(mass, dim=1, keepdim=True), min=1e-6))
        weights = torch.sqrt(mass) * rsqrt_vol  # (B, N, 1)
        coeff = torch.bmm((px * weights).mT, (rhs * weights))  # (B, kv_dim, nsamples)
        return torch.sigmoid(self.output_norm) * coeff

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        kvx: torch.Tensor,
        rhs: torch.Tensor,
        mass: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the solver over batched point clouds.

        Ensures the output stays in qx space by projecting to kv_dim and
        performing the Petrov–Galerkin computation.

        Args:
            x: coordinates (B, N, phys_dim).
            qx: input p-basis features (B, N, q_dim).
            kvx: q-basis features (B, N, kv_dim).
            rhs: right-hand side values at points (B, N, nsamples).
            mass: optional per-point weights.

        Returns:
            Tuple of (solution coefficients in q-basis, projected p-basis features).

        NOTE: kvx is integrated as input features in the lifting layer.
        """
        if mass is None:
            m = torch.ones_like(qx[..., :1])
        else:
            m = mass / torch.mean(mass, dim=1, keepdim=True)

        # do not preprocess qx.
        p = self.forward_lift(x, qx, kvx)
        b, n, d = p.shape

        x_flat = x.view(b * n, -1) # (B * N, phys_dim)
        p_flat = p.view(b * n, -1) # (B * N, d_model)
        batch = torch.arange(b, device=x.device).repeat_interleave(n) # (B * N,)
        p = self.model(p_flat, x_flat, batch)
        p = p.view(b, n, -1)

        p_basis = self.project(p)
        solution = self.forward_project_pq(kvx, p_basis, rhs, m)
        return solution, p_basis
