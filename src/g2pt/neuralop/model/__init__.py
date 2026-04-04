from g2pt.neuralop.model.transolver_experimental import TransolverExpModel
from .transolver import TransolverModel, TransolverSolverModel, TransolverModelConfig
from .transolver_next import TransolverNeXtModel, TransolverNeXtSolverModel, TransolverNeXtModelConfig
from .transolver2 import Transolver2Model, Transolver2ModelConfig, Transolver2SolverModel
from .only_pos_embed import OnlyPositionalEmbeddingModel, OnlyPositionalEmbeddingModelConfig
from .pointnet2 import PointNet2Model, PointNet2ModelConfig
from .transolver0 import Transolver0Model, Transolver0ModelConfig


def get_model(phys_dim: int, func_dim: int, out_dim: int, config):
    if config.name == "transolver":
        return TransolverModel(phys_dim, func_dim, out_dim, config)
    elif config.name == "transolver_next":
        return TransolverNeXtModel(phys_dim, func_dim, out_dim, config)
    elif config.name == "transolver2":
        return Transolver2Model(phys_dim, func_dim, out_dim, config)
    elif config.name == "point_transformer":
        from .point_transformer import PointTransformerModel

        return PointTransformerModel(phys_dim, func_dim, out_dim, config)
    elif config.name == "only_pos_embed":
        return OnlyPositionalEmbeddingModel(phys_dim, func_dim, out_dim, config)
    elif config.name == "pointnet2":
        return PointNet2Model(phys_dim, func_dim, out_dim, config)
    elif config.name == "transolver0":
        return Transolver0Model(phys_dim, func_dim, out_dim, config)

    elif config.name == "transolver_experimental":
        return TransolverExpModel(phys_dim, func_dim, out_dim, config)
    else:
        raise ValueError(f"Unknown model name: {config.name}")


def get_sol_model(phys_dim: int, q_dim: int, kv_dim: int, config):
    if config.name == "transolver":
        return TransolverSolverModel(phys_dim, q_dim, kv_dim, config)
    elif config.name == "transolver_next":
        return TransolverNeXtSolverModel(phys_dim, q_dim, kv_dim, config)
    elif config.name == "transolver2":
        return Transolver2SolverModel(phys_dim, q_dim, kv_dim, config)
    elif config.name == "point_transformer":
        from .point_transformer import PointTransformerSolverModel

        return PointTransformerSolverModel(phys_dim, q_dim, kv_dim, config)
    else:
        raise ValueError(f"Unknown model name: {config.name}")


__all__ = [
    "TransolverModel",
    "TransolverSolverModel",
    "TransolverModelConfig",
    "TransolverNeXtModel",
    "TransolverNeXtSolverModel",
    "TransolverNeXtModelConfig",
    "Transolver2Model",
    "Transolver2ModelConfig",
    "Transolver2SolverModel",
    "OnlyPositionalEmbeddingModel",
    "OnlyPositionalEmbeddingModelConfig",
    "Transolver0Model",
    "Transolver0ModelConfig",
    "PointNet2Model",
    "PointNet2ModelConfig",
    "get_model",
    "get_sol_model",
]
