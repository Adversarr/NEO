from torch import nn
from importlib.util import find_spec

def get_normalization(norm: str, channels: int) -> nn.Module:
    """
    Get the normalization layer based on the specified type.

    Args:
        norm (str): Type of normalization.
        channels (int): Number of channels for the normalization layer.
    Returns:
        nn.Module: Normalization layer.
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    
    has_liger = find_spec("liger_kernel") is not None
    norm = norm.lower()
    if norm == "layer" or norm == "layernorm":
        if has_liger:
            from liger_kernel.transformers import LigerLayerNorm
            return LigerLayerNorm(channels) # TODO: LigerLayerNorm has bias=False as default.
        else:
            return nn.LayerNorm(channels, bias=False)
    elif norm == "rmsnorm" or norm == "rms":
        if has_liger:
            from liger_kernel.transformers import LigerRMSNorm
            return LigerRMSNorm(channels)
        else:
            return nn.RMSNorm(channels)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}")
