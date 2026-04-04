from importlib.util import find_spec

import torch
import torch.nn as nn

from g2pt.neuralop.layers.act import get_activation
from dataclasses import dataclass

from g2pt.utils.common import roundup

class FeedForwardWithGating_Eager(nn.Module):
    """
    Eager implementation of feed-forward layer with gating mechanism.
    
    This class applies an activation function to the first input tensor and
    multiplies the result with the second input tensor. This is a fundamental
    component in gated feed-forward networks.
    
    Args:
        act (str, optional): Activation function to use. Defaults to "gelu".
    """
    
    def __init__(
        self,
        act="gelu",
    ):
        super(FeedForwardWithGating_Eager, self).__init__()
        self.act = get_activation(act)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply the gating mechanism to the input tensors.
        
        Args:
            a (torch.Tensor): Gate tensor to which activation is applied.
            b (torch.Tensor): Value tensor to be gated.
            
        Returns:
            torch.Tensor: Result of activation(a) * b.
        """
        return self.act(a) * b


class FeedForwardWithGating_Wrapper(nn.Module):
    """
    Wrapper class for external implementations of gated feed-forward operations.
    
    This class wraps external implementations (like those from liger_kernel)
    to provide a consistent interface for gated feed-forward operations.
    
    Args:
        impl: External implementation object that provides an apply method.
    """
    
    def __init__(self, impl):
        super(FeedForwardWithGating_Wrapper, self).__init__()
        self.impl = impl

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply the wrapped implementation to the input tensors.
        
        Args:
            a (torch.Tensor): Gate tensor.
            b (torch.Tensor): Value tensor.
            
        Returns:
            torch.Tensor: Result of the wrapped implementation's apply method.
        """
        return self.impl.apply(a, b)

@dataclass
class FeedForwardWithGatingConfig:
    hidden_features: int | None = None
    mlp_ratio: float = 2.6666666
    bias: bool = False
    act: str = "gelu"

class FeedForwardWithGating(nn.Module):
    """
    Feed-forward layer with gating mechanism that optionally uses optimized kernels.
    
    This class implements a gated feed-forward layer that can use either the
    eager implementation or optimized kernels from liger_kernel when available.
    The gating mechanism computes y = down_proj(act(gate_proj(x)) * up_proj(x)).
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_features (int): Number of hidden features in the intermediate layers.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        act (str, optional): Activation function to use. Defaults to "gelu".
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: FeedForwardWithGatingConfig,
    ):
        super(FeedForwardWithGating, self).__init__()
        hidden_features = config.hidden_features
        if hidden_features is None:
            hidden_features = roundup(in_features * config.mlp_ratio, 16)

        bias = config.bias
        act = config.act

        has_liger = find_spec("liger_kernel") is not None
        if has_liger and config.act == "gelu":
            from liger_kernel.ops.geglu import LigerGELUMulFunction

            self.impl = FeedForwardWithGating_Wrapper(LigerGELUMulFunction)
        elif has_liger and act == "silu":
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction

            self.impl = FeedForwardWithGating_Wrapper(LigerSiLUMulFunction)
        else:
            self.impl = FeedForwardWithGating_Eager(act=act)

        self.up_proj = nn.Linear(in_features, hidden_features, bias)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias)

    def reset_parameters(self):
        """Reset the parameters of the feed-forward layer."""
        self.up_proj.reset_parameters()
        self.gate_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward layer with gating mechanism.

        y = down_proj(act(gate_proj(x)) * up_proj(x))

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Transformed tensor with the corresponding output shape.
        """
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        x_down = self.down_proj(self.impl(x_gate, x_up))
        return x_down

@dataclass
class MultiLayerFeedForwardConfig:
    hidden_features: int | None = None
    num_hidden_layers: int = 0
    bias: bool = False
    act: str = "gelu"

class MultiLayerFeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: MultiLayerFeedForwardConfig,
    ):
        """
        Initialize a multi-layer feed-forward neural network.

        - If `num_hidden_layers` is 1, it behaves like a simple feed-forward layer.
        - If `num_hidden_layers` is greater than 1, it creates a multi-layer structure
          with `num_hidden_layers - 1` hidden layers, each with `hidden_features` units.
        - If `num_hidden_layers` is 0, it creates a simple linear transformation.
        - If `num_hidden_layers` is less than 0, it bypasses all linear transformations,
          requiring in_features to equal out_features, effectively creating an identity layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            hidden_features (int): Number of hidden features in each layer.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
            act (str, optional): Activation function to use. Defaults to "gelu".
        """

        super(MultiLayerFeedForward, self).__init__()
        hidden_features = config.hidden_features
        num_hidden_layers = config.num_hidden_layers
        bias = config.bias
        act = config.act

        if num_hidden_layers > 0:
            assert hidden_features is not None, "hidden_features must be specified for multi-layer feed-forward"
        else:
            hidden_features = out_features

        # TODO: Consider allowing default hidden_features when num_hidden_layers > 0
        # Reason: hidden_features could be derived from in_features using a ratio.

        hidden_features = hidden_features or out_features
        if num_hidden_layers >= 0:
            self.up = nn.Linear(in_features, hidden_features, bias=bias)
        else:
            self.up = nn.Identity()
            assert in_features == out_features, "in_features must match out_features when num_hidden_layers < 0"


        if num_hidden_layers <= 0:
            self.down = self.body = None

        else:
            self.down = nn.Sequential(
                get_activation(act),
                nn.Linear(hidden_features, out_features, bias=bias),
            )
            self.body = nn.ModuleList(
                [
                    nn.Sequential(
                        get_activation(act),
                        nn.Linear(hidden_features, hidden_features, bias=bias),
                    )
                    for _ in range(num_hidden_layers - 1)
                ]
            )
        self.reset_parameters()
        self.num_hidden_layers = num_hidden_layers

    def reset_parameters(self):
        """Reset the parameters of the multi-layer feed-forward layer."""
        if isinstance(self.up, nn.Linear):
            self.up.reset_parameters()  # type: ignore
        if self.down is not None and self.body is not None:
            self.down[1].reset_parameters()  # type: ignore
            for layer in self.body:
                layer[1].reset_parameters()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the multi-layer feed-forward layer.

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Transformed tensor with the corresponding output shape.
        """
        x = self.up(x)
        if self.down is not None and self.body is not None:
            for layer in self.body:
                x = layer(x)
            return self.down(x)
        else:
            return x

__all__ = [
    "FeedForwardWithGating",
    "FeedForwardWithGatingConfig",
    "MultiLayerFeedForward",
    "MultiLayerFeedForwardConfig",
]
