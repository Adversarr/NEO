from torch import nn


def get_activation(activation: str) -> nn.Module:
    """
    Returns the activation function based on the input string.

    Args:
        activation (str): Name of the activation function.

    Returns:
        nn.Module: Corresponding activation function.

    Raises:
        TypeError: If the activation is not a string.
        ValueError: If the activation function is not supported.
    """
    if not isinstance(activation, str):
        raise TypeError(f"Activation must be a string, got {type(activation)}")
    
    activation_map = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid
    }
    activation = activation.lower()
    if activation in activation_map:
        return activation_map[activation]()
    raise ValueError(f"Unsupported activation function: {activation}.")
