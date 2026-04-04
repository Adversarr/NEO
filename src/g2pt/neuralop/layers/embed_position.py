import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class EmbedPositionConfig:
    name: str = "sinusoidal"
    num_freqs: int = 6
    min_freqs_exp: float = -1
    max_freqs_exp: float | None = None


class BaseEmbedPosition(nn.Module):
    def __init__(self, phys_dim: int = 0):
        super().__init__()
        self.phys_dim = phys_dim

    @property
    def output_channels(self):
        """Estimate the number of output channels based on the input dimensions."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed the position of each element in the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., phys_dim).

        Returns:
            torch.Tensor: Output tensor with positional embeddings added.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class DirectEmbedPosition(BaseEmbedPosition):
    def __init__(self, phys_dim: int):
        super().__init__(phys_dim=phys_dim)

    @property
    def output_channels(self) -> int:
        return self.phys_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SinusoidalEmbedPosition(BaseEmbedPosition):
    def __init__(
        self,
        phys_dim: int,
        num_freqs: int = 6,
        min_freqs_exp: float = -1,
        max_freqs_exp: float | None = None,
    ):
        super().__init__(phys_dim=phys_dim)
        # Validate num_freqs parameter
        if num_freqs <= 0:
            raise ValueError(f"num_freqs must be positive, got {num_freqs}")

        # [phys_dim, ]
        self.num_freqs = num_freqs
        self.phys_dim = phys_dim
        self.max_freqs_exp = max_freqs_exp or float(num_freqs - 1)
        self.min_freqs_exp = min_freqs_exp
        self.register_buffer(
            "freqs",
            torch.linspace(self.min_freqs_exp, self.max_freqs_exp, num_freqs, dtype=torch.float32),
        )

    @property
    def output_channels(self) -> int:
        """
        Output for each channel:
            PE(pos, 2i) = sin(pos * min_freqs), PE(pos, 2i+1) = cos(pos * max_freqs)
            d_model = self.num_freqs
        """
        return self.phys_dim * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed the position of each element in the input tensor using fixed grid sinusoidal embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, phys_dim).

        Returns:
            torch.Tensor: Output tensor with positional embeddings added.
        """
        # Input validation
        if x is None:
            raise ValueError("Input tensor cannot be None")
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {x.dim()}")
        if x.size(-1) != self.phys_dim:
            raise ValueError(f"Last dimension of input must match phys_dim ({self.phys_dim}), got {x.size(-1)}")

        # Use dynamic autocast based on device type
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # output = [x[..., [i]] * self.freqs.view(1, 1, -1).exp2() * torch.pi for i in range(self.phys_dim)]
            # sines = [torch.sin(output[i]) for i in range(self.phys_dim)]
            # cosines = [torch.cos(output[i]) for i in range(self.phys_dim)]
            # return torch.cat(sines + cosines, dim=-1)
            # [batch_size, num_tokens, phys_dim * num_freqs]
            pos_freqs = (x.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1).exp2() * torch.pi).flatten(-2).float()
            sines = torch.sin(pos_freqs)
            cosines = torch.cos(pos_freqs)
            output = torch.cat([sines, cosines], dim=-1)  # [batch_size, num_tokens, phys_dim * num_freqs * 2]
        return output.to(x.dtype)


class NoEmbedPosition(BaseEmbedPosition):
    def __init__(self, phys_dim: int):
        super().__init__(phys_dim=phys_dim)

    @property
    def output_channels(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        No positional embedding, just return the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., phys_dim).

        Returns:
            torch.Tensor: Output tensor unchanged.
        """
        return torch.ones_like(x[..., :1])  # Return a tensor of ones with the same shape as the input's last dimension


def get_embed_position(
    phys_dim: int,
    config: EmbedPositionConfig,
) -> BaseEmbedPosition:
    """
    Get the position embedding layer based on the specified name.

    Args:
        name (str): Name of the embedding type.
        phys_dim (int): Physical dimensions of the input data.
        config (EmbedPositionConfig): Configuration for the embedding layer.

    Returns:
        BaseEmbedPosition: An instance of the specified position embedding layer.
    """
    name = config.name.lower()
    supported_types = ["sinusoidal", "nerf", "direct", "no_embed"]
    if name == "sinusoidal" or name == "nerf":
        return SinusoidalEmbedPosition(
            phys_dim=phys_dim,
            num_freqs=config.num_freqs,
            min_freqs_exp=config.min_freqs_exp,
            max_freqs_exp=config.max_freqs_exp,
        )

    elif name == "direct":
        return DirectEmbedPosition(phys_dim=phys_dim)

    elif name == "no_embed":
        return NoEmbedPosition(phys_dim=phys_dim)

    raise ValueError(f"Unknown position embedding type: {name}. Supported types: {supported_types}")
