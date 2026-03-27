import torch
import torch.nn as nn
from typing import List


class Autoencoder(nn.Module):
    """Configurable symmetric autoencoder for IoT sensor data."""

    def __init__(self, input_dim: int = 6, hidden_dims: List[int] | None = None) -> None:
        super().__init__()
        # Default architecture: 6 -> 4 -> 2 -> 4 -> 6
        if hidden_dims is None:
            hidden_dims = [4, 2]
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two values, e.g. [4, 2].")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
