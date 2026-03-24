import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple 6->3->6 autoencoder for IoT sensor data."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 3) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
