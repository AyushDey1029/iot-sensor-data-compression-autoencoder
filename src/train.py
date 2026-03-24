from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import Autoencoder


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/autoencoder.pth")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50


def load_processed_data() -> np.ndarray:
    """Load preprocessed training data from .npy."""
    train_path = PROCESSED_DIR / "X_train.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training data not found at: {train_path}")
    return np.load(train_path)


def to_dataloader(data: np.ndarray, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Convert numpy data to a PyTorch DataLoader."""
    tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_autoencoder() -> List[float]:
    """Train the autoencoder and save model weights."""
    x_train = load_processed_data()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=6, latent_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataloader = to_dataloader(x_train, batch_size=BATCH_SIZE)
    losses: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(len(dataloader), 1)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved to: {MODEL_PATH}")
    return losses
