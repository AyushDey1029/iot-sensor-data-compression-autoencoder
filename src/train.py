from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.model import Autoencoder


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/autoencoder.pth")
OUTPUT_DIR = Path("outputs")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.5


def load_processed_data() -> np.ndarray:
    """Load preprocessed training data from .npy."""
    train_path = PROCESSED_DIR / "X_train.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training data not found at: {train_path}")
    return np.load(train_path).astype(np.float32)


def to_dataloader(data: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """Convert numpy data to a PyTorch DataLoader."""
    tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def split_train_validation(
    x_train: np.ndarray, validation_split: float = VALIDATION_SPLIT, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Create train/validation split for early stopping and monitoring."""
    x_train_split, x_val_split = train_test_split(
        x_train, test_size=validation_split, random_state=random_state, shuffle=True
    )
    return x_train_split, x_val_split


def _run_epoch(
    model: Autoencoder, dataloader: DataLoader, criterion: nn.Module, optimizer=None, device=None
) -> float:
    """Run one training or validation epoch and return average loss."""
    training = optimizer is not None
    model.train(mode=training)
    running_loss = 0.0

    for (batch_x,) in dataloader:
        batch_x = batch_x.to(device)
        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def _plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Save training/validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_autoencoder() -> Dict[str, List[float]]:
    """Train the autoencoder with validation tracking and early stopping."""
    x_train = load_processed_data()
    x_train_split, x_val_split = split_train_validation(x_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=6, hidden_dims=[4, 2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Reduce learning rate over time for smoother convergence.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    train_loader = to_dataloader(x_train_split, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = to_dataloader(x_val_split, batch_size=BATCH_SIZE, shuffle=False)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model=model, dataloader=val_loader, criterion=criterion, device=device
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping: keep the best model based on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        scheduler.step()

    _plot_training_curves(history, OUTPUT_DIR / "loss_curve.png")
    print(f"Training complete. Best model saved to: {MODEL_PATH}")
    return history
