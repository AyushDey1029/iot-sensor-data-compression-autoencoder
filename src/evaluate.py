from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model import Autoencoder


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/autoencoder.pth")
OUTPUT_DIR = Path("outputs")


def load_test_data() -> np.ndarray:
    """Load test split from data/processed."""
    test_path = PROCESSED_DIR / "X_test.npy"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test data not found at: {test_path}")
    return np.load(test_path)


def evaluate_model(plot_samples: int = 50) -> float:
    """Evaluate reconstruction MSE and save optional comparison plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    x_test_np = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=6, latent_dim=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    x_test = torch.tensor(x_test_np, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()

    with torch.no_grad():
        reconstructed = model(x_test)
        mse = criterion(reconstructed, x_test).item()

    print(f"Final reconstruction error (MSE): {mse:.6f}")
    plot_original_vs_reconstructed(
        x_test.cpu().numpy(),
        reconstructed.cpu().numpy(),
        OUTPUT_DIR / "original_vs_reconstructed.png",
        max_samples=plot_samples,
    )
    return mse


def plot_original_vs_reconstructed(
    original: np.ndarray, reconstructed: np.ndarray, save_path: Path, max_samples: int = 50
) -> None:
    """Plot one feature from original and reconstructed arrays."""
    n = min(max_samples, len(original))
    if n == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(original[:n, 0], label="Original (co)")
    plt.plot(reconstructed[:n, 0], label="Reconstructed (co)", linestyle="--")
    plt.title("Original vs Reconstructed Sensor Values")
    plt.xlabel("Sample index")
    plt.ylabel("Scaled value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
