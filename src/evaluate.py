from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    # Works when running from project root (e.g., python main.py).
    from src.model import Autoencoder
except ModuleNotFoundError:
    # Works when running this file directly (e.g., python src/evaluate.py).
    from model import Autoencoder


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/autoencoder.pth")
OUTPUT_DIR = Path("outputs")


class LegacyAutoencoder(torch.nn.Module):
    """Backward-compatible model for older 6->3->6 checkpoints."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 3) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, input_dim),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def load_test_data() -> np.ndarray:
    """Load test split from data/processed."""
    test_path = PROCESSED_DIR / "X_test.npy"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test data not found at: {test_path}")
    return np.load(test_path)


def evaluate_model(plot_samples: int = 50) -> float:
    """Evaluate reconstruction quality and save optional comparison plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    x_test_np = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_for_checkpoint(device)
    model.eval()

    x_test = torch.tensor(x_test_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(x_test)
        # Global reconstruction metrics over all samples and features.
        error = ((x_test - reconstructed) ** 2).mean()
        mse = error.item()
        mae = (x_test - reconstructed).abs().mean().item()

        # Accuracy-like score requested: higher is better, lower error -> higher accuracy.
        reconstruction_accuracy = max(0.0, 1 - mse)

        # Per-sample reconstruction error for threshold-based accuracy.
        sample_error = ((x_test - reconstructed) ** 2).mean(dim=1)
        best_threshold, best_threshold_accuracy = find_best_threshold(sample_error)

    print(f"Reconstruction Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Reconstruction Accuracy: {reconstruction_accuracy * 100:.2f}%")
    print(
        "Threshold Accuracy "
        f"(best threshold={best_threshold:.4f}): {best_threshold_accuracy:.2f}%"
    )

    plot_error_histogram(sample_error.cpu().numpy(), OUTPUT_DIR / "reconstruction_error_hist.png")
    plot_original_vs_reconstructed(
        x_test.cpu().numpy(),
        reconstructed.cpu().numpy(),
        OUTPUT_DIR / "original_vs_reconstructed.png",
        max_samples=plot_samples,
    )
    return mse


def load_model_for_checkpoint(device: torch.device) -> torch.nn.Module:
    """Load checkpoint with either new or legacy architecture."""
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # Try the current deep architecture first.
    try:
        model = Autoencoder(input_dim=6, hidden_dims=[4, 2]).to(device)
        model.load_state_dict(state_dict)
        print("Loaded model architecture: deep autoencoder (6->4->2->4->6)")
        return model
    except RuntimeError:
        # Fallback for older checkpoints saved with the simple architecture.
        legacy_model = LegacyAutoencoder(input_dim=6, latent_dim=3).to(device)
        legacy_model.load_state_dict(state_dict)
        print("Loaded model architecture: legacy autoencoder (6->3->6)")
        print("Tip: retrain with python main.py to use the improved deep architecture.")
        return legacy_model


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


def find_best_threshold(sample_error: torch.Tensor) -> Tuple[float, float]:
    """Search thresholds and return threshold with highest sample-level accuracy."""
    # Dense threshold search to maximize threshold-based accuracy.
    thresholds = np.linspace(0.001, 0.02, 200)
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0

    for threshold in thresholds:
        accuracy = (sample_error < threshold).float().mean().item() * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def plot_error_histogram(sample_error: np.ndarray, save_path: Path) -> None:
    """Plot distribution of per-sample reconstruction errors."""
    plt.figure(figsize=(8, 5))
    plt.hist(sample_error, bins=30)
    plt.title("Histogram of Reconstruction Errors")
    plt.xlabel("Per-sample MSE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    evaluate_model()
