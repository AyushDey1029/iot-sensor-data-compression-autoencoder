from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.data_preprocessing import FEATURE_COLUMNS, clean_dataset, load_data
from src.model import Autoencoder


MODEL_PATH = Path("models/autoencoder.pth")
SCALER_PATH = Path("data/processed/scaler.pkl")
THRESHOLD_PATH = Path("outputs/best_threshold.txt")


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


@dataclass(frozen=True)
class PredictionResult:
    mse: float
    mae: float
    accuracy: float
    threshold: float
    threshold_accuracy: float
    anomaly_percent: float
    sample_error: np.ndarray
    original: np.ndarray
    reconstructed: np.ndarray
    anomalies: List[Dict[str, Any]]


def load_scaler():
    """Load MinMaxScaler from preprocessing."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}. Run `python main.py` to generate it."
        )
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_threshold(default_value: float = 0.01) -> float:
    """Load best threshold from evaluation output."""
    if THRESHOLD_PATH.exists():
        try:
            return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            return default_value
    return default_value


def load_model(device: torch.device) -> torch.nn.Module:
    """Load trained model (supports deep + legacy checkpoints)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    try:
        model = Autoencoder(input_dim=6, hidden_dims=[4, 2]).to(device)
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = LegacyAutoencoder(input_dim=6, latent_dim=3).to(device)
        model.load_state_dict(state_dict)

    model.eval()
    return model


def load_dataframe_from_upload(file_bytes: bytes, filename: str):
    """
    Load CSV/XLSX upload into a DataFrame using the same logic as training.
    """
    buffer = BytesIO(file_bytes)
    return load_data(buffer, filename)


def predict_from_dataframe(
    df,
    model: torch.nn.Module,
    scaler,
    threshold: float,
    max_rows: int = 5000,
) -> PredictionResult:
    """
    Run preprocessing + inference and compute metrics.

    Notes:
    - We return scaled original/reconstructed values (consistent with training).
    - max_rows limits the payload to keep JSON response manageable.
    """
    features_df = clean_dataset(df)  # drops ts (if present) + validates columns + drops NaNs

    if len(features_df) == 0:
        raise ValueError("No valid rows after cleaning (check for missing/non-numeric values).")

    if len(features_df) > max_rows:
        features_df = features_df.head(max_rows).copy()

    original = scaler.transform(features_df).astype(np.float32)
    x = torch.tensor(original, dtype=torch.float32)

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        reconstructed = model(x).detach().cpu().numpy().astype(np.float32)

    diff = original - reconstructed
    sample_error = (diff**2).mean(axis=1)
    mse = float((diff**2).mean())
    mae = float(np.abs(diff).mean())
    accuracy = max(0.0, 1.0 - mse)

    threshold_accuracy = float((sample_error < threshold).mean() * 100.0)
    anomaly_mask = sample_error > threshold
    anomaly_percent = float(anomaly_mask.mean() * 100.0)

    anomalies: List[Dict[str, Any]] = []
    if anomaly_mask.any():
        idxs = np.where(anomaly_mask)[0]
        for i in idxs:
            row = {col: float(features_df.iloc[i][col]) for col in FEATURE_COLUMNS}
            anomalies.append(
                {
                    "index": int(i),
                    "error": float(sample_error[i]),
                    "label": "Anomaly",
                    "row": row,
                }
            )

    return PredictionResult(
        mse=mse,
        mae=mae,
        accuracy=accuracy,
        threshold=threshold,
        threshold_accuracy=threshold_accuracy,
        anomaly_percent=anomaly_percent,
        sample_error=sample_error,
        original=original,
        reconstructed=reconstructed,
        anomalies=anomalies,
    )

