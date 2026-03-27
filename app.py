from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.model import Autoencoder
from src.data_preprocessing import FEATURE_COLUMNS, clean_dataset, load_data


MODEL_PATH = Path("models/autoencoder.pth")
SCALER_PATH = Path("data/processed/scaler.pkl")
THRESHOLD_PATH = Path("outputs/best_threshold.txt")
MAX_TABLE_ROWS = 5000


class LegacyAutoencoder(torch.nn.Module):
    """Fallback model for older checkpoints saved with 6->3->6 architecture."""

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


@st.cache_resource
def load_model() -> torch.nn.Module:
    """Load trained model from disk (supports new and legacy checkpoints)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(MODEL_PATH, map_location=device)

    try:
        model = Autoencoder(input_dim=6, hidden_dims=[4, 2]).to(device)
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = LegacyAutoencoder(input_dim=6, latent_dim=3).to(device)
        model.load_state_dict(state_dict)

    model.eval()
    return model


@st.cache_resource
def load_scaler():
    """Load scaler used during preprocessing/training."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}. Run `python main.py` to regenerate it."
        )
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_best_threshold(default_value: float = 0.01) -> float:
    """Read best threshold from evaluation output, with safe fallback."""
    if THRESHOLD_PATH.exists():
        try:
            return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            return default_value
    return default_value


def run_inference(df_features: pd.DataFrame, model: torch.nn.Module, scaler) -> pd.DataFrame:
    """Scale input, reconstruct with autoencoder, and compute errors/labels."""
    x_scaled = scaler.transform(df_features)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(x_tensor).cpu().numpy()

    sample_error = ((x_scaled - reconstructed) ** 2).mean(axis=1)
    mse = float(((x_scaled - reconstructed) ** 2).mean())
    threshold = load_best_threshold()
    labels = np.where(sample_error > threshold, "Anomaly", "Normal")

    result_df = df_features.copy()
    result_df["reconstruction_error"] = sample_error
    result_df["label"] = labels
    result_df.attrs["mse"] = mse
    result_df.attrs["threshold"] = threshold
    result_df.attrs["scaled_original"] = x_scaled
    result_df.attrs["scaled_reconstructed"] = reconstructed
    return result_df


def plot_original_vs_reconstructed(original: np.ndarray, reconstructed: np.ndarray, feature_idx: int):
    """Plot selected feature values before and after reconstruction."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original[:, feature_idx], label=f"Original ({FEATURE_COLUMNS[feature_idx]})")
    ax.plot(
        reconstructed[:, feature_idx],
        label=f"Reconstructed ({FEATURE_COLUMNS[feature_idx]})",
        linestyle="--",
    )
    ax.set_title("Original vs Reconstructed")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Scaled value")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_error(sample_error: np.ndarray):
    """Plot reconstruction error per sample."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sample_error, color="tab:red")
    ax.set_title("Reconstruction Error per Sample")
    ax.set_xlabel("Sample")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    return fig


def show_limited_dataframe(title: str, df: pd.DataFrame, max_rows: int = MAX_TABLE_ROWS) -> None:
    """Render a safe-sized table to avoid Streamlit memory issues on huge datasets."""
    st.subheader(title)
    total_rows = len(df)
    if total_rows == 0:
        st.info("No rows to display.")
        return

    rows_to_show = min(total_rows, max_rows)
    if total_rows > max_rows:
        st.warning(
            f"Displaying first {rows_to_show:,} of {total_rows:,} rows "
            f"to prevent browser memory issues."
        )
    st.dataframe(df.head(rows_to_show), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="IoT Autoencoder Anomaly Detector", layout="wide")
    st.title("IoT Autoencoder - Reconstruction and Anomaly Detection")
    st.caption("Upload new sensor data to reconstruct signals and detect anomalies.")

    uploaded_file = st.file_uploader("Upload IoT Data", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info(
            "Please upload a CSV or Excel file with columns: "
            "co, humidity, light, lpg, smoke, temp."
        )
        return

    try:
        # Reuse the same file-loading rules as preprocessing (CSV/XLS/XLSX).
        raw_df = load_data(uploaded_file, uploaded_file.name)
        feature_df = clean_dataset(raw_df)
        model = load_model()
        scaler = load_scaler()
        result_df = run_inference(feature_df, model, scaler)
    except Exception as exc:
        st.error(str(exc))
        st.error(f"Required columns: {FEATURE_COLUMNS}")
        return

    st.subheader("Uploaded Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    sample_error = result_df["reconstruction_error"].to_numpy()
    mse = result_df.attrs["mse"]
    threshold = result_df.attrs["threshold"]
    anomaly_pct = float((result_df["label"] == "Anomaly").mean() * 100)
    avg_error = float(sample_error.mean())

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Error (MSE)", f"{avg_error:.6f}")
    col2.metric("Reconstruction Accuracy", f"{max(0.0, 1 - mse) * 100:.2f}%")
    col3.metric("Anomalies", f"{anomaly_pct:.2f}%")
    st.write(f"Using threshold: `{threshold:.6f}`")

    st.subheader("Visualization")
    feature_name = st.selectbox("Feature for comparison plot", FEATURE_COLUMNS, index=0)
    feature_idx = FEATURE_COLUMNS.index(feature_name)
    st.pyplot(
        plot_original_vs_reconstructed(
            result_df.attrs["scaled_original"],
            result_df.attrs["scaled_reconstructed"],
            feature_idx,
        )
    )
    st.pyplot(plot_reconstruction_error(sample_error))

    anomalies = result_df[result_df["label"] == "Anomaly"]
    if anomalies.empty:
        st.subheader("Anomaly Rows")
        st.success("No anomalies found with the current threshold.")
    else:
        show_limited_dataframe("Anomaly Rows", anomalies)

    show_limited_dataframe("All Scored Rows", result_df)

    # Provide full results as a download instead of rendering massive tables in browser.
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Scored Results (CSV)",
        data=csv_bytes,
        file_name="scored_iot_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
