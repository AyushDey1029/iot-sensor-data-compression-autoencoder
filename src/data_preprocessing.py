from pathlib import Path
from typing import Tuple
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RAW_DATA_PATH = Path("data/raw/iot_telemetry_data.csv")
RAW_DATA_XLSX_PATH = Path("data/raw/iot_telemetry_data.xlsx")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
FEATURE_COLUMNS = ["co", "humidity", "light", "lpg", "smoke", "temp"]


def load_data(file_path, file_name: str | None = None) -> pd.DataFrame:
    """
    Load IoT data from CSV/XLS/XLSX based on file extension.

    file_path can be a filesystem path or a file-like object (e.g., Streamlit upload).
    """
    source_name = file_name if file_name is not None else str(file_path)
    _, ext = os.path.splitext(source_name)
    ext = ext.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported file format: {ext or '[no extension]'}. "
        "Supported formats are .csv, .xlsx, and .xls."
    )


def load_dataset() -> pd.DataFrame:
    """Load training dataset from default raw-data location."""
    if RAW_DATA_PATH.exists():
        return load_data(RAW_DATA_PATH)
    if RAW_DATA_XLSX_PATH.exists():
        return load_data(RAW_DATA_XLSX_PATH)

    raise FileNotFoundError(
        "Dataset not found. Expected one of:\n"
        f"- {RAW_DATA_PATH}\n"
        f"- {RAW_DATA_XLSX_PATH}"
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ts, keep model features, and handle missing values."""
    if "ts" in df.columns:
        df = df.drop(columns=["ts"])

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {missing_cols}. "
            f"Required columns: {FEATURE_COLUMNS}"
        )

    clean_df = df[FEATURE_COLUMNS].copy()
    clean_df = clean_df.apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.dropna().reset_index(drop=True)
    return clean_df


def normalize_and_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale features to [0, 1] and split into train/test."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df)
    x_test = scaler.transform(test_df)
    return x_train, x_test, scaler


def save_processed_data(x_train: np.ndarray, x_test: np.ndarray, scaler: MinMaxScaler) -> None:
    """Save numpy arrays to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", x_train)
    np.save(PROCESSED_DIR / "X_test.npy", x_test)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)


def preprocess_data() -> Tuple[np.ndarray, np.ndarray]:
    """Run full preprocessing pipeline and print output shapes."""
    raw_df = load_dataset()
    clean_df = clean_dataset(raw_df)
    x_train, x_test, scaler = normalize_and_split(clean_df)
    save_processed_data(x_train, x_test, scaler)

    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    return x_train, x_test
