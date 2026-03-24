from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RAW_DATA_PATH = Path("data/raw/iot_telemetry_data.csv")
RAW_DATA_XLSX_PATH = Path("data/raw/iot_telemetry_data.xlsx")
PROCESSED_DIR = Path("data/processed")
FEATURE_COLUMNS = ["co", "humidity", "light", "lpg", "smoke", "temp"]


def load_dataset(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load IoT telemetry data from CSV or XLSX."""
    if csv_path.exists():
        return pd.read_csv(csv_path)

    if RAW_DATA_XLSX_PATH.exists():
        return pd.read_excel(RAW_DATA_XLSX_PATH)

    raise FileNotFoundError(
        "Dataset not found. Expected one of:\n"
        f"- {csv_path}\n"
        f"- {RAW_DATA_XLSX_PATH}"
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ts, keep model features, and handle missing values."""
    if "ts" in df.columns:
        df = df.drop(columns=["ts"])

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    clean_df = df[FEATURE_COLUMNS].copy()
    clean_df = clean_df.apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.dropna().reset_index(drop=True)
    return clean_df


def normalize_and_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale features to [0, 1] and split into train/test."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df)
    x_test = scaler.transform(test_df)
    return x_train, x_test


def save_processed_data(x_train: np.ndarray, x_test: np.ndarray) -> None:
    """Save numpy arrays to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", x_train)
    np.save(PROCESSED_DIR / "X_test.npy", x_test)


def preprocess_data() -> Tuple[np.ndarray, np.ndarray]:
    """Run full preprocessing pipeline and print output shapes."""
    raw_df = load_dataset()
    clean_df = clean_dataset(raw_df)
    x_train, x_test = normalize_and_split(clean_df)
    save_processed_data(x_train, x_test)

    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    return x_train, x_test
