from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import torch

from backend.ml import (
    load_dataframe_from_upload,
    load_model,
    load_scaler,
    load_threshold,
    predict_from_dataframe,
)


app = FastAPI(title="IoT Autoencoder API", version="1.0.0")

# Allow the React frontend to call this API in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), max_rows: int = 5000) -> Dict[str, Any]:
    """
    Upload a CSV/XLSX file and get reconstruction + anomaly metrics.

    The response returns scaled original/reconstructed arrays (same scale as training).
    max_rows limits response size for large uploads.
    """
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    try:
        file_bytes = await file.read()
        df = load_dataframe_from_upload(file_bytes, filename)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Failed to parse file as a table.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device)
        scaler = load_scaler()
        threshold = load_threshold()

        result = predict_from_dataframe(
            df=df, model=model, scaler=scaler, threshold=threshold, max_rows=max_rows
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    # Convert numpy arrays to JSON-serializable lists.
    return {
        "mse": result.mse,
        "mae": result.mae,
        "accuracy": result.accuracy,
        "threshold": result.threshold,
        "threshold_accuracy": result.threshold_accuracy,
        "anomaly_percent": result.anomaly_percent,
        "anomalies": result.anomalies,
        "sample_error": result.sample_error.tolist(),
        "original": result.original.tolist(),
        "reconstructed": result.reconstructed.tolist(),
    }

