from pathlib import Path

from src.data_preprocessing import preprocess_data
from src.evaluate import evaluate_model
from src.train import train_autoencoder


def ensure_project_dirs() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_project_dirs()

    print("Starting data preprocessing...")
    preprocess_data()

    print("Training autoencoder...")
    train_autoencoder()

    print("Evaluating model...")
    evaluate_model()

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
