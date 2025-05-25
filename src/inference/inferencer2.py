"""
Batch inference entry point (aligned with notebook).

Usage:
    python -m src.inference.inferencer \
        data/inference/new_data.csv config.yaml data/inference/output_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_pickle(path: str, label: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def run_inference(input_csv: str, config_yaml: str, output_csv: str) -> None:
    _setup_logging()

    # Load config
    with open(config_yaml, "r") as fh:
        config = yaml.safe_load(fh)

    model_path = config.get("artifacts", {}).get("model_path", "models/model.pkl")
    logger.info("Loading trained model: %s", model_path)
    model = _load_pickle(model_path, "model")

    logger.info("Reading input CSV: %s", input_csv)
    input_df = pd.read_csv(input_csv)
    logger.info("Input shape: %s", input_df.shape)

    rename_map = config.get("preprocessing", {}).get("rename_columns", {})
    if rename_map:
        input_df = input_df.rename(columns=rename_map)

    raw_features = config.get("raw_features", [])
    missing = [c for c in raw_features if c not in input_df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[raw_features].copy()

    # Label Encoding (as per notebook logic)
    categorical = config.get("features", {}).get("categorical", [])
    for col in categorical:
        X_raw[col] = X_raw[col].astype("category").cat.codes

    # Predictions
    logger.info("Generating predictions")
    input_df["prediction"] = model.predict(X_raw)
    if hasattr(model, "predict_proba"):
        input_df["prediction_proba"] = model.predict_proba(X_raw)[:, 1]

    logger.info("Writing predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_csv, index=False)
    logger.info("Inference complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV file")
    parser.add_argument("input_csv", help="Path to raw input CSV")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Destination for predictions CSV")
    args = parser.parse_args()

    run_inference(args.input_csv, args.config_yaml, args.output_csv)


if __name__ == "__main__":
    main()