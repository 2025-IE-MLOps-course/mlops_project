"""
model.py

Leakage-proof, end-to-end MLOps pipeline:
- Splits raw data first
- Fits preprocessing pipeline ONLY on train split, applies to valid/test
- Trains model, evaluates, and saves model and preprocessing artifacts
"""

import os
import logging
import json
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from src.preprocess.preprocessing import build_preprocessing_pipeline, get_output_feature_names, run_preprocessing_pipeline
from src.evaluation.evaluator import evaluate_classification


logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def train_model(X_train, y_train, model_type, params):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def evaluate_model(model, X, y, metrics):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(
        model, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    results = {}
    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower == "accuracy":
            results["Accuracy"] = accuracy_score(y, y_pred)
        elif metric_lower in ["precision", "precision (ppv)", "positive predictive value (ppv)"]:
            results["Precision (PPV)"] = precision_score(
                y, y_pred, zero_division=0)
        elif metric_lower in ["recall", "sensitivity"]:
            results["Recall (Sensitivity)"] = recall_score(
                y, y_pred, zero_division=0)
        elif metric_lower == "specificity":
            results["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif metric_lower == "f1 score":
            results["F1 Score"] = f1_score(y, y_pred, zero_division=0)
        elif metric_lower == "negative predictive value (npv)":
            results["Negative Predictive Value (NPV)"] = tn / \
                (tn + fn) if (tn + fn) > 0 else 0.0
        elif metric_lower == "roc auc":
            results["ROC AUC"] = roc_auc_score(
                y, y_prob) if y_prob is not None else float("nan")
    return results


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")


def format_metrics(metrics: dict, ndigits: int = 2) -> dict:
    return {k: round(float(v), ndigits) if isinstance(v, (float, int)) else v for k, v in metrics.items()}


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    df = run_preprocessing_pipeline(df, config)
    assert config["target"] in df.columns, f"{config['target']} column not found in DataFrame after preprocessing"

    # 1. Split data using only raw features (present in the original file)
    raw_features = config.get("raw_features", [])
    target = config["target"]
    split_cfg = config["data_split"]
    input_features_raw = [f for f in raw_features if f != target]

    X = df[input_features_raw]
    y = df[target]
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + valid_size), random_state=random_state, stratify=y
    )
    rel_valid = valid_size / (test_size + valid_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_valid, random_state=random_state, stratify=y_temp
    )
    # --- Save raw data splits ---
    splits_dir = config.get("artifacts", {}).get("splits_dir", "data/splits")
    os.makedirs(splits_dir, exist_ok=True)
    X_train.assign(
        **{target: y_train}).to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    X_valid.assign(
        **{target: y_valid}).to_csv(os.path.join(splits_dir, "valid.csv"), index=False)
    X_test.assign(**{target: y_test}
                  ).to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # 2. Fit preprocessing pipeline on X_train, transform all splits
    preprocessor = build_preprocessing_pipeline(config)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_valid_pp = preprocessor.transform(X_valid)
    X_test_pp = preprocessor.transform(X_test)

    # 3. Create DataFrames with engineered feature columns
    engineered_features = config.get("features", {}).get("engineered", [])
    out_cols = get_output_feature_names(
        preprocessor, input_features_raw, config)
    X_train_pp = pd.DataFrame(X_train_pp, columns=out_cols)
    X_valid_pp = pd.DataFrame(X_valid_pp, columns=out_cols)
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols)

    # 4. Use only engineered features for modeling
    input_features = [
        f for f in engineered_features if f in X_train_pp.columns]
    X_train_pp = X_train_pp[input_features]
    X_valid_pp = X_valid_pp[input_features]
    X_test_pp = X_test_pp[input_features]

    # Save processed data splits
    processed_dir = config.get("artifacts", {}).get(
        "processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    X_train_pp.assign(**{target: y_train}).to_csv(
        os.path.join(processed_dir, "train_processed.csv"), index=False)
    X_valid_pp.assign(**{target: y_valid}).to_csv(
        os.path.join(processed_dir, "valid_processed.csv"), index=False)
    X_test_pp.assign(
        **{target: y_test}).to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)

    # Save preprocessing pipeline artifact
    preproc_path = config.get("artifacts", {}).get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
    save_artifact(preprocessor, preproc_path)

    # Train model
    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]
    model_type = active
    params = active_model_cfg.get("params", {})
    model = train_model(X_train_pp.values, y_train, model_type, params)

    # Save model artifact
    model_path = config.get("artifacts", {}).get(
        "model_path", "models/model.pkl")
    save_artifact(model, model_path)

    active = model_config.get("active", "decision_tree")
    algo_model_path = model_config.get(active, {}).get("save_path", f"models/{active}.pkl")
    save_artifact(model, algo_model_path)

    # Evaluate and log/save metrics using evaluation.py
    artifacts_cfg = config.get("artifacts", {})
    metrics_path = artifacts_cfg.get("metrics_path", "models/metrics.json")

    results_valid = evaluate_classification(
        model, X_valid_pp.values, y_valid, config, split="validation")
    results_test = evaluate_classification(
        model, X_test_pp.values, y_test,  config, split="test")

    def round_metrics(metrics_dict, ndigits=2):
        rounded = {}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):  # For nested dicts (e.g., Confusion Matrix)
                rounded[k] = {ik: (round(iv, ndigits) if isinstance(
                    iv, float) else iv) for ik, iv in v.items()}
            elif isinstance(v, float):
                rounded[k] = round(v, ndigits)
            else:
                rounded[k] = v
        return rounded

    validation_rounded = round_metrics(results_valid)
    test_rounded = round_metrics(results_test)

    metrics_path = config.get("artifacts", {}).get(
        "metrics_path", "models/metrics.json")

    # Save both splits' metrics as one artifact
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "validation": validation_rounded,
            "test": test_rounded
        }, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


# CLI for standalone training
if __name__ == "__main__":
    import sys
    import yaml
    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        from src.data_load.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        data_path = config["data_source"]["raw_path"]
        df = pd.read_csv(data_path)
    run_model_pipeline(df, config)