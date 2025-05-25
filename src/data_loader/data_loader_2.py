"""
data_loader.py

Modular data ingestion utility for CSV and Excel files.
- Loads configuration from config.yaml
- Loads secrets from .env (using python-dotenv)
- Supports robust error handling and logging (configured by main.py)
- Designed for production and as a teaching example for MLOps best practices
"""

import os
import logging
import pandas as pd
import yaml
from dotenv import load_dotenv
from typing import Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config2.yaml") -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_env(env_path: str = ".env"):
    load_dotenv(dotenv_path=env_path, override=True)


def load_data(
    path: str,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified in configuration.")
        raise ValueError("No valid data path specified in configuration.")
    if not os.path.isfile(path):
        logger.error(f"Data file does not exist: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path, delimiter=delimiter, header=header, encoding=encoding)
        logger.info(f"Loaded data from {path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        raise


def get_data(
    config_path: str = "config2.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"
) -> pd.DataFrame:
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error(f"Unknown data_stage: {data_stage}")
        raise ValueError(f"Unknown data_stage: {data_stage}")
    if not path or not isinstance(path, str):
        logger.error(
            "No valid data path specified in configuration for data_stage='%s'.", data_stage)
        raise ValueError(
            f"No valid data path specified in configuration for data_stage='{data_stage}'.")
    df = load_data(
        path=path,
        delimiter=data_cfg.get("delimiter", ","),
        header=data_cfg.get("header", 0),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger.setLevel(logging.INFO)  # 🔸 Ensure your logger gets configured too
