"""
data_validation.py

Reusable, config-driven, production-quality data validation module for MLOps pipelines.
- Validates schema, types, missing values, ranges, allowed values
- Validation rules are read from config.yaml under data_validation.schema
- Behavior on validation failure is configurable (raise or warn)
- Validation report is saved as an artifact (JSON) for reproducibility and traceability
- Meant for initial data load to ensure quality and pipeline stability

How to use (in main.py, after loading data):

    from src.data_validation.data_validation import validate_data
    validate_data(df, config)

Rationale for each step is explained inline as a teaching resource.
"""

import logging
import os
import json
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _is_dtype_compatible(series, expected_dtype: str) -> bool:
    """
    Returns True if pandas Series dtype matches expected_dtype (int, float, str, bool).
    Accepts small numeric type mismatches (e.g., int64 vs int32).
    """
    kind = series.dtype.kind
    if expected_dtype == "int":
        return kind in ("i", "u")
    elif expected_dtype == "float":
        return kind == "f"
    elif expected_dtype == "str":
        return kind in ("O", "U", "S")
    elif expected_dtype == "bool":
        return kind == "b"
    return False


def _validate_column(
    df: pd.DataFrame,
    col_schema: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    report: Dict[str, Any]
) -> None:
    col = col_schema["name"]
    col_report = {}
    if col not in df.columns:
        if col_schema.get("required", True):
            msg = f"Missing required column: {col}"
            errors.append(msg)
            col_report["status"] = "missing"
            col_report["error"] = msg
        else:
            col_report["status"] = "not present (optional)"
        report[col] = col_report
        return

    col_series = df[col]
    col_report["status"] = "present"

    # Type check
    dtype_expected = col_schema.get("dtype")
    if dtype_expected and not _is_dtype_compatible(col_series, dtype_expected):
        msg = f"Column '{col}' has dtype '{col_series.dtype}', expected '{dtype_expected}'"
        errors.append(msg)
        col_report["dtype"] = str(col_series.dtype)
        col_report["dtype_expected"] = dtype_expected
        col_report["error"] = msg

    # Missing values check
    missing_count = col_series.isnull().sum()
    if missing_count > 0:
        if col_schema.get("required", True):
            msg = f"Column '{col}' has {missing_count} missing values (required)"
            errors.append(msg)
        else:
            msg = f"Column '{col}' has {missing_count} missing values (optional)"
            warnings.append(msg)
        col_report["missing_count"] = int(missing_count)

    # Value checks: min, max, allowed_values
    if "min" in col_schema:
        min_val = col_schema["min"]
        below = (col_series < min_val).sum()
        if below > 0:
            msg = f"Column '{col}' has {below} values below min ({min_val})"
            errors.append(msg)
            col_report["below_min"] = int(below)
    if "max" in col_schema:
        max_val = col_schema["max"]
        above = (col_series > max_val).sum()
        if above > 0:
            msg = f"Column '{col}' has {above} values above max ({max_val})"
            errors.append(msg)
            col_report["above_max"] = int(above)
    if "allowed_values" in col_schema:
        allowed = set(col_schema["allowed_values"])
        invalid = ~col_series.isin(allowed)
        n_invalid = invalid.sum()
        if n_invalid > 0:
            msg = f"Column '{col}' has {n_invalid} values not in allowed set {allowed}"
            errors.append(msg)
            col_report["invalid_values_count"] = int(n_invalid)

    # Value distribution for report
    try:
        col_report["sample_values"] = col_series.dropna().unique()[:5].tolist()
    except Exception:
        col_report["sample_values"] = "unavailable"

    report[col] = col_report


def validate_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> None:
    """
    Run full data validation as per config.yaml rules

    Args:
        df: DataFrame to validate
        config: full config dict (expects config['data_validation'])

    Saves validation report as artifact. Raises or warns depending on config.
    """
    dv_cfg = config.get("data_validation", {})
    enabled = dv_cfg.get("enabled", True)
    if not enabled:
        logger.info("Data validation is disabled in config.")
        return

    schema = dv_cfg.get("schema", {}).get("columns", [])
    if not schema:
        logger.warning(
            "No data_validation.schema.columns defined in config. Skipping validation.")
        return

    action_on_error = dv_cfg.get("action_on_error", "raise").lower()
    report_path = dv_cfg.get("report_path", "logs/validation_report.json")
    errors, warnings = [], []
    report = {}

    # Validate each column as per schema
    for col_schema in schema:
        _validate_column(df, col_schema, errors, warnings, report)

    # Save validation report (teaching: critical for reproducibility & audits)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "result": "fail" if errors else "pass",
            "errors": errors,
            "warnings": warnings,
            "details": report
        }, f, indent=2)

    # Log summary
    if errors:
        logger.error(
            f"Data validation failed with {len(errors)} errors. See {report_path}")
        for e in errors:
            logger.error(e)
    if warnings:
        logger.warning(f"Data validation warnings: {len(warnings)}")
        for w in warnings:
            logger.warning(w)

    # Teaching: You want strict validation in prod, warnings for research
    if errors:
        if action_on_error == "raise":
            raise ValueError(
                f"Data validation failed with errors. See {report_path} for details"
            )
        elif action_on_error == "warn":
            logger.warning(
                "Data validation errors detected but proceeding as per config."
            )
        else:
            logger.warning(
                f"Unknown action_on_error '{action_on_error}'. Proceeding but data may be invalid."
            )
    else:
        logger.info(f"Data validation passed. Details saved to {report_path}")


# CLI support (optional)
if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python -m src.data_validation.data_validation <data.csv> <config.yaml>")
        sys.exit(1)
    data_path, config_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    validate_data(df, config)