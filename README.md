
# MLOps Pipeline for Opioid Abuse Disorder Prediction

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/2025-IE-MLOps-course/mlops_project)

This repository provides a modular, **production-quality** MLOps pipeline for binary classification of opioid abuse disorder, built as part of an academic project in the fundamentals of MLOps. The codebase is designed to bridge the gap between research prototypes (Jupyter notebooks) and scalable, maintainable machine learning systems in production.

---

## 🚦 Project Status

**Phase 1: Modularization, Testing, and Best Practices**
- Jupyter notebook translated into well-documented, test-driven Python modules
- End-to-end pipeline: data ingestion, validation, feature engineering, preprocessing, model training, evaluation, and batch inference
- Robust configuration via `config.yaml` and reproducibility through explicit artifact management
- Extensive unit testing with pytest
- Strict adherence to software engineering and MLOps best practices
---

## 📁 Repository Structure

```text
.
├── README.md
├── config.yaml                  # Central pipeline configuration
├── environment.yml              # Reproducible conda environment
├── data/                        # All project data (raw, splits, processed, inference)
├── models/                      # Serialized models, metrics, and preprocessing artifacts
├── logs/                        # Logging and validation artifacts
├── notebooks/                   # Source Jupyter notebook
├── src/
│   ├── data_load/               # Data ingestion utilities
│   ├── data_validation/         # Config-driven schema and data validation
│   ├── evaluation/              # Model evaluation (metrics, confusion matrix, etc.)
│   ├── features/                # Feature engineering transformers (scikit-learn compatible)
│   ├── inference/               # Batch inference script
│   ├── model/                   # End-to-end training and evaluation logic
│   └── preprocess/              # Preprocessing pipeline assembly
├── tests/                       # Unit and integration tests (pytest)
```

---

## 🔬 Problem Description

The pipeline predicts **opioid abuse disorder** based on anonymized claims data, engineered features, and diagnostic group flags. It applies rigorous validation and modular design suitable for research, teaching, and real-world deployment.

### Data Dictionary

| Feature        | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| OD             | Opioid abuse disorder flag (target)                                       |
| Low_inc        | Low income indicator                                                      |
| Surgery        | Major surgery in 2 years                                                  |
| rx ds          | Days supply of opioid drugs over 2 years                                  |
| A-V            | ICD-10 disease chapter flags (see full mapping in the code base)          |

*Full dictionary in project root.*

---

## 🛠️ Pipeline Modules

### 0. Pipeline Orchestration (`src/main.py`)
- Single entry point that orchestrates the entire MLOps workflow
- Supports configurable pipeline stages: data validation, model training, batch inference
- Integrates all modular components with robust logging and error handling
- Enables reproducible, scriptable runs for both research and production

### 1. Data Loading (`src/data_load/data_loader.py`)
- Loads data from CSV/Excel as specified in `config.yaml`
- Loads secrets from `.env` (for secure environments)
- Robust error handling and logging

### 2. Data Validation (`src/data_validation/data_validator.py`)
- Validates schema, column types, ranges, missing values
- Configurable strictness: `raise` or `warn` on errors
- Outputs detailed validation reports (JSON)

### 3. Feature Engineering (`src/features/feature_eng.py`)
- Implements scikit-learn compatible transformers (e.g., clinical risk score)
- Easily extendable for future feature construction

### 4. Preprocessing Pipeline (`src/preprocess/preprocessing.py`)
- Modular scikit-learn pipelines
- Prevents data leakage: pipeline fit **only on train split**

### 5. Model Training and Evaluation (`src/model/model.py`)
- Data split (train/valid/test) with stratification
- Model registry (easily swap DecisionTree, LogisticRegression, RandomForest)
- Evaluation: accuracy, precision, recall, specificity, F1, ROC AUC, etc.
- Metrics, models, and pipeline saved as artifacts

### 6. Batch Inference (`src/inference/inferencer.py`)
- Loads preprocessing and model artifacts
- Transforms new data, predicts outcomes, exports CSV

### 7. Unit Testing (`tests/`)
- Full pytest suite for each module
- Sample/mock data provided for CI/CD

---

## ⚙️ Configuration and Reproducibility

- **config.yaml**: All settings for data, model, features, metrics, and paths
- **environment.yml**: Reproducible Conda environment (Python, scikit-learn, pandas, PyYAML, etc.)
- **Artifacts**: All intermediate and final artifacts (preprocessing, models, metrics) are versioned and stored

---

## 🚀 Quickstart

**Environment setup:**
```bash
conda env create -f environment.yml
conda activate mlops-opioid
```

**Run end-to-end pipeline:**
```bash
python -m src.main --config config.yaml --stage all
```

**Run inference from any server (after cloning repo and installing dependencies):**
```bash
python -m src.main --stage infer --config config.yaml --input_csv data/inference/new_data.csv --output_csv data/inference/output_predictions.csv
```

**Run tests:**
```bash
pytest
```

---

## 📈 Next Steps (Planned Enhancements)

- **Experiment Tracking:** Integrate MLflow and/or Weights & Biases for automated tracking of parameters, metrics, and artifacts
- **CI/CD Automation:** Add GitHub Actions for linting, testing, and pipeline automation
- **Dynamic Configuration:** Migrate to Hydra for flexible multi-environment settings
- **Production Monitoring:** Add robust logging, alerting, and drift monitoring tools

---

## 📚 Academic and Teaching Notes

- **Best Practices:** Each module demonstrates academic best practices in code quality, modularity, and reproducibility, suitable for both teaching and production environments
- **Extensibility:** All logic is driven by config and easily extensible for advanced topics in MLOps
- **Assessment:** Project is fully test-driven, with sample data and clear structure for student/peer evaluation

---

## 👩‍💻 Authors and Acknowledgments

- Project led by Prof. Ivan Diaz
- Part of MLOps (IE University)
- Inspired by open-source MLOps community and real-world healthcare analytics use cases

---

## 📜 License

This project is for academic and educational purposes. See `LICENSE` for details.

---

**For questions or collaboration, open an issue or contact [maintainer email].**
