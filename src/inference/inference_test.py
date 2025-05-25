import os
import tempfile
import pandas as pd
import pickle  # ✅ You can also use cloudpickle if needed
import yaml
#from src.preprocess.preprocessing import get_output_feature_names

from inference.inferencer import run_inference


# ✅ This must NOT be indented — it should be at top level
class DummyModel:
    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


def test_run_inference_creates_predictions():
    input_df = pd.DataFrame({
        "amt": [100, 200],
        "merch_lat": [45.0, 46.0],
        "merch_long": [-75.0, -74.0],
        "category": ["grocery", "travel"],
        "gender": ["F", "M"],
        "state": ["NY", "CA"]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        model_path = os.path.join(tmpdir, "model.pkl")
        config_path = os.path.join(tmpdir, "config.yaml")
        output_csv = os.path.join(tmpdir, "output.csv")

        input_df.to_csv(input_csv, index=False)

        # ✅ DummyModel is now picklable
        with open(model_path, "wb") as f:
            pickle.dump(DummyModel(), f)

        config = {
            "artifacts": {
                "model_path": model_path
            },
            "raw_features": ["amt", "merch_lat", "merch_long", "category", "gender", "state"],
            "features": {
                "categorical": ["category", "gender"]
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        run_inference(input_csv, config_path, output_csv)

        result_df = pd.read_csv(output_csv)
        assert "prediction" in result_df.columns
        assert "prediction_proba" in result_df.columns
        assert len(result_df) == 2



