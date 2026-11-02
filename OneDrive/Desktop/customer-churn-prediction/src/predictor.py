
from pathlib import Path
import json
import joblib
import pandas as pd

from src.config import MODEL_FILE
from src.logger import logger
from src.explainer import ModelExplainer


class Predictor:
    """
    Loads the trained pipeline and performs predictions
    using the optimized classification threshold.
    """

    def __init__(
        self,
        model_path: Path = MODEL_FILE,
        threshold_file: Path = Path(
            "artifacts/evaluation/evaluation_metrics.json"
        ),
    ):

        self.model_path = model_path
        self.threshold_file = threshold_file

        self.model = self.load_model()
        self.threshold = self.load_threshold()

        # Initialize SHAP explainer
        self.explainer = ModelExplainer()

    def load_model(self):

        logger.info("Loading trained model")

        return joblib.load(self.model_path)

    def load_threshold(self):

        logger.info("Loading optimized classification threshold")

        with open(self.threshold_file, "r") as file:
            metrics = json.load(file)

        threshold = metrics.get("best_threshold")

        if threshold is None:
            threshold = metrics.get("threshold")

        if threshold is None:
            raise ValueError(
                "Optimized threshold not found in evaluation_metrics.json"
            )

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid classification threshold")

        logger.info(f"Optimized threshold loaded: {threshold}")

        return float(threshold)

    def predict(self, data: pd.DataFrame) -> dict:

        logger.info("Generating churn prediction")

        probabilities = self.model.predict_proba(data)[0]

        churn_probability = probabilities[1]

        prediction = int(
            churn_probability >= self.threshold
        )

        # Generate SHAP explanation using the same threshold
        explanation = self.explainer.explain_prediction(
            data,
            threshold=self.threshold
        )

        top_features = [
            {
                "feature": item["feature"],
                "impact": round(float(item["shap_value"]), 2)
            }
            for item in explanation["top_features"]
        ]

        result = {
            "prediction": (
                "Likely to Churn"
                if prediction == 1
                else "Not Likely to Churn"
            ),

            "probability": round(
                float(churn_probability) * 100, 2
            ),

            "top_features": top_features
        }

        logger.info("Churn prediction generated successfully")

        return result
