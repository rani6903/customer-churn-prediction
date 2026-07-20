from pathlib import Path

import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODEL_FILE
from src.logger import logger

class ModelExplainer:
    """
    Generate SHAP explanations for the trained Gradient Boosting model.
    """

    def __init__(self, model_path: Path = MODEL_FILE):

        logger.info("Loading trained pipeline.")

        self.pipeline = joblib.load(model_path)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        self.model = self.pipeline.named_steps["classifier"]
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features and return a DataFrame with feature names.
        """

        X_processed = self.preprocessor.transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index
        )

        return X_processed
    
    def get_explanation(self, X_processed):

        explainer = shap.TreeExplainer(self.model)

        explanation = explainer(X_processed)

        return explanation
    
    def summary_plot(self, X: pd.DataFrame):

        logger.info("Generating summary plot")

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        plt.figure(figsize=(10, 6))

        shap.plots.beeswarm(
            explanation,
            show=False
        )

        plt.savefig(
            self.output_dir / "summary_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def feature_importance(self, X: pd.DataFrame):

        logger.info("Generating feature importance")

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        plt.figure(figsize=(10, 6))

        shap.plots.bar(
            explanation,
            show=False
        )

        plt.savefig(
            self.output_dir / "feature_importance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    def waterfall_plot(self, X: pd.DataFrame):

        logger.info("Generating waterfall plot")

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        plt.figure(figsize=(8, 6))

        shap.plots.waterfall(
            explanation[0],
            show=False
        )

        plt.savefig(
            self.output_dir / "waterfall_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    def explain_prediction(self, X: pd.DataFrame, top_n: int = 5):
        """
        Return the top SHAP feature contributions for one prediction.
        """

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        values = explanation.values[0]
        feature_names = X_processed.columns

        feature_importance = []

        for feature, value in zip(feature_names, values):
            feature_importance.append(
                {
                    "feature": feature,
                    "impact": float(value)
                }
            )

        feature_importance.sort(
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        return feature_importance[:top_n]
        