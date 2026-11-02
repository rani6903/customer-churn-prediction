
from pathlib import Path

import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODEL_FILE
from src.logger import logger


class ModelExplainer:
    """
    Generate SHAP explanations for a trained classification pipeline.

    Supports:
    1. SHAP summary plot
    2. SHAP feature importance plot
    3. SHAP waterfall plot
    4. Individual customer prediction explanation
    """

    def __init__(self, model_path=MODEL_FILE):

        self.output_dir = Path("artifacts/explanations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model(model_path)

    # -------------------------------------------------
    # 1. Load trained model
    # -------------------------------------------------

    def _load_model(self, model_path):

        logger.info("Loading trained pipeline.")

        self.pipeline = joblib.load(model_path)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        self.model = self.pipeline.named_steps["classifier"]

        # Create SHAP explainer once and reuse it
        self.explainer = shap.TreeExplainer(self.model)

        logger.info("Model and SHAP explainer loaded successfully.")

    # -------------------------------------------------
    # 2. Preprocess input data
    # -------------------------------------------------

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw customer data and preserve feature names.
        """

        X_processed = self.preprocessor.transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index
        )

    # -------------------------------------------------
    # 3. Generate SHAP explanation
    # -------------------------------------------------

    def get_explanation(self, X_processed):
        """
        Generate SHAP values for transformed input data.
        """

        return self.explainer(X_processed)

    # -------------------------------------------------
    # 4. Save generated plots
    # -------------------------------------------------

    def _save_plot(self, filename):
        """
        Save the current Matplotlib figure.
        """

        save_path = self.output_dir / filename

        plt.gcf().savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close("all")

        logger.info("Plot saved: %s", save_path)

        return save_path

    # -------------------------------------------------
    # 5. SHAP summary plot
    # -------------------------------------------------

    def summary_plot(self, X: pd.DataFrame):
        """
        Show the distribution of SHAP values
        across multiple customers.
        """

        logger.info("Generating SHAP summary plot.")

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        shap.plots.beeswarm(
            explanation,
            max_display=15,
            show=False
        )

        return self._save_plot("summary_plot.png")

    # -------------------------------------------------
    # 6. SHAP feature importance plot
    # -------------------------------------------------

    def feature_importance(self, X: pd.DataFrame):
        """
        Show global feature importance using mean absolute SHAP values.
        """

        logger.info("Generating SHAP feature importance plot.")

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        shap.plots.bar(
            explanation,
            max_display=15,
            show=False
        )

        return self._save_plot("feature_importance.png")

    # -------------------------------------------------
    # 7. SHAP waterfall plot
    # -------------------------------------------------

    def waterfall_plot(self, X: pd.DataFrame):
        """
        Explain the prediction for one individual customer.
        """

        logger.info("Generating SHAP waterfall plot.")

        if len(X) != 1:
            raise ValueError(
                "Waterfall plot requires exactly one customer."
            )

        X_processed = self.preprocess(X)

        explanation = self.get_explanation(X_processed)

        single_explanation = explanation[0]

        # Explicitly preserve actual feature names
        waterfall_explanation = shap.Explanation(
            values=single_explanation.values,
            base_values=single_explanation.base_values,
            data=X_processed.iloc[0].values,
            feature_names=X_processed.columns.tolist()
        )

        shap.plots.waterfall(
            waterfall_explanation,
            max_display=15,
            show=False
        )

        return self._save_plot("waterfall_plot.png")

    # -------------------------------------------------
    # 8. Determine SHAP contribution direction
    # -------------------------------------------------

    @staticmethod
    def _get_feature_direction(value):
        """
        Determine the direction of a SHAP contribution.

        Assumes positive SHAP values push towards Churn
        for the binary Gradient Boosting model.
        """

        if value > 0:
            return "Towards Churn"

        elif value < 0:
            return "Towards Not Churn"

        return "No contribution"

    # -------------------------------------------------
    # 9. Explain individual customer prediction
    # -------------------------------------------------

    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n=5,
        threshold=0.5
    ):
        """
        Explain one customer's prediction.
        """

        if len(X) != 1:
            raise ValueError(
                "Please provide exactly one customer."
            )

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid classification threshold.")

        logger.info("Generating individual SHAP explanation.")

        # Transform customer data
        X_processed = self.preprocess(X)

        # Generate SHAP explanation
        explanation = self.get_explanation(X_processed)

        values = explanation.values

        # Handle common binary classification output shapes
        if values.ndim == 3:
            # For class-wise SHAP output, select positive class
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

        feature_names = X_processed.columns

        if len(feature_names) != len(values):
            raise ValueError(
                "SHAP values and feature names have different lengths."
            )

        feature_importance = []

        for feature, value, feature_value in zip(
            feature_names,
            values,
            X_processed.iloc[0].values
        ):

            feature_importance.append({
            "feature": self.clean_feature_name(feature),
            "feature_value": float(feature_value),
            "shap_value": float(value),
            "direction": self._get_feature_direction(value)
        })

        feature_importance.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        churn_probability = float(
            self.pipeline.predict_proba(X)[0, 1]
        )

        prediction = int(
            churn_probability >= threshold
        )

        prediction_label = (
            "Churn" if prediction == 1 else "Not Churn"
        )

        logger.info("Individual SHAP explanation generated.")

        return {
            "churn_probability": churn_probability,
            "threshold": float(threshold),
            "prediction": prediction_label,
            "top_features": feature_importance[:top_n]
        }




    @staticmethod
    def clean_feature_name(feature):

        if feature.startswith("num__"):
            feature = feature.replace("num__", "", 1)

        elif feature.startswith("cat__"):
            feature = feature.replace("cat__", "", 1)

        if feature == "tenure":
            return "Tenure"

        return feature