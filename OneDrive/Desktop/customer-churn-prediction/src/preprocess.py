from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logger

import joblib

class DataPreprocessor:
    """
    Builds a preprocessing pipeline for the Telco Customer Churn dataset.
    """

    def __init__(self):
        self.numeric_features = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]

        self.categorical_features = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]

        self.pipeline = None

    def build_pipeline(self):
        """
        Build preprocessing pipeline.
        """

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    self.numeric_features,
                ),
                (
                    "cat",
                    categorical_pipeline,
                    self.categorical_features,
                ),
            ]
        )

        logger.info("Preprocessing pipeline created.")

        return self.pipeline
    
        joblib.dump(
            pipeline,
    "       artifacts/preprocessor.pkl",
        )