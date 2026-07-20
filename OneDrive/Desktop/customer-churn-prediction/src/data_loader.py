import pandas as pd

from src.config import DATA_FILE
from src.exceptions import DataLoaderError
from src.logger import logger


class DataLoader:
    """
    Loads and validates the customer churn dataset.
    """

    REQUIRED_COLUMNS = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    def __init__(self, file_path=DATA_FILE):
        self.file_path = file_path

    def load_data(self):
        try:
            logger.info(f"Loading dataset from {self.file_path}")

            df = pd.read_csv(self.file_path)

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(str(e))
            raise DataLoaderError(str(e))

    def validate_columns(self, df):

        missing = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:

            raise DataLoaderError(
                f"Missing required columns: {missing}"
            )

        logger.info("Dataset validation successful.")

    def clean_data(self, df):
        """
        Basic cleaning before preprocessing.
        """

        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace(" ", None)
        )

        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(float)
        )

        return df