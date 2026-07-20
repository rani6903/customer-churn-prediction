import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["0001", "0002"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
        ],
        "MonthlyCharges": [65.0, 85.5],
        "TotalCharges": [780.0, 2052.0],
        "Churn": ["No", "Yes"],
    })