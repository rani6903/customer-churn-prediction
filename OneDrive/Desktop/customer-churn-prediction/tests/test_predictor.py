import pandas as pd

from src.predictor import Predictor


def test_prediction():

    predictor = Predictor()

    sample = pd.DataFrame({

        "gender":["Female"],

        "SeniorCitizen":[0],

        "Partner":["Yes"],

        "Dependents":["No"],

        "tenure":[12],

        "PhoneService":["Yes"],

        "MultipleLines":["No"],

        "InternetService":["DSL"],

        "OnlineSecurity":["Yes"],

        "OnlineBackup":["No"],

        "DeviceProtection":["No"],

        "TechSupport":["Yes"],

        "StreamingTV":["No"],

        "StreamingMovies":["No"],

        "Contract":["One year"],

        "PaperlessBilling":["No"],

        "PaymentMethod":["Bank transfer (automatic)"],

        "MonthlyCharges":[65.5],

        "TotalCharges":[786.0],

    })

    result = predictor.predict(sample)

    assert "prediction" in result
    assert "probability" in result