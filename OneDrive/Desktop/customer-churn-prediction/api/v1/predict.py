from fastapi import APIRouter
import pandas as pd

from api.schemas import (
    CustomerRequest,
    PredictionResponse,
)

from src.predictor import Predictor

router = APIRouter()

predictor = Predictor()

from src.explainer import ModelExplainer

predictor = Predictor()
explainer = ModelExplainer()


@router.post("/predict")
def predict(customer: CustomerRequest):

    df = pd.DataFrame([customer.model_dump()])

    prediction = predictor.predict(df)

    features = explainer.explain_prediction(df)

    prediction["top_features"] = features

    return prediction