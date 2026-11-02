from fastapi import APIRouter
import pandas as pd
from api.schemas import CustomerRequest, PredictionResponse
from src.predictor import Predictor


predictor = Predictor()
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_customer(customer: CustomerRequest):

    data = pd.DataFrame([customer.model_dump()])

    result = predictor.predict(data)

    return result