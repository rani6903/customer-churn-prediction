from pydantic import BaseModel, Field
from src.explainer import ModelExplainer

class CustomerRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    prediction: str = Field(
    ...,
    json_schema_extra={
        "example": "Likely to Churn"
    }
)

    probability: float = Field(
    ...,
    json_schema_extra={
        "example": 87.45
    }
)
    


class FeatureImportance(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    top_features: list[FeatureImportance]



