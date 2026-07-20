from fastapi import FastAPI

from api.middleware import LoggingMiddleware
from api.v1.predict import router as predict_router
from api.exceptions import (
    PredictionException,
    prediction_exception_handler,
)

app = FastAPI(
    title="Customer Churn API",
    version="1.0.0",
)

# Register middleware
app.add_middleware(LoggingMiddleware)

# Register exception handler
app.add_exception_handler(
    PredictionException,
    prediction_exception_handler,
)

# Register router
app.include_router(
    predict_router,
    prefix="/api/v1",
    tags=["Prediction"],
)


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}