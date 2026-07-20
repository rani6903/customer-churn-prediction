from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# Health endpoint

def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

def test_root():

    response = client.get("/")

    assert response.status_code == 200

def test_prediction():

    payload = {

        "gender":"Female",

        "SeniorCitizen":0,

        "Partner":"Yes",

        "Dependents":"No",

        "tenure":12,

        "PhoneService":"Yes",

        "MultipleLines":"No",

        "InternetService":"DSL",

        "OnlineSecurity":"Yes",

        "OnlineBackup":"No",

        "DeviceProtection":"No",

        "TechSupport":"Yes",

        "StreamingTV":"No",

        "StreamingMovies":"No",

        "Contract":"One year",

        "PaperlessBilling":"No",

        "PaymentMethod":"Bank transfer (automatic)",

        "MonthlyCharges":65.5,

        "TotalCharges":786.0
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body

    assert "probability" in body