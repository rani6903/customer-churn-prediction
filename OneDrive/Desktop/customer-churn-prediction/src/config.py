from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

DATA_FILE = DATA_DIR / "telco-customer-churn.csv"

MODEL_FILE = MODEL_DIR / "churn_model.pkl"
RESULTS_FILE = ARTIFACT_DIR / "model_results.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)