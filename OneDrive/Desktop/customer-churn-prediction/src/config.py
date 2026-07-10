from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Files
DATA_FILE = DATA_DIR / "telco_customer_churn.csv"

# Create directories if missing
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)