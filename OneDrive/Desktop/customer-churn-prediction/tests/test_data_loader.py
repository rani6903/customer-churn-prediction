import pandas as pd

from src.data_loader import DataLoader


def test_load_data():

    loader = DataLoader()

    df = loader.load_data()

    assert isinstance(df, pd.DataFrame)

def test_required_columns():

    loader = DataLoader()

    df = loader.load_data()

    loader.validate_columns(df)

    assert "Churn" in df.columns

def test_clean_data():

    loader = DataLoader()

    df = loader.load_data()

    cleaned = loader.clean_data(df)

    assert cleaned["TotalCharges"].dtype == float

