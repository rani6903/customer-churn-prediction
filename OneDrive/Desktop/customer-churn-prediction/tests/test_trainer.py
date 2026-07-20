from src.data_loader import DataLoader

from src.trainer import ModelTrainer

def test_split_data():

    loader = DataLoader()

    df = loader.clean_data(
        loader.load_data()
    )

    trainer = ModelTrainer()

    X_train, X_test, y_train, y_test = (
        trainer.split_data(df)
    )

    assert len(X_train) > 0

    assert len(X_test) > 0

from sklearn.pipeline import Pipeline


def test_build_pipeline():

    trainer = ModelTrainer()

    model = trainer.models["Logistic Regression"]

    pipeline = trainer.build_pipeline(model)

    assert isinstance(
        pipeline,
        Pipeline,
    )

def test_training():

    loader = DataLoader()

    df = loader.clean_data(
        loader.load_data()
    )

    trainer = ModelTrainer()

    results = trainer.train_models(df)

    assert not results.empty