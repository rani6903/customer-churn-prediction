from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_FILE
from src.logger import logger

from src.explainer import ModelExplainer




class Predictor:
    """
    Loads the trained pipeline and performs predictions.
    """

    def __init__(self, model_path: Path = MODEL_FILE):

        self.model_path = model_path

        self.model = self.load_model()
        

    def load_model(self):

        logger.info("Loading trained model")

        return joblib.load(self.model_path)
    


    def predict(self, data: pd.DataFrame) -> dict:

        prediction = self.model.predict(data)[0]

        probability = self.model.predict_proba(data)[0]

        return {

    "prediction":

        "Likely to Churn"

        if prediction==1

        else

        "Not Likely to Churn",

    "probability":

        round(

            probability[1]*100,

            2,

        ),
    }





   