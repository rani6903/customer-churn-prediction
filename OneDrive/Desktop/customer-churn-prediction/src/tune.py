import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from src.preprocess import DataPreprocessor
from src.config import MODEL_FILE

class ModelTuner:

    def __init__(self):

        self.preprocessor = (
            DataPreprocessor().build_pipeline()
        )

        self.model = RandomForestClassifier(
            random_state=42
        )

    def build_pipeline(self):

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    self.preprocessor
                ),

                (
                    "classifier",
                    self.model
                )
            ]
        )

        return pipeline
    
    def parameter_grid(self):

        return {

            "classifier__n_estimators":

                [100,200,300],

            "classifier__max_depth":

                [5,10,20,None],

            "classifier__min_samples_split":

                [2,5,10],

            "classifier__min_samples_leaf":

                [1,2,4]
        }
    
    def tune(self, X_train, y_train):

        pipeline = self.build_pipeline()

        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=self.parameter_grid(),

            n_iter=10,

            cv=5,

            scoring="roc_auc",

            random_state=42,

            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train
        )

        return search
    
    def save_best_model(self, search):

        joblib.dump(

            search.best_estimator_,

            MODEL_FILE
        )


    