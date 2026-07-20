import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier

from src.preprocess import DataPreprocessor

from src.config import (
    MODEL_FILE,
    RESULTS_FILE,
)

from src.logger import logger


class ModelTrainer:

    def __init__(self):

        self.preprocessor = (
            DataPreprocessor().build_pipeline()
        )

        self.models = {

            "Logistic Regression":

                LogisticRegression(
                    max_iter=1000
                ),

            "Random Forest":

                RandomForestClassifier(
                    random_state=42
                ),

            "Gradient Boosting":

                GradientBoostingClassifier(
                    random_state=42
                ),

            "XGBoost":

                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                ),
        }

    def split_data(self, df):
        """
        Split dataset into training and testing sets.
        """

        X = df.drop(columns=["customerID", "Churn"])

        y = df["Churn"].map({
            "No": 0,
            "Yes": 1
        })

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        return X_train, X_test, y_train, y_test


    def build_pipeline(self, model):

        return Pipeline(

            steps=[

                (
                    "preprocessor",
                    self.preprocessor,
                ),

                (
                    "classifier",
                    model,
                )
            ]
        )
    
    def evaluate(self,pipeline,X_test,y_test,model_name,):

        prediction = pipeline.predict(X_test)

        probability = pipeline.predict_proba(X_test)[:,1]

        return {

            "Model": model_name,

            "Accuracy":
                accuracy_score(
                    y_test,
                    prediction,
                ),

            "Precision":
                precision_score(
                    y_test,
                    prediction,
                ),

            "Recall":
                recall_score(
                    y_test,
                    prediction,
                ),

            "F1":
                f1_score(
                    y_test,
                    prediction,
                ),

            "ROC_AUC":
                roc_auc_score(
                    y_test,
                    probability,
                ),
        }
    
    def train_models(self, df):

        X_train, X_test, y_train, y_test = (
            self.split_data(df)
        )

        results = []

        best_pipeline = None

        best_auc = 0

        for name, model in self.models.items():

            pipeline = self.build_pipeline(model)

            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = self.evaluate(
                pipeline,
                X_test,
                y_test,
                name,
            )

            results.append(metrics)

            if metrics["ROC_AUC"] > best_auc:

                best_auc = metrics["ROC_AUC"]

                best_pipeline = pipeline

        results = pd.DataFrame(results)

        results.to_csv(
            RESULTS_FILE,
            index=False,
        )

        self.save_model(
            best_pipeline
        )

        return results
    
    def tune_best_model(
        self,
        X_train,
        y_train,
    ):

        pipeline = self.build_pipeline(

            RandomForestClassifier(
                random_state=42
            )
        )

        parameters = {

            "classifier__n_estimators":

                [100,200,300],

            "classifier__max_depth":

                [5,10,20,None],

            "classifier__min_samples_leaf":

                [1,2,4],
        }

        search = RandomizedSearchCV(

            pipeline,

            parameters,

            cv=5,

            scoring="roc_auc",

            random_state=42,

            n_iter=10,

            n_jobs=-1,
        )

        search.fit(
            X_train,
            y_train,
        )

        return search
    
    def save_model(self, pipeline):

        joblib.dump(

            pipeline,

            MODEL_FILE,
        )