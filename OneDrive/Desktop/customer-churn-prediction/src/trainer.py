
import joblib
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
from src.config import MODEL_FILE, RESULTS_FILE
from src.logger import logger
from src.evaluation import run_final_evaluation


class ModelTrainer:

    def __init__(self):

        # Four candidate algorithms
        self.models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Random Forest":
                RandomForestClassifier(random_state=42),

            "Gradient Boosting":
                GradientBoostingClassifier(random_state=42),

            "XGBoost":
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                ),
        }

        # Hyperparameter search spaces
        self.parameters = {

            "Logistic Regression": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
            },

            "Random Forest": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, 20, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },

            "Gradient Boosting": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 5],
                "classifier__min_samples_leaf": [1, 2, 4],
            },

            "XGBoost": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.8, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            },
        }

    # -------------------------------------------------
    # 1. Split data into train, validation and test
    # -------------------------------------------------

    def split_data(self, df):

        X = df.drop(columns=["customerID", "Churn"])

        y = df["Churn"].map({
            "No": 0,
            "Yes": 1,
        })

        if y.isna().any():
            raise ValueError(
                "Churn contains missing or unexpected labels."
            )

        # Separate 20% test data
        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

        # Remaining 80% -> 60% train and 20% validation
        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.25,
                random_state=42,
                stratify=y_train_val,
            )
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
        )

    # -------------------------------------------------
    # 2. Build preprocessing + classifier pipeline
    # -------------------------------------------------

    def build_pipeline(self, model):

        preprocessor = DataPreprocessor().build_pipeline()

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clone(model)),
            ]
        )

    # -------------------------------------------------
    # 3. Calculate metrics using default threshold
    # -------------------------------------------------

    def evaluate(self, pipeline, X, y):

        prediction = pipeline.predict(X)

        probability = pipeline.predict_proba(X)[:, 1]

        return {
            "Accuracy": accuracy_score(y, prediction),

            "Precision": precision_score(
                y, prediction, zero_division=0
            ),

            "Recall": recall_score(
                y, prediction, zero_division=0
            ),

            "F1": f1_score(
                y, prediction, zero_division=0
            ),

            "ROC_AUC": roc_auc_score(y, probability),
        }

    # -------------------------------------------------
    # 4. Train and compare all four models
    # -------------------------------------------------

    def compare_models(self, X_train, y_train, X_val, y_val):

        results = []

        best_name = None
        best_auc = -1
        best_pipeline = None

        for name, model in self.models.items():

            logger.info("Training %s", name)

            pipeline = self.build_pipeline(model)

            pipeline.fit(X_train, y_train)

            metrics = self.evaluate(
                pipeline,
                X_val,
                y_val,
            )

            metrics["Model"] = name
            results.append(metrics)

            logger.info(
                "%s validation ROC-AUC: %.4f",
                name,
                metrics["ROC_AUC"],
            )

            if metrics["ROC_AUC"] > best_auc:

                best_auc = metrics["ROC_AUC"]
                best_name = name
                best_pipeline = pipeline

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="ROC_AUC",
            ascending=False,
        )

        Path(RESULTS_FILE).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            RESULTS_FILE,
            index=False,
        )

        return best_name, best_pipeline, results_df

    # -------------------------------------------------
    # 5. Tune selected algorithm using training data
    # -------------------------------------------------

    def tune_best_model(self, model_name, X_train, y_train):

        print("\nTuning selected model:", model_name)

        selected_pipeline = self.build_pipeline(
            self.models[model_name]
        )

        search = RandomizedSearchCV(
            estimator=selected_pipeline,
            param_distributions=self.parameters[model_name],
            n_iter=10,
            cv=5,
            scoring="roc_auc",
            random_state=42,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nBest CV ROC-AUC:")
        print(search.best_score_)

        return search.best_estimator_

    # -------------------------------------------------
    # 6. Complete training and evaluation workflow
    # -------------------------------------------------

    def train_models(self, df):

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
        ) = self.split_data(df)

        print("Training samples:", len(X_train))
        print("Validation samples:", len(X_val))
        print("Testing samples:", len(X_test))

        # Compare all four models
        best_name, _, results = self.compare_models(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        print("\nModel comparison:")
        print(results.to_string(index=False))

        print("\nSelected algorithm:", best_name)

        # Tune selected algorithm using training data ONLY
        tuned_pipeline = self.tune_best_model(
            best_name,
            X_train,
            y_train,
        )

        # Threshold selection, final refitting and test evaluation
        (
            final_model,
            best_threshold,
            test_metrics,
            importance_df,
        ) = run_final_evaluation(
            tuned_pipeline=tuned_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            output_dir="artifacts/evaluation",
            target_recall=0.70,
            model_file=MODEL_FILE,
        )


        print("\nSelected algorithm:", best_name)
        print("Selected threshold:", best_threshold)

        print("\nFinal test metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nTop 10 feature importances:")
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))

        return (
            results,
            best_name,
            test_metrics,
            best_threshold,
            importance_df,
        )

    # -------------------------------------------------
    # 7. Save model (optional utility)
    # -------------------------------------------------

    def save_model(self, pipeline):

        Path(MODEL_FILE).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            MODEL_FILE,
        )

        logger.info("Final tuned model saved.")
