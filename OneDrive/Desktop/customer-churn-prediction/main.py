
from pathlib import Path

import pandas as pd

from src.data_loader import DataLoader
from src.trainer import ModelTrainer
from src.explainer import ModelExplainer


#STEP 1: Load and clean dataset
loader = DataLoader()

df = loader.clean_data(
    loader.load_data()
)


# STEP 2: Create trainer
trainer = ModelTrainer()


# STEP 3: Run complete ML workflow
(
    results,
    best_name,
    test_metrics,
    best_threshold,
    importance_df,
) = trainer.train_models(df)


# STEP 4: Display selected model
print("\nBest model:", best_name)


# STEP 5: Display model comparison
print("\nModel comparison:")
print(results.to_string(index=False))


# # STEP 6: Display selected threshold
# print("\nSelected classification threshold:")
# print(best_threshold)


# # STEP 7: Display final test metrics
# print("\nFinal test metrics:")
# for metric, value in test_metrics.items():
#     print(f"{metric}: {value:.4f}")


# # STEP 8: Display feature importance
# print("\nTop 10 important features:")

# if importance_df is not None:
#     print(importance_df.head(10).to_string(index=False))


# # STEP 9: Explain a prediction using saved model
# X = df.drop(columns=["customerID", "Churn"])

# explainer = ModelExplainer()

# result = explainer.explain_prediction(
#     X.iloc[[0]],
#     threshold=best_threshold
# )

# print("\nPrediction explanation:")
# print(result)

# print("\nPrediction explanation:")
# print(result)





