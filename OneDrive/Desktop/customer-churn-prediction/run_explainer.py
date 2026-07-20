from src.data_loader import DataLoader
from src.explainer import ModelExplainer

loader = DataLoader()

df = loader.clean_data(loader.load_data())

X = df.drop(columns=["customerID", "Churn"])

explainer = ModelExplainer()

explainer.summary_plot(X)

explainer.feature_importance(X)

explainer.waterfall_plot(X.iloc[[0]])

print("SHAP plots generated successfully.")