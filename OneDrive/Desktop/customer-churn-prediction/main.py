from src.data_loader import DataLoader
from src.explainer import ModelExplainer

loader = DataLoader()

df = loader.clean_data(loader.load_data())

X = df.drop(columns=["customerID", "Churn"])

explainer = ModelExplainer()

result = explainer.explain_prediction(X.iloc[[0]])

print(result)