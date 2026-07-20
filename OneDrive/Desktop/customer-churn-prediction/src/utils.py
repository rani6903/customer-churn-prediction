import joblib

pipeline = joblib.load("model/churn_model.pkl")

print(type(pipeline.named_steps["classifier"]))