import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, log_loss, confusion_matrix

# Load test dataset
test = pd.read_csv("test.csv")
X_test = test.iloc[:, :-1]  
y_test = test.iloc[:, -1]    

results = {}

# Load and evaluate each model
for model_name, model_file in [("RandomForest", "random_forest.pkl"), ("LogReg", "logistic_regression.pkl")]:
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    results[model_name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

# Save results
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model evaluation complete. Results saved in 'metrics.json'.")
