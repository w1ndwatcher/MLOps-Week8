import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import os

# NEW: import your rewritten poisoning function
from data_poisoning import inject_label_noise

# Configuration
MODEL_TRACKING_URI = "http://127.0.0.1:8100/"
MODEL_NAME = "IRIS-classifier-dt-w8"
LOCAL_MODEL_PATH = "model/model.joblib"

mlflow.set_tracking_uri(MODEL_TRACKING_URI)
mlflow.set_experiment("Data Poisoning: Graded Assignment Week 8")

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
print("Loading data...")
data = pd.read_csv('MLOps-Week6/data/iris.csv')
print("Data loaded successfully!")
print(data.head())

# Split once → training labels only will be poisoned
train_df, test_df = train_test_split(
    data, 
    test_size=0.4, 
    stratify=data['species'], 
    random_state=42
)

X_train = train_df[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train_df.species

X_test = test_df[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test_df.species


# -------------------------------------------------------
# Define hyperparameters & poisoning levels
# -------------------------------------------------------
param_grid = [
    {"max_depth": 2, "random_state": 1},
    {"max_depth": 3, "random_state": 1},
    {"max_depth": 4, "random_state": 1}
]

# NEW: define levels of label noise
poison_levels = [0.0, 0.10, 0.50]

print("\nStarting training runs with poisoning variations...\n")

# -------------------------------------------------------
# Outer loop → iterate through poison levels
# Inner loop → hyperparameter tuning
# -------------------------------------------------------
for poison_level in poison_levels:

    print("====================================================")
    print(f"   Running experiments with poison level: {poison_level*100:.1f}%")
    print("====================================================")

    # NEW: produce a poisoned copy of y_train
    y_train_poisoned = inject_label_noise(y_train.copy(), poison_level)

    # hyperparameter tuning loop
    for params in param_grid:

        with mlflow.start_run():

            # Train using possibly-poisoned training labels
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train_poisoned)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log parameters + poison_level
            mlflow.log_param("poison_level", poison_level)
            mlflow.log_params(params)

            mlflow.log_metric("accuracy", acc)
            mlflow.set_tag("Training_Info", "Decision Tree with label noise experiment")

            signature = infer_signature(X_train, model.predict(X_train))

            # Register version
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=X_train
            )

            print(f"Completed: params={params}, poison_level={poison_level}, accuracy={acc:.3f}")


print("\nAll experiments completed and logged to MLflow successfully!\n")


# -------------------------------------------------------
# Fetch best model version using accuracy comparison
# -------------------------------------------------------
print("Fetching best model version from MLflow Registry...")

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

best_version = None
best_accuracy = -1

for v in versions:
    run_id = v.run_id
    run_metrics = client.get_run(run_id).data.metrics
    acc = run_metrics.get("accuracy", -1)

    if acc > best_accuracy:
        best_accuracy = acc
        best_version = v

if not best_version:
    raise ValueError(f"No registered versions found for model: {MODEL_NAME}")

print(f"Best version found: v{best_version.version} (accuracy={best_accuracy:.3f})")

# -------------------------------------------------------
# Load and save best model locally
# -------------------------------------------------------
best_model_uri = f"models:/{MODEL_NAME}/{best_version.version}"
best_model = mlflow.pyfunc.load_model(model_uri=best_model_uri)

os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
joblib.dump(best_model, LOCAL_MODEL_PATH)

print(f"Best model saved to: {LOCAL_MODEL_PATH}")
print("Training complete. FastAPI can now load this model for inference.")