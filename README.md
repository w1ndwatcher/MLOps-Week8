# IRIS Data Poisoning Experiment with MLflow

This project demonstrates how label poisoning affects model performance on the IRIS dataset.  
It trains multiple Decision Tree models under different poisoning levels and logs all results in MLflow.

---

## 📁 Project Files

### `data_poisoning.py`
This script contains the function responsible for injecting label noise into the training dataset.

**Purpose**
- Simulates adversarial **label-flipping attacks**.
- Randomly changes a percentage of training labels based on a specified `noise_ratio`.
- Used to evaluate model robustness under poisoned training conditions.

**Key Function**
- `inject_label_noise(labels, noise_ratio)`
  - `labels`: pandas Series of true labels  
  - `noise_ratio`: fraction of labels to flip (e.g., 0.05 → 5%)  
  - Returns a modified label Series with injected noise.

---

### `train.py`
This is the **main training pipeline** for the project.

**Responsibilities**
- Loads the IRIS dataset from `data/iris.csv`.
- Splits into training and testing sets (test set remains clean).
- Applies different poisoning levels (0%, 10%, 50%).
- Trains Decision Tree models with multiple hyperparameters.
- Logs all experiments to **MLflow**, including:
  - Hyperparameters
  - Poison levels
  - Accuracy metrics
  - Model artifacts
- Registers each trained model version in the MLflow Model Registry.
- Selects the **best-performing model** from MLflow based on accuracy.
- Saves the best model locally at: