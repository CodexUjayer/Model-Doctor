# Doctors

The ModelDoctor pipeline is powered by specialized analysis modules called **Doctors**. Each Doctor is responsible for evaluating a specific dimension of model health.

---

## OverfittingDoctor

**Purpose:** Detects if the model has memorized the training data and fails to generalize to unseen data.

- **Signals collected:** Generalization Gap, Memorization, CV Variance, Excessive Capacity, Unrestricted Tree Depth.
- **Typical findings:** `Generalization Gap` (Train accuracy is significantly higher than test accuracy).
- **Typical recommendations:** Apply regularization, reduce tree depth, or collect more training data.
- **Supported models:** Classification and Regression.

---

## LeakageDoctor

**Purpose:** Identifies target leakage where future information or identifiers are mistakenly included in the feature set.

- **Signals collected:** High Correlation, Feature Importance Concentration.
- **Typical findings:** `Potential Data Leakage Detected` (A single feature perfectly predicts the target).
- **Typical recommendations:** Remove the leaky feature or timestamp from the training data.
- **Supported models:** Classification and Regression.

---

## DataDoctor

**Purpose:** Assesses the foundational quality of the dataset before the model is even evaluated.

- **Signals collected:** Missing Values, Class Imbalance, Duplicate Rows, Duplicate Columns.
- **Typical findings:** `Severe Class Imbalance`, `High Missing Value Ratio`.
- **Typical recommendations:** Apply SMOTE, adjust class weights, or impute missing values.
- **Supported models:** Classification and Regression.

---

## FeatureDoctor

**Purpose:** Analyzes feature engineering quality and dimensionality.

- **Signals collected:** High Dimensionality, Constant Features, Feature Importance Concentration.
- **Typical findings:** `Constant Features` (Features with zero variance).
- **Typical recommendations:** Drop zero-variance features, apply PCA, or perform feature selection.
- **Supported models:** Classification and Regression.

---

## PredictionDoctor

**Purpose:** Evaluates raw predictive power and identifies if the model is learning anything useful.

- **Signals collected:** Test Accuracy, F1 Score, R² Score.
- **Typical findings:** `Poor Prediction Quality Detected` (Model performs no better than random guessing).
- **Typical recommendations:** Ensure data is normalized, tune hyperparameters, or switch to a more complex model architecture.
- **Supported models:** Classification and Regression.

---

## CalibrationDoctor

**Purpose:** Determines if a classifier's predicted probabilities actually reflect real-world likelihoods.

- **Signals collected:** Expected Calibration Error (ECE), Brier Score, Overconfidence.
- **Typical findings:** `Poor Calibration` (Model is highly confident but frequently wrong).
- **Typical recommendations:** Apply Platt Scaling or Isotonic Regression to calibrate probabilities.
- **Supported models:** Classification only (models must implement `predict_proba`).

---

## ProductionDoctor

**Purpose:** Evaluates whether the model is physically suited for a production deployment environment.

- **Signals collected:** Model Size, Inference Latency.
- **Typical findings:** `Large Serialized Model` (Model exceeds 100MB), `Inference Latency` (Predictions take >500ms).
- **Typical recommendations:** Prune trees, switch to a lighter framework, or serve via C++ runtime (ONNX).
- **Supported models:** Classification and Regression.

---

## GeneralizationDoctor

**Purpose:** Analyzes cross-validation stability and dataset split integrity.

- **Signals collected:** CV Variance, Small Validation Set.
- **Typical findings:** `High CV Variance` (Model performance fluctuates wildly across folds).
- **Typical recommendations:** Increase dataset size or use Stratified K-Fold.
- **Supported models:** Classification and Regression.
