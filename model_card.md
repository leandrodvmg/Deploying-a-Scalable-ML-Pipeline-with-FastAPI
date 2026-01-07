# Model Card

This document describes the machine learning model included in this repository. It summarizes the model architecture, intended use, training and evaluation data, metrics, ethical considerations, and caveats. Where accurate numeric metrics are required they are marked as placeholders; run the provided training script to compute the exact values.

**Model Details**:
- **Model type**: `LogisticRegression` (scikit-learn implementation).
- **Input features**: Raw tabular features from the U.S. Census dataset in `data/census.csv` plus one-hot encoded categorical features.
- **Categorical features**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
- **Preprocessing**: Categorical features are encoded with `OneHotEncoder` (trained during preprocessing). The label is binarized using `LabelBinarizer`.
- **Artifacts produced**: Serialized model `model/model.pkl` and encoder `model/encoder.pkl` when `train_model.py` is run.

**Intended Use**:
- **Primary use case**: Predict whether an individual's annual income is greater than 50K based on demographic and employment features in the provided Census dataset.
- **Primary user**: Data scientists and engineers exploring a simple classification baseline and a demonstration of a deployable ML pipeline (FastAPI integration in the repository).
- **Out-of-scope uses**: This model is not suitable as-is for high-stakes decision making (loan approvals, hiring, insurance underwriting) without further validation, calibration, fairness analysis, and domain-specific regulatory review.

**Training Data**:
- **Source**: `data/census.csv` included in this repository. See [data/census.csv](data/census.csv) for full contents.
- **Preprocessing**: The `process_data` function in `ml/data.py` performs one-hot encoding for categorical variables and label binarization. Continuous features are left as-is (no scaling is applied).
- **Train/test split**: A random 70/30 split is used (`train_test_split(test_size=0.30, random_state=42)`) as implemented in [train_model.py](train_model.py).

**Evaluation Data**:
- The held-out 30% test split from `data/census.csv` is used for model evaluation when running `train_model.py`.
- Per-slice evaluation across categorical feature values is computed by `performance_on_categorical_slice` in `ml/model.py` and written to `slice_output.txt` by `train_model.py`.

**Metrics**:
- **Metrics used**: Precision, Recall, and F1-score (computed via `precision_score`, `recall_score`, and `fbeta_score` with `beta=1` in `ml/model.py`).

  - **Precision**: 0.7178
  - **Recall**: 0.2627
  - **F1-score**: 0.3847
-- **Test set (aggregated)** (results observed when I ran `train_model.py` in the workspace after adding feature scaling and increasing `max_iter`):
  - **Precision**: 0.7357
  - **Recall**: 0.6050
  - **F1-score**: 0.6640

-- **Training notes**: After adding `StandardScaler` for continuous features and increasing `max_iter` for `LogisticRegression`, the model's recall and F1 improved substantially. These changes are implemented in `train_model.py` and `ml/model.py` (see `max_iter=500` and scaler usage). Consider further hyperparameter tuning and cross-validation for additional gains.

-- **Training notes**: When the model was trained the solver issued a convergence warning (`lbfgs failed to converge`). Consider increasing `max_iter` or scaling continuous features (e.g., `StandardScaler`) to improve convergence and possibly recall.

Instructions to generate the numeric metrics locally:
- Ensure dependencies are installed (see `requirements.txt`). Example:

```bash
pip install -r requirements.txt
python train_model.py
```

Running `train_model.py` will:
- Train a `LogisticRegression` model on the 70% training split.
- Save artifacts to `model/model.pkl` and `model/encoder.pkl`.
- Print the aggregated precision, recall and F1 for the test set to stdout.
- Append per-slice precision/recall/F1 values to `slice_output.txt` in the repository root.

**Ethical Considerations**:
- **Potential bias**: The dataset contains demographic attributes (race, sex, native-country, etc.) that correlate with socioeconomic outcomes. Without careful fairness analysis and mitigation, the model could reproduce or amplify existing biases present in the training data.
- **Privacy**: The dataset appears to be a public UCI-style census extract. If deploying with real user data, ensure compliance with relevant privacy regulations and consider de-identification where appropriate.
- **Explainability**: Logistic regression offers interpretable coefficients for numerical features; however, after one-hot encoding of categorical variables, the feature space is high-dimensional and explanations require mapping one-hot columns back to human-readable categories.

**Caveats and Recommendations**:
- **No calibration or hyperparameter tuning**: The current training uses the default `LogisticRegression` hyperparameters. Hyperparameter tuning and cross-validation are recommended to improve generalization.
- **No feature scaling**: Continuous features are not scaled; for some models this can affect convergence and performance. Consider adding scaling (e.g., `StandardScaler`) if substituting different model families.
- **Limited evaluation**: Only aggregate precision/recall/F1 and categorical-slice performance are computed. Add additional metrics (ROC AUC, precision-recall curve, calibration metrics) and confusion matrices for a more complete evaluation.
- **Fairness testing**: Run subgroup fairness checks (e.g., disparate impact, equalized odds) across protected attributes such as `race` and `sex` before any deployment.

If you would like, I can:
- Install dependencies and run `train_model.py` to compute and write the exact metrics into this model card.
- Or, if you prefer to run locally, paste the printed metrics here and I will update `model_card.md` with the exact values and per-slice summaries.

---

This model card was generated from the code in the repository (notably `ml/model.py`, `ml/data.py`, and `train_model.py`) and the included dataset. For implementation details, see the referenced files.
