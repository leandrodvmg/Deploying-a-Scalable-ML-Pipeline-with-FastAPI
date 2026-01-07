import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# Load data
project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)
# Split data into train and test sets
train, test = train_test_split(data, test_size=0.30, random_state=42)

# Specify categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Scale features (assume continuous features are the first columns before one-hot columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Apply scaler to test features
X_test = scaler.transform(X_test)

# Train model
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
scaler_path = os.path.join(project_path, "model", "scaler.pkl")
save_model(scaler, scaler_path)

# load the model
model = load_model(
    model_path
) 

# Make inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# Remove previous slice output if present
if os.path.exists("slice_output.txt"):
    os.remove("slice_output.txt")
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            "salary",
            encoder,
            lb,
            model,
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
