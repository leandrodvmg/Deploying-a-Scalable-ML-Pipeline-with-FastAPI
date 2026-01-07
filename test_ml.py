import os
import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


def test_process_data_returns_expected_types():
    """process_data should return (X, y, encoder, lb) with correct types."""
    data_path = os.path.join("data", "census.csv")
    df = pd.read_csv(data_path)
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
    X, y, encoder, lb = process_data(df.head(50), categorical_features=cat_features, label="salary", training=True)

    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be an OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be a LabelBinarizer"


def test_train_model_returns_logistic_regression():
    """train_model should return a fitted LogisticRegression instance."""
    data_path = os.path.join("data", "census.csv")
    df = pd.read_csv(data_path)
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
    X, y, encoder, lb = process_data(df.head(200), categorical_features=cat_features, label="salary", training=True)
    # train a small model
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), "model should be a LogisticRegression instance"


def test_inference_and_metrics_behave_as_expected():
    """inference should return an array and compute_model_metrics should return numeric metrics."""
    # Create a small synthetic example for deterministic metrics
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Precision = 1/(1) = 1.0, Recall = 1/2 = 0.5, F1 = 2 * (1*0.5)/(1+0.5) = 0.666...
    assert pytest.approx(precision, rel=1e-3) == 1.0
    assert pytest.approx(recall, rel=1e-3) == 0.5
    assert pytest.approx(fbeta, rel=1e-3) == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5), rel=1e-3)

    # Test inference output type for a trivial model
    # Train on tiny synthetic dataset
    X_train = np.array([[0.0], [1.0], [1.0], [0.0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]


def test_train_test_split_sizes_and_types():
    """Verify that dataset split produces DataFrame outputs and respects approx. 70/30 split."""
    data_path = os.path.join("data", "census.csv")
    df = pd.read_csv(data_path)
    assert isinstance(df, pd.DataFrame)
    n = len(df)
    # perform the split exactly as in train_model.py
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.30, random_state=42)
    # types
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    # proportions within tolerance
    assert abs((len(test) / n) - 0.30) < 0.01

