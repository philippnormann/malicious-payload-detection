import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.rf import evaluate_classifier, load_data, train_classifier

# Example data
example_data = {
    "feature1": [0.1, 0.2, 0.3],
    "feature2": [0.1, 0.2, 0.3],
    "payload": ["a", "b", "c"],
    "label": ["benign", "malicious", "benign"],
    "category": ["x", "y", "z"],
}

# Example DataFrame
example_df = pd.DataFrame(example_data)


@pytest.fixture
def sample_params():
    return {"n_estimators": 10, "criterion": "entropy", "max_depth": 5, "random_state": 42}


@pytest.fixture
def sample_classifier(sample_params):
    return RandomForestClassifier(**sample_params)


def test_load_data(monkeypatch):
    def mock_read_parquet(*args, **kwargs):
        return example_df

    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)
    x, y = load_data("fakepath")

    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == len(y)
    assert len(x.columns) == 2
    assert len(x) == 3
    assert list(y) == ["benign", "malicious", "benign"]


def test_train_classifier(sample_params):
    x, y = example_df[["feature1", "feature2"]], example_df["label"]
    clf = train_classifier(x, y, sample_params)

    predicted = clf.predict(x)

    assert predicted.shape[0] == y.shape[0]
    assert predicted[0] in ["benign", "malicious"]

    assert clf.n_estimators == sample_params["n_estimators"]
    assert clf.criterion == sample_params["criterion"]
    assert clf.max_depth == sample_params["max_depth"]


def test_evaluate_classifier(sample_classifier):
    x, y = example_df[["feature1", "feature2"]], example_df["label"]
    y_pred, y_pred_proba, metrics = evaluate_classifier(sample_classifier.fit(x, y), x, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    assert y_pred.shape[0] == y.shape[0]
    assert y_pred_proba.shape[0] == y.shape[0]
    assert y_pred_proba.shape[1] == 2
    assert min(y_pred_proba[:, 1]) >= 0
    assert max(y_pred_proba[:, 1]) <= 1
    assert y_pred[0] in ["benign", "malicious"]

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
