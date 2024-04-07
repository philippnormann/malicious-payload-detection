from collections import namedtuple

import pandas as pd
import pytest
import torch
from transformers import RobertaTokenizer

from src.bert import PayloadDataset, compute_metrics, load_and_preprocess_data


@pytest.fixture
def sample_encodings():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    texts = ["example payload", "another payload"]
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    return encodings


def test_PayloadDataset(sample_encodings):
    sample_labels = [1, 0]
    sample_texts = ["example payload", "another payload"]
    dataset = PayloadDataset(sample_encodings, sample_labels, sample_texts)

    # Testing length method
    assert len(dataset) == 2

    # Testing get item method
    item = dataset[0]
    assert torch.equal(item["input_ids"], sample_encodings["input_ids"][0])
    assert item["labels"] == sample_labels[0]


def mock_read_parquet(*args, **kwargs):
    """Mock function to replace pd.read_parquet."""
    data = {"payload": ["<script>alert('hello')</script>", "normalText"], "label": ["malicious", "benign"]}
    return pd.DataFrame(data)


def test_load_and_preprocess_data(
    monkeypatch,
):
    """Test load_and_preprocess_data with mocked parquet reading."""
    # Replace pd.read_parquet with our mock function
    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

    sample_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = load_and_preprocess_data("fakepath", sample_tokenizer)

    # Test if dataset contains expected data
    assert len(dataset) == 2  # as we provided two examples
    assert dataset[0]["labels"] == 1  # as per our mock function, first example should be malicious
    assert dataset.texts[0] == "<script>alert('hello')</script>"  # text check


def test_compute_metrics():
    Pred = namedtuple("Pred", ["predictions", "label_ids"])
    pred = Pred(predictions=torch.tensor([[0.1, 0.9], [0.9, 0.1]]).numpy(), label_ids=torch.tensor([1, 0]).numpy())
    metrics = compute_metrics(pred)

    # Basic checks
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # As we provided perfect predictions, all metrics should be 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

    # Test with non-perfect predictions
    pred = Pred(predictions=torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]).numpy(), label_ids=torch.tensor([1, 1, 0]).numpy())
    metrics = compute_metrics(pred)

    assert metrics["accuracy"] == 1 / 3
    assert metrics["precision"] == 1 / 2
    assert metrics["recall"] == 1 / 2
    assert metrics["f1"] == 1 / 2
    assert metrics["roc_auc"] == 1 / 4
