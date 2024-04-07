import os

import pandas as pd
import nltk
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tfidf import fit_tfidf, transform_tfidf

# Sample data
sample_data = pd.DataFrame({"payload": ["This is a test payload", "Another test payload"]})
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


@pytest.fixture
def sample_vectorizer():
    return fit_tfidf(sample_data, {"max_features": 5})


def test_fit_tfidf(sample_vectorizer):
    # Check if function returns TfidfVectorizer and has been fitted
    assert isinstance(sample_vectorizer, TfidfVectorizer)
    assert hasattr(sample_vectorizer, "vocabulary_")
    assert len(sample_vectorizer.vocabulary_) == 5


def test_transform_tfidf(sample_vectorizer):
    # Check if function returns data with TF-IDF features added
    transformed_data = transform_tfidf(sample_data, sample_vectorizer)
    assert len(transformed_data) == 2
    assert len(transformed_data.columns) == 6  # 5 TF-IDF features + 1 payload column
    assert [f"tfidf_{i}" for i in range(5)] == list(transformed_data.columns[1:6])
