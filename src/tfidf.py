"""Transforms text data into TF-IDF features."""
import logging
import pickle

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def fit_tfidf(data, tfidf_params):
    """Fit TF-IDF Vectorizer on train data."""
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, **tfidf_params)
    tfidf_vectorizer.fit(data["payload"])
    return tfidf_vectorizer


def transform_tfidf(data, tfidf_vectorizer):
    """Transform data with TF-IDF Vectorizer."""
    tfidf_matrix = tfidf_vectorizer.transform(data["payload"])
    tfidf_columns = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns)
    return pd.concat([data.reset_index(drop=True), tfidf_df], axis=1)


def main(cfg):
    """Load data, fit TF-IDF Vectorizer on train data, and transform train and test data."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logging.info("Downloading punkt tokenizer")
        nltk.download("punkt")

    logging.info("Loading train data from %s", cfg["train_file"])
    train_data = pd.read_parquet(cfg["train_file"])
    logging.info("Successfully loaded train data from %s", cfg["train_file"])

    logging.info("Loading test data from %s", cfg["test_file"])
    test_data = pd.read_parquet(cfg["test_file"])
    logging.info("Successfully loaded test data from %s", cfg["test_file"])

    logging.info("Fitting TF-IDF Vectorizer")
    tfidf_vectorizer = fit_tfidf(train_data, cfg["tfidf_params"])
    logging.info("Successfully fitted TF-IDF Vectorizer on train data.")

    logging.info("Saving TF-IDF Vectorizer to %s", cfg["tfidf_vectorizer"])
    with open(cfg["tfidf_vectorizer"], "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    logging.info("Transforming train data with TF-IDF")
    train_tfidf = transform_tfidf(train_data, tfidf_vectorizer)
    train_tfidf.to_parquet(cfg["output_train_tfidf"])
    logging.info("Train data transformed and saved to %s", cfg["output_train_tfidf"])

    logging.info("Transforming test data with TF-IDF")
    test_tfidf = transform_tfidf(test_data, tfidf_vectorizer)
    test_tfidf.to_parquet(cfg["output_test_tfidf"])
    logging.info("Test data transformed and saved to %s", cfg["output_test_tfidf"])


if __name__ == "__main__":
    config = {
        "train_file": "data/train/processed.parquet",
        "test_file": "data/test/processed.parquet",
        "output_train_tfidf": "data/train/tfidf.parquet",
        "output_test_tfidf": "data/test/tfidf.parquet",
        "tfidf_vectorizer": "data/tfidf_vectorizer.pickle",
        "tfidf_params": {"max_features": 5000},
    }

    main(config)
