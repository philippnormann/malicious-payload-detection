"""Preprocess web payload dataset."""
import html
import logging
import json
import re
from pathlib import Path
from urllib.parse import unquote_plus

import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def drop_na(data):
    """Drop rows with missing values."""
    rows = len(data)
    data.dropna(inplace=True)
    logging.info("Dropped %d rows with missing values", rows - len(data))


def drop_duplicates(data):
    """Drop rows with duplicate payloads."""
    rows = len(data)
    data.drop_duplicates(subset="payload", inplace=True)
    logging.info("Dropped %d rows with duplicate payloads", rows - len(data))


def url_decode_payload(encoded_string):
    """Decode URL encoded payload."""
    return unquote_plus(encoded_string)


def html_decode_payload(encoded_string):
    """Decode HTML encoded payload."""
    return html.unescape(encoded_string)


def count_payload_len(data):
    """Create feature for payload length."""
    data["payload_len"] = data["payload"].str.len()


def count_special_chars(data):
    """Create feature for number of special characters in payload."""
    data["special_chars_count"] = data["payload"].apply(lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", x)))


def calculate_stats(data):
    """Calculate stats for dataset."""
    label_counts = data["label"].value_counts()
    category_counts = data["category"].value_counts()

    return {
        "num_rows": len(data),
        "label_counts": label_counts.to_dict(),
        "category_counts": category_counts.to_dict(),
    }


def main(cfg):
    """Preprocess dataset."""
    logging.info("Reading dataset from %s", cfg["in_file"])
    data = pd.read_csv(cfg["in_file"])
    logging.info("Raw dataset contains %d rows", len(data))

    if cfg["preprocessing"]["drop_na"]:
        drop_na(data)

    if cfg["preprocessing"]["drop_duplicates"]:
        drop_duplicates(data)

    if cfg["preprocessing"]["create_features"]["payload_len"]:
        count_payload_len(data)

    if cfg["preprocessing"]["create_features"]["special_chars_count"]:
        count_special_chars(data)

    logging.info("Preprocessing payload")
    if cfg["preprocessing"]["preprocess_payload"]["lowercase"]:
        data["payload"] = data["payload"].str.lower()
    if cfg["preprocessing"]["preprocess_payload"]["url_decode"]:
        data["payload"] = data["payload"].apply(url_decode_payload)
    if cfg["preprocessing"]["preprocess_payload"]["html_decode"]:
        data["payload"] = data["payload"].apply(html_decode_payload)

    logging.info("Processed dataset contains %d rows", len(data))

    if cfg["preprocessing"]["train_test_split"]:
        train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])

        train_out_path = Path(cfg["out_folder"]) / "train" / "processed.parquet"
        test_out_path = Path(cfg["out_folder"]) / "test" / "processed.parquet"

        train_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_out_path.parent.mkdir(parents=True, exist_ok=True)

        train.to_parquet(train_out_path)
        train_stats = calculate_stats(train)
        logging.info("Train stats: %s", train_stats)
        with open(train_out_path.parent / "stats.json", "w", encoding="utf-8") as f:
            json.dump(train_stats, f, indent=2)

        test.to_parquet(test_out_path)
        test_stats = calculate_stats(test)
        logging.info("Test stats: %s", test_stats)
        with open(test_out_path.parent / "stats.json", "w", encoding="utf-8") as f:
            json.dump(test_stats, f, indent=2)

    else:
        out_path = Path(cfg["out_folder"])
        data.to_parquet(out_path / "processed.parquet")
        stats = data["label"].value_counts()
        logging.info("Stats: %s", stats)
        with open(out_path / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    config = {
        "in_file": "data/dataset.csv",
        "out_folder": "data",
        "preprocessing": {
            "drop_na": True,
            "drop_duplicates": True,
            "create_features": {
                "payload_len": True,
                "special_chars_count": True,
            },
            "preprocess_payload": {
                "lowercase": True,
                "url_decode": True,
                "html_decode": True,
            },
            "train_test_split": True,
        },
    }

    main(config)
