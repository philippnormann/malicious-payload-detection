"""End-to-end pipeline for preprocessing and training, evaluation."""
import logging
import json
import argparse
import src.preprocess as preprocess
import src.tfidf as tfidf
import src.rf as rf
import src.bert as bert

logging.basicConfig(level=logging.INFO)


def main(config_file):
    """End-to-end pipeline for preprocessing and training, evaluation and saving a payload classifier."""
    # Read config
    logging.info("Reading config from %s", config_file)
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Preprocess data
    if "preprocessing" in cfg:
        logging.info("-" * 80)
        logging.info("Preprocessing data")
        preprocess.main(cfg["preprocessing"])

    # Create TF-IDF features
    if "tfidf" in cfg:
        logging.info("-" * 80)
        logging.info("Creating TF-IDF features")
        tfidf.main(cfg["tfidf"])

    # Train and evaluate Random Forest classifier
    if "random_forest" in cfg:
        logging.info("-" * 80)
        logging.info("Training and evaluating Random Forest classifier")
        rf.main(cfg["random_forest"])

    # Train and evaluate RoBERTa model
    if "bert" in cfg:
        logging.info("-" * 80)
        logging.info("Training and evaluating RoBERTa model")
        bert.main(cfg["bert"])

    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for preprocessing and training, evaluation and saving a payload classifier."
    )
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    args = parser.parse_args()
    main(args.config)
