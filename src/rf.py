"""Train and evaluate a Random Forest classifier for payload classification."""
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
sns.set_style("darkgrid")


def load_data(filepath):
    """Load data for training or evaluation."""
    logging.info("Loading data from %s", filepath)
    data = pd.read_parquet(filepath)
    x = data.drop(columns=["payload", "label", "category"])
    y = data["label"]
    return x, y


def train_classifier(x_train, y_train, params):
    """Train Random Forest classifier."""
    logging.info("Training Random Forest classifier")
    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    logging.info("Successfully trained Random Forest classifier")
    return clf


def evaluate_classifier(clf, x_test, y_test):
    """Evaluate classifier on test set."""
    logging.info("Evaluating classifier")
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="malicious"),
        "recall": recall_score(y_test, y_pred, pos_label="malicious"),
        "f1": f1_score(y_test, y_pred, pos_label="malicious"),
        "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
    }
    logging.info("Accuracy: %f", metrics["accuracy"])
    logging.info("Precision: %f", metrics["precision"])
    logging.info("Recall: %f", metrics["recall"])
    logging.info("F1: %f", metrics["f1"])
    logging.info("ROC AUC: %f", metrics["roc_auc"])

    return y_pred, y_pred_proba, metrics


def plot_precision_recall(y_test, y_pred_proba, save_path):
    """Plot Precision-Recall curve."""
    logging.info("Plotting Precision-Recall curve")
    fig, ax = plt.subplots(figsize=(12, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1], pos_label="malicious")
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    fig.savefig(save_path)


def plot_roc(y_test, y_pred_proba, save_path):
    """Plot ROC curve."""
    logging.info("Plotting ROC curve")
    fig, ax = plt.subplots(figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label="malicious")
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    fig.savefig(save_path)


def main(cfg):
    """Train and evaluate a Random Forest classifier for payload classification."""
    # Create subdirectory for model run
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg["output_dir"] = f"{cfg['output_dir']}/{timestamp}"
    logging.info("Starting model run %s", cfg["output_dir"])

    # Load data
    x_train, y_train = load_data(cfg["train_file"])
    x_test, y_test = load_data(cfg["test_file"])

    # Train classifier
    clf = train_classifier(x_train, y_train, cfg["model_params"])

    # Evaluate classifier
    _y_pred, y_pred_proba, metrics = evaluate_classifier(clf, x_test, y_test)

    # Save model and metrics
    output_dir = Path(cfg["output_dir"])
    reports_folder = output_dir / "report"
    reports_folder.mkdir(parents=True, exist_ok=True)

    logging.info("Writing model params to %s", reports_folder / "model_params.json")
    with open(reports_folder / "model_params.json", "w", encoding="utf-8") as f:
        json.dump(cfg["model_params"], f, indent=2)

    logging.info("Creating metric plots in %s", reports_folder)
    plot_precision_recall(y_test, y_pred_proba, reports_folder / "precision_recall.png")
    plot_roc(y_test, y_pred_proba, reports_folder / "roc.png")

    logging.info("Writing metrics to %s", reports_folder / "metric.json")
    with open(reports_folder / "metric.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Saving classifier to %s", output_dir / "classifier.joblib")
    joblib.dump(clf, output_dir / "classifier.joblib")


if __name__ == "__main__":
    # Best params from hyperparameter tuning
    config = {
        "train_file": "models/rf/train/tfidf.parquet",
        "test_file": "models/rf/test/tfidf.parquet",
        "model_params": {
            "n_estimators": 100,
            "criterion": "entropy",
            "max_depth": 500,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 2,
        },
        "output_dir": "models/rf/runs",
    }

    main(config)
