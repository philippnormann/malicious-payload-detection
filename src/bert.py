"""Train and evaluate a RoBERTa model for payload classification."""
import logging
import time
from functools import partial

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class PayloadDataset(torch.utils.data.Dataset):
    """Dataset for web payload classification."""

    def __init__(self, encodings, labels, texts):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_and_preprocess_data(filepath, tokenizer):
    """Load and preprocess data for training or evaluation."""
    logging.info("Loading data from %s", filepath)
    data = pd.read_parquet(filepath)
    texts = data["payload"].tolist()
    labels = [1 if label == "malicious" else 0 for label in data["label"].tolist()]

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    dataset = PayloadDataset(encodings, labels, texts)
    return dataset


def compute_metrics(pred):
    """Compute classification metrics."""
    labels = pred.label_ids
    probs = F.softmax(torch.from_numpy(pred.predictions), dim=-1).numpy()
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }


def main(cfg):
    """Train and evaluate a RoBERTa model for payload classification."""
    # Create subdirectory for model run
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg["output_dir"] = f"{cfg['output_dir']}/{timestamp}"
    logging.info("Starting model run %s", cfg["output_dir"])

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(cfg["roberta_model"])
    model = RobertaForSequenceClassification.from_pretrained(
        cfg["roberta_model"],
        num_labels=2,
        **cfg["model_params"],
    )

    # Load data
    train_dataset = load_and_preprocess_data(cfg["train_file"], tokenizer)
    test_dataset = load_and_preprocess_data(cfg["test_file"], tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["training_args"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training_args"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training_args"]["per_device_eval_batch_size"],
        learning_rate=cfg["training_args"]["learning_rate"],
        logging_steps=cfg["training_args"]["logging_steps"],
        logging_dir=f"{cfg['output_dir']}/logs",
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        eval_steps=500,
        save_steps=500,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=partial(compute_metrics),
    )

    trainer.train()

    # Evaluate on test set
    results = trainer.evaluate(test_dataset, metric_key_prefix="eval")
    logging.info("Evaluation Results: %s", results)

    # Save model and tokenizer
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logging.info("Model and Tokenizer saved to %s", cfg["output_dir"])


if __name__ == "__main__":
    config = {
        "train_file": "data/train/processed.parquet",
        "test_file": "data/test/processed.parquet",
        "roberta_model": "microsoft/codebert-base",
        "model_params": {
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
        },
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "logging_steps": 10,
            "learning_rate": 5e-5,
        },
        "output_dir": "data/codebert_for_payloads",
    }

    main(config)
