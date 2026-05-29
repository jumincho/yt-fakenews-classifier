"""Fine-tune XLM-RoBERTa on the REAL/FAKE news corpus.

CLI:
    python -m src.train --epochs 3 --batch-size 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from .data import (
    DEFAULT_DATA_DIR,
    NewsDataset,
    load_news_dataframe,
    split_train_val_test,
)
from .model import DEFAULT_MODEL_NAME, load_classifier, load_tokenizer


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = Path("./output")
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    seed: int = 42


def _compute_metrics(pred):
    preds = pred.predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1": f1_score(pred.label_ids, preds, average="binary"),
    }


def train(cfg: TrainConfig) -> Trainer:
    """Train + evaluate on test split. Saves the model under `output_dir/final`."""
    df = load_news_dataframe(cfg.data_dir)
    train_df, val_df, test_df = split_train_val_test(df, seed=cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    model = load_classifier(cfg.model_name, num_labels=2)

    train_ds = NewsDataset(train_df, tokenizer, max_length=cfg.max_length)
    val_ds = NewsDataset(val_df, tokenizer, max_length=cfg.max_length)
    test_ds = NewsDataset(test_df, tokenizer, max_length=cfg.max_length)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(cfg.output_dir / "logs"),
        logging_steps=100,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(test_ds)
    print(f"test metrics: {test_metrics}")

    final = cfg.output_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    return trainer


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa on the REAL/FAKE news corpus."
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("./output"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, dest="learning_rate", default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return TrainConfig(**vars(p.parse_args()))


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
