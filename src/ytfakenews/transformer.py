"""Fine-tune and run a transformer encoder (default: multilingual XLM-RoBERTa).

The classifier is trained on English articles only. A multilingual backbone is used
because it maps many languages into a shared representation, so the fine-tuned model
can in principle be applied to transcripts in other languages (zero-shot
cross-lingual transfer). How well that works for this task has not been measured.

Long articles are shortened with head+tail truncation: the first ``head_tokens`` and
the last ``max_length - head_tokens - special`` tokens are kept, which works better
than keeping only the head for long-document classification (Sun et al., 2019, "How
to Fine-Tune BERT for Text Classification?"). Training uses the Hugging Face
``Trainer`` with early stopping on validation F1.

Requires the ``transformer`` extra; torch and transformers are imported lazily, so
the rest of the package works without them.
"""

from __future__ import annotations

import inspect
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ytfakenews._optional import require
from ytfakenews.artifacts import METRICS_FILE, Manifest, read_manifest, write_json, write_manifest
from ytfakenews.data import DEFAULT_DATA_PATH, ID2LABEL, LABEL2ID, SplitConfig, prepare_splits
from ytfakenews.errors import ModelLoadError, YTFakeNewsError
from ytfakenews.evaluation import compute_metrics, evaluate_classifier

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_DIR",
    "HeadTailEncoder",
    "TransformerClassifier",
    "TransformerConfig",
    "special_affixes",
    "train_transformer",
    "truncate_head_tail",
]

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "xlm-roberta-base"
DEFAULT_OUTPUT_DIR = Path("models") / "transformer"


@dataclass(frozen=True)
class TransformerConfig:
    """Fine-tuning settings. The defaults target a single 16 GB GPU (e.g. a Colab T4)."""

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    head_tokens: int = 128
    epochs: float = 4.0
    batch_size: int = 8
    grad_accum_steps: int = 2
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    seed: int = 42
    fp16: bool | None = None
    """Mixed precision; ``None`` enables it whenever a CUDA GPU is used."""
    max_train_samples: int | None = None
    """Train on a random subset (for quick experiments); validation/test stay complete."""
    cpu: bool = False


# ----------------------------------------------------------------------------- encoding


def truncate_head_tail(ids: Sequence[int], head: int, tail: int) -> list[int]:
    """Keep the first ``head`` and the last ``tail`` items of a long sequence."""
    if len(ids) <= head + tail:
        return list(ids)
    return list(ids[:head]) + (list(ids[len(ids) - tail :]) if tail else [])


def special_affixes(tokenizer: Any) -> tuple[list[int], list[int]]:
    """The special token ids a tokenizer puts before and after a single sequence.

    Found by encoding a probe text with and without special tokens, which works for
    any tokenizer (BERT's ``[CLS] ... [SEP]``, RoBERTa's ``<s> ... </s>``, ...).
    """
    probe = "a short probe text"
    plain = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    full = list(tokenizer(probe)["input_ids"])
    for start in range(len(full) - len(plain) + 1):
        if full[start : start + len(plain)] == plain:
            return full[:start], full[start + len(plain) :]
    raise ValueError("cannot locate the text inside the tokenizer's special tokens")


class HeadTailEncoder:
    """Tokenise texts to at most ``max_length`` ids with head+tail truncation."""

    def __init__(self, tokenizer: Any, *, max_length: int = 512, head_tokens: int = 128) -> None:
        self.tokenizer = tokenizer
        self.prefix, self.suffix = special_affixes(tokenizer)
        budget = max_length - len(self.prefix) - len(self.suffix)
        if budget < 1:
            raise ValueError(f"max_length={max_length} leaves no room for text tokens")
        if not 0 <= head_tokens <= budget:
            raise ValueError(f"head_tokens must be between 0 and {budget}, got {head_tokens}")
        self.head = head_tokens
        self.tail = budget - head_tokens

    def encode(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self.tokenizer(list(texts), add_special_tokens=False, verbose=False)
        return [
            self.prefix + truncate_head_tail(ids, self.head, self.tail) + self.suffix
            for ids in encoded["input_ids"]
        ]


# ---------------------------------------------------------------------------- inference


def resolve_device(device: str = "auto") -> Any:
    """``auto`` picks CUDA, then Apple MPS, then the CPU."""
    torch = require("torch", extra="transformer")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise YTFakeNewsError(f"invalid device {device!r}: {exc}") from exc


class TransformerClassifier:
    """A fine-tuned sequence-classification model behind the common interface."""

    backend = "transformer"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int = 512,
        head_tokens: int = 128,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = HeadTailEncoder(tokenizer, max_length=max_length, head_tokens=head_tokens)
        self.batch_size = batch_size
        self.fake_index = int(model.config.label2id.get("FAKE", LABEL2ID["FAKE"]))

    @classmethod
    def load(
        cls, directory: str | Path, *, device: str = "auto", manifest: Manifest | None = None
    ) -> TransformerClassifier:
        """Load a directory written by :func:`train_transformer`."""
        # torch first: transformers imports without it but then cannot load models.
        require("torch", extra="transformer")
        transformers = require("transformers", extra="transformer")
        manifest = manifest or read_manifest(directory)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(directory))
            model = transformers.AutoModelForSequenceClassification.from_pretrained(str(directory))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(
                f"could not load the transformer model in {directory}: {exc}"
            ) from exc
        model.to(resolve_device(device))
        model.eval()
        return cls(
            model,
            tokenizer,
            max_length=int(manifest.config.get("max_length", 512)),
            head_tokens=int(manifest.config.get("head_tokens", 128)),
        )

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        torch = require("torch", extra="transformer")
        encoded = self.encoder.encode(texts) if len(texts) else []
        scores = np.empty(len(encoded), dtype=np.float64)
        # Batch texts of similar length together to minimise padding.
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch = self.tokenizer.pad(
                    {"input_ids": [encoded[index] for index in batch_indices]},
                    return_tensors="pt",
                )
                logits = self.model(
                    **{key: value.to(device) for key, value in batch.items()}
                ).logits
                probabilities = torch.softmax(logits.float(), dim=-1)[:, self.fake_index]
                scores[batch_indices] = probabilities.cpu().numpy()
        self.model.train(was_training)
        return scores


# ----------------------------------------------------------------------------- training


def _trainer_metrics(eval_prediction: Any) -> dict[str, float]:
    """``compute_metrics`` hook for the Trainer (logits -> the shared metrics)."""
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_fake = exp[:, LABEL2ID["FAKE"]] / exp.sum(axis=1)
    metrics = compute_metrics(eval_prediction.label_ids, p_fake)
    names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    return {name: float(metrics[name]) for name in names if metrics[name] is not None}


def _training_arguments(config: TransformerConfig, output_dir: Path) -> Any:
    torch = require("torch", extra="transformer")
    transformers = require("transformers", extra="transformer")
    use_cuda = torch.cuda.is_available() and not config.cpu
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "logging_steps": 50,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": use_cuda if config.fp16 is None else config.fp16,
        "use_cpu": config.cpu,
    }
    # transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio).
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = config.warmup_ratio
    else:
        kwargs["warmup_steps"] = config.warmup_ratio
    return transformers.TrainingArguments(**kwargs)


def _features(encoder: HeadTailEncoder, frame: pd.DataFrame) -> list[dict[str, Any]]:
    encoded = encoder.encode(frame["text"].tolist())
    labels = frame["label"].tolist()
    return [
        {"input_ids": ids, "labels": int(label)} for ids, label in zip(encoded, labels, strict=True)
    ]


def train_transformer(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    config: TransformerConfig | None = None,
    split: SplitConfig | None = None,
) -> dict[str, Any]:
    """Fine-tune ``config.model_name`` and save model, tokenizer, metrics and manifest.

    ``model_name`` is a Hugging Face Hub id (downloaded on first use) or a local
    directory. A fresh two-way classification head is initialised (an existing head
    of a different size, e.g. from an NLI checkpoint, is replaced). The checkpoint
    with the best validation F1 is kept; intermediate checkpoints are deleted.
    """
    config = config or TransformerConfig()
    require("torch", extra="transformer")
    transformers = require("transformers", extra="transformer")
    require("accelerate", extra="transformer")
    splits, provenance = prepare_splits(data_path, split)
    train_frame = splits.train
    if config.max_train_samples is not None and config.max_train_samples < len(train_frame):
        train_frame = train_frame.sample(n=config.max_train_samples, random_state=config.seed)

    transformers.set_seed(config.seed)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    except (OSError, ValueError) as exc:  # not found, offline, no classification head, ...
        raise ModelLoadError(
            f"could not load {config.model_name!r} (a Hugging Face Hub id or a local "
            f"directory): {exc}"
        ) from exc
    try:
        encoder = HeadTailEncoder(
            tokenizer, max_length=config.max_length, head_tokens=config.head_tokens
        )
    except ValueError as exc:
        raise YTFakeNewsError(f"invalid truncation settings: {exc}") from exc

    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    trainer = transformers.Trainer(
        model=model,
        args=_training_arguments(config, checkpoints),
        train_dataset=_features(encoder, train_frame),
        eval_dataset=_features(encoder, splits.val),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=_trainer_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )
    logger.info(
        "Fine-tuning %s on %d articles (max %s epochs)",
        config.model_name,
        len(train_frame),
        config.epochs,
    )
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started

    # The Trainer has reloaded the best checkpoint; score it through the same code
    # path that `ytfakenews predict` uses.
    classifier = TransformerClassifier(
        trainer.model,
        tokenizer,
        max_length=config.max_length,
        head_tokens=config.head_tokens,
        batch_size=config.eval_batch_size,
    )
    metrics: dict[str, Any] = {"backend": "transformer", "base_model": config.model_name}
    for name, frame in (("validation", splits.val), ("test", splits.test)):
        metrics[name] = evaluate_classifier(
            classifier, frame["text"].tolist(), frame["label"].to_numpy()
        )
    metrics.update(
        {
            "train_seconds": round(train_seconds, 1),
            "train_samples": len(train_frame),
            "epochs_run": trainer.state.epoch,
            "best_validation_f1": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
            "data": provenance,
        }
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.rmtree(checkpoints, ignore_errors=True)
    write_json(output_dir / METRICS_FILE, metrics)
    write_manifest(output_dir, backend="transformer", config=asdict(config), data=provenance)
    logger.debug("Saved transformer model to %s", output_dir)
    return metrics
