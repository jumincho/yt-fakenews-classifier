"""Tokenizer and binary classifier factory."""

from __future__ import annotations

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


DEFAULT_MODEL_NAME = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def load_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    num_labels: int = 2,
) -> PreTrainedModel:
    """Load the backbone, swap the 3-way NLI head for a binary head.

    The pretrained checkpoint ships with a 3-way head. `num_labels=2` plus
    `ignore_mismatched_sizes=True` re-initializes the classifier to match
    the REAL/FAKE task instead of silently reusing the NLI weights.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
