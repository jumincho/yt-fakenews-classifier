"""Evaluation metrics shared by all backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ytfakenews.data import LABELS
from ytfakenews.predict import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP,
    DEFAULT_THRESHOLD,
    classify_texts,
)

if TYPE_CHECKING:
    from ytfakenews.predict import Classifier

__all__ = ["compute_metrics", "evaluate_classifier"]


def compute_metrics(
    y_true: ArrayLike, p_fake: ArrayLike, *, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """Binary classification metrics with FAKE (label 1) as the positive class.

    Returns accuracy, precision, recall, F1, ROC-AUC (``None`` when only one class is
    present) and the confusion matrix (rows = true label, columns = prediction, both
    in the order REAL, FAKE).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_fake, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError(f"y_true and p_fake must be 1-D of equal length, got {y.shape}, {p.shape}")
    if y.size == 0:
        raise ValueError("cannot compute metrics on an empty set")
    y_pred = (p >= threshold).astype(int)
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])
    roc_auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return {
        "n": int(y.size),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "threshold": threshold,
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows": "true",
            "columns": "predicted",
            "values": matrix.astype(int).tolist(),
        },
    }


def evaluate_classifier(
    classifier: Classifier,
    texts: Sequence[str],
    labels: ArrayLike,
    *,
    chunked: bool = False,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Score ``texts`` with ``classifier`` and compute :func:`compute_metrics`.

    By default every text is scored as one document, exactly as the model saw its
    training articles. With ``chunked=True`` each text goes through the same
    clean -> chunk -> average path that is used for transcripts.
    """
    texts = list(texts)
    if chunked:
        predictions = classify_texts(
            texts, classifier, chunk_words=chunk_words, overlap=overlap, threshold=threshold
        )
        p_fake = np.array([prediction.p_fake for prediction in predictions])
        mode: dict[str, Any] = {"mode": "chunked", "chunk_words": chunk_words, "overlap": overlap}
    else:
        p_fake = np.asarray(classifier.predict_proba(texts), dtype=float)
        mode = {"mode": "document"}
    metrics = compute_metrics(labels, p_fake, threshold=threshold)
    metrics["input"] = mode
    return metrics
