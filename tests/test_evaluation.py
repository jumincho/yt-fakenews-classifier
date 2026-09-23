from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from ytfakenews.evaluation import compute_metrics, evaluate_classifier


class WordCountClassifier:
    """P(fake) grows with the number of words, so truncation is visible."""

    backend = "stub"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.seen.extend(texts)
        return np.array([min(len(text.split()) / 10, 1.0) for text in texts])


def test_compute_metrics_on_a_known_example() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert metrics["n"] == 4
    assert metrics["accuracy"] == metrics["precision"] == metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert metrics["confusion_matrix"]["values"] == [[1, 1], [1, 1]]
    assert metrics["confusion_matrix"]["labels"] == ["REAL", "FAKE"]


def test_compute_metrics_threshold_and_single_class() -> None:
    assert compute_metrics([1, 1], [0.4, 0.6], threshold=0.3)["recall"] == 1.0
    assert compute_metrics([1, 1], [0.4, 0.6])["roc_auc"] is None


@pytest.mark.parametrize(("y", "p"), [([], []), ([0, 1], [0.5]), ([[0, 1]], [[0.5, 0.5]])])
def test_compute_metrics_rejects_bad_input(y: list[int], p: list[float]) -> None:
    with pytest.raises(ValueError, match=r"empty|1-D"):
        compute_metrics(y, p)


def test_evaluate_document_mode() -> None:
    classifier = WordCountClassifier()
    metrics = evaluate_classifier(classifier, ["one two", " ".join(["w"] * 12)], [0, 1])
    assert metrics["input"] == {"mode": "document"}
    assert metrics["accuracy"] == 1.0
    assert classifier.seen == ["one two", " ".join(["w"] * 12)]


def test_evaluate_chunked_mode() -> None:
    classifier = WordCountClassifier()
    metrics = evaluate_classifier(
        classifier, [" ".join(["w"] * 12)], [1], chunked=True, chunk_words=4, overlap=0
    )
    assert metrics["input"] == {"mode": "chunked", "chunk_words": 4, "overlap": 0}
    assert classifier.seen == ["w w w w"] * 3


def test_evaluate_on_truncated_texts() -> None:
    classifier = WordCountClassifier()
    metrics = evaluate_classifier(classifier, [" ".join(["w"] * 12)], [1], max_words=3)
    assert classifier.seen == ["w w w"]
    assert metrics["input"] == {"mode": "document", "max_words": 3}
    assert metrics["recall"] == 0.0
    with pytest.raises(ValueError, match="max_words"):
        evaluate_classifier(classifier, ["w"], [1], max_words=0)
