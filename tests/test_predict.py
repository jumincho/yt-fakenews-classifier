from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ytfakenews.artifacts import MANIFEST_FILE
from ytfakenews.baseline import BaselineClassifier
from ytfakenews.errors import ModelLoadError
from ytfakenews.predict import Prediction, classify_text, classify_texts, load_classifier


class KeywordClassifier:
    """Deterministic stand-in: P(fake) is the share of words equal to "fake"."""

    backend = "keyword"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(list(texts))
        return np.array([text.split().count("fake") / len(text.split()) for text in texts])


def test_short_text_is_one_chunk() -> None:
    prediction = classify_text("fake fake real real", KeywordClassifier())
    assert prediction.n_chunks == 1
    assert prediction.p_fake == 0.5
    assert prediction.label == "FAKE"  # the threshold is inclusive
    assert prediction.chunks[0].start_word == 0
    assert prediction.chunks[0].end_word == 4


def test_long_text_is_chunked_and_averaged() -> None:
    text = " ".join(["real"] * 400 + ["fake"] * 200)
    classifier = KeywordClassifier()
    prediction = classify_text(text, classifier, chunk_words=200, overlap=0)

    assert [chunk.p_fake for chunk in prediction.chunks] == [0.0, 0.0, 1.0]
    assert prediction.p_fake == pytest.approx(1 / 3)
    assert prediction.label == "REAL"
    assert prediction.n_words == 600
    assert [(c.start_word, c.end_word) for c in prediction.chunks] == [
        (0, 200),
        (200, 400),
        (400, 600),
    ]
    assert prediction.chunks[0].preview == " ".join(["real"] * 12) + " ..."


def test_threshold_changes_the_label() -> None:
    text = "fake real real real"
    assert classify_text(text, KeywordClassifier()).label == "REAL"
    assert classify_text(text, KeywordClassifier(), threshold=0.25).label == "FAKE"
    with pytest.raises(ValueError, match="threshold"):
        classify_text(text, KeywordClassifier(), threshold=1.5)


def test_text_is_cleaned_before_chunking() -> None:
    prediction = classify_text("[Music] fake >> real <i>real</i>", KeywordClassifier())
    assert prediction.n_words == 3
    raw = classify_text("[Music] fake", KeywordClassifier(), clean=False)
    assert raw.n_words == 2


@pytest.mark.parametrize("text", ["", "   ", "[Music] [Applause]"])
def test_empty_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        classify_text(text, KeywordClassifier())


def test_classify_texts_scores_all_chunks_in_one_batch() -> None:
    classifier = KeywordClassifier()
    predictions = classify_texts(
        ["fake " * 30, "real " * 10, "fake real " * 40], classifier, chunk_words=20, overlap=5
    )
    assert len(classifier.calls) == 1
    assert [p.label for p in predictions] == ["FAKE", "REAL", "FAKE"]
    assert sum(p.n_chunks for p in predictions) == len(classifier.calls[0])
    assert classify_texts([], classifier) == []


def test_classifier_output_shape_is_checked() -> None:
    class Broken(KeywordClassifier):
        def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
            return np.zeros(len(texts) + 1)

    with pytest.raises(ValueError, match="shape"):
        classify_text("some words", Broken())


def test_prediction_to_dict_is_json_serialisable() -> None:
    prediction = classify_text("fake " * 500, KeywordClassifier())
    payload = json.loads(json.dumps(prediction.to_dict()))
    assert payload["label"] == "FAKE"
    assert payload["aggregation"] == "mean"
    assert payload["n_chunks"] == len(payload["chunks"]) == 2
    assert isinstance(prediction, Prediction)


def test_load_classifier_baseline(baseline_dir: Path) -> None:
    classifier = load_classifier(baseline_dir)
    assert isinstance(classifier, BaselineClassifier)
    assert classifier.backend == "baseline"


def test_load_classifier_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="model directory not found"):
        load_classifier(tmp_path / "missing")
    with pytest.raises(ModelLoadError, match="not a ytfakenews model directory"):
        load_classifier(tmp_path)

    manifest = tmp_path / MANIFEST_FILE
    manifest.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_classifier(tmp_path)
    manifest.write_text(json.dumps({"format_version": 99, "backend": "baseline"}), "utf-8")
    with pytest.raises(ModelLoadError, match="format_version"):
        load_classifier(tmp_path)
    manifest.write_text(json.dumps({"format_version": 1, "backend": "svm"}), "utf-8")
    with pytest.raises(ModelLoadError, match="unknown backend"):
        load_classifier(tmp_path)
