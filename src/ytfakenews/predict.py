"""Backend-independent classification: cleanup, chunking, aggregation and model loading.

A transcript can be much longer than what a model handles well (or was trained on), so
:func:`classify_text` cleans it, splits it into overlapping word windows
(:func:`ytfakenews.text.chunk_spans`), scores every window and reports the **mean** of
the window probabilities as ``P(fake)``. The label is ``FAKE`` when that mean is at
least the threshold (0.5 by default). The mean treats every part of the transcript
equally, so one sensational passage in a long, sober video moves the verdict only in
proportion to its length; the per-chunk scores are returned so such passages are
still visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ytfakenews.artifacts import read_manifest
from ytfakenews.errors import ModelLoadError
from ytfakenews.text import chunk_spans, clean_text

__all__ = [
    "AGGREGATION",
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_OVERLAP",
    "DEFAULT_THRESHOLD",
    "ChunkScore",
    "Classifier",
    "Prediction",
    "classify_text",
    "classify_texts",
    "load_classifier",
]

DEFAULT_MODEL_DIR = Path("models") / "baseline"
DEFAULT_CHUNK_WORDS = 300
DEFAULT_OVERLAP = 50
DEFAULT_THRESHOLD = 0.5
AGGREGATION = "mean"
_PREVIEW_WORDS = 12


@runtime_checkable
class Classifier(Protocol):
    """A trained model that returns, for every input text, the probability it is FAKE."""

    backend: str

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return ``P(FAKE)`` for each text, as a 1-D array of the same length."""
        ...


@dataclass(frozen=True)
class ChunkScore:
    """Score of one word window; ``start_word``/``end_word`` index the cleaned text."""

    index: int
    start_word: int
    end_word: int
    p_fake: float
    preview: str


@dataclass(frozen=True)
class Prediction:
    """Document-level verdict plus the per-chunk scores it was aggregated from."""

    label: str
    p_fake: float
    threshold: float
    n_words: int
    chunk_words: int
    overlap: int
    chunks: tuple[ChunkScore, ...]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        payload = asdict(self)
        payload["n_chunks"] = self.n_chunks
        payload["aggregation"] = AGGREGATION
        return payload


def _preview(words: Sequence[str]) -> str:
    text = " ".join(words[:_PREVIEW_WORDS])
    return text + " ..." if len(words) > _PREVIEW_WORDS else text


def classify_texts(
    texts: Sequence[str],
    classifier: Classifier,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
    clean: bool = True,
) -> list[Prediction]:
    """Classify several texts; all chunks are scored in a single batch.

    Raises :class:`ValueError` if a text has no words left after cleaning.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    documents: list[tuple[list[str], list[tuple[int, int]]]] = []
    chunks: list[str] = []
    for position, text in enumerate(texts):
        words = (clean_text(text) if clean else text).split()
        if not words:
            raise ValueError(f"text #{position} is empty after cleaning; nothing to classify")
        spans = chunk_spans(len(words), max_words=chunk_words, overlap=overlap)
        documents.append((words, spans))
        chunks.extend(" ".join(words[start:end]) for start, end in spans)
    if not chunks:
        return []

    scores = np.asarray(classifier.predict_proba(chunks), dtype=float)
    if scores.shape != (len(chunks),):
        raise ValueError(f"classifier returned shape {scores.shape} for {len(chunks)} chunks")

    predictions = []
    offset = 0
    for words, spans in documents:
        doc_scores = scores[offset : offset + len(spans)]
        offset += len(spans)
        chunk_scores = tuple(
            ChunkScore(
                index=index,
                start_word=start,
                end_word=end,
                p_fake=float(score),
                preview=_preview(words[start:end]),
            )
            for index, ((start, end), score) in enumerate(zip(spans, doc_scores, strict=True))
        )
        p_fake = float(np.mean(doc_scores))
        predictions.append(
            Prediction(
                label="FAKE" if p_fake >= threshold else "REAL",
                p_fake=p_fake,
                threshold=threshold,
                n_words=len(words),
                chunk_words=chunk_words,
                overlap=overlap,
                chunks=chunk_scores,
            )
        )
    return predictions


def classify_text(
    text: str,
    classifier: Classifier,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
    clean: bool = True,
) -> Prediction:
    """Classify one text (typically a transcript); see the module docstring."""
    (prediction,) = classify_texts(
        [text],
        classifier,
        chunk_words=chunk_words,
        overlap=overlap,
        threshold=threshold,
        clean=clean,
    )
    return prediction


def load_classifier(
    model_dir: str | Path = DEFAULT_MODEL_DIR, *, device: str = "auto"
) -> Classifier:
    """Load a trained model directory of any backend.

    ``device`` only applies to the transformer backend (``auto``, ``cpu``, ``cuda``, ...).
    Model files are deserialised with pickle/joblib or PyTorch, so only load models you
    trained yourself or otherwise trust.
    """
    manifest = read_manifest(model_dir)
    if manifest.backend == "baseline":
        from ytfakenews.baseline import BaselineClassifier  # imports scikit-learn

        return BaselineClassifier.load(model_dir)
    raise ModelLoadError(f"unsupported backend {manifest.backend!r} in {model_dir}")
