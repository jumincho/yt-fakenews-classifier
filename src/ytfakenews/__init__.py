"""Transcribe YouTube videos and classify the transcript as REAL or FAKE news.

The command-line interface (``ytfakenews --help``) covers the whole pipeline. The
Python API for classification is intentionally small::

    from ytfakenews import classify_text, load_classifier

    classifier = load_classifier("models/baseline")
    prediction = classify_text("Transcript text ...", classifier)
    print(prediction.label, prediction.p_fake)
"""

from ytfakenews.errors import YTFakeNewsError
from ytfakenews.predict import (
    ChunkScore,
    Classifier,
    Prediction,
    classify_text,
    classify_texts,
    load_classifier,
)

__version__ = "0.1.0"

__all__ = [
    "ChunkScore",
    "Classifier",
    "Prediction",
    "YTFakeNewsError",
    "__version__",
    "classify_text",
    "classify_texts",
    "load_classifier",
]
