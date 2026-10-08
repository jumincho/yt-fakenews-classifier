"""Transcribe YouTube videos and classify the transcript as REAL or FAKE news.

The command-line interface (``ytfakenews --help``) covers the whole pipeline. The
Python API for classification is intentionally small::

    from ytfakenews import classify_text, load_classifier

    classifier = load_classifier("models/baseline")
    prediction = classify_text("Transcript text ...", classifier)
    print(prediction.label, prediction.p_fake)

The rest lives in subpackages: :mod:`ytfakenews.models` (the two classifier backends
and their training functions), :mod:`ytfakenews.asr` (speech-to-text with yt-dlp and
faster-whisper, behind the ``asr`` extra) and :mod:`ytfakenews.cli`.
"""

from ytfakenews._version import __version__
from ytfakenews.errors import YTFakeNewsError
from ytfakenews.models import Classifier, load_classifier
from ytfakenews.predict import ChunkScore, Prediction, classify_text, classify_texts

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
