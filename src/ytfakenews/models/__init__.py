"""Classifier backends and the protocol they share.

A backend turns texts into P(FAKE). Two are available:

- ``baseline`` (:mod:`ytfakenews.models.baseline`): TF-IDF + logistic regression; trains
  in seconds on a CPU.
- ``transformer`` (:mod:`ytfakenews.models.transformer`): a fine-tuned transformer
  encoder, multilingual XLM-RoBERTa by default; needs the ``transformer`` extra.

Each backend trains into a model directory whose ``manifest.json`` names the backend
(see :mod:`ytfakenews.artifacts`), so :func:`load_classifier` opens a directory of either
kind. A backend module is only imported when one of its models is loaded, which keeps
the transformer's dependencies optional.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ytfakenews.artifacts import MANIFEST_FILE, Manifest, read_manifest
from ytfakenews.config import DEFAULT_BASELINE_DIR
from ytfakenews.errors import ModelLoadError

__all__ = ["BACKENDS", "Classifier", "load_classifier"]


@runtime_checkable
class Classifier(Protocol):
    """A trained model that returns, for every input text, the probability it is FAKE.

    A classifier scores each text as given. Cleaning, chunking and averaging the chunks
    of a long transcript is done for every backend alike by
    :func:`ytfakenews.predict.classify_text`.
    """

    backend: str
    """The backend's name, as recorded in the manifest of its model directories."""

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return ``P(FAKE)`` for each text, as a 1-D array of the same length."""
        ...


def _load_baseline(directory: Path, manifest: Manifest, device: str) -> Classifier:
    # The baseline runs on the CPU and keeps its settings inside model.joblib.
    from ytfakenews.models.baseline import BaselineClassifier

    return BaselineClassifier.load(directory)


def _load_transformer(directory: Path, manifest: Manifest, device: str) -> Classifier:
    from ytfakenews.models.transformer import TransformerClassifier

    return TransformerClassifier.load(directory, device=device, manifest=manifest)


# Backend name, as written to manifest.json -> function that loads such a directory.
_LOADERS: dict[str, Callable[[Path, Manifest, str], Classifier]] = {
    "baseline": _load_baseline,
    "transformer": _load_transformer,
}
BACKENDS: tuple[str, ...] = tuple(_LOADERS)


def load_classifier(
    model_dir: str | Path = DEFAULT_BASELINE_DIR, *, device: str = "auto"
) -> Classifier:
    """Load a model directory written by either backend.

    ``device`` only applies to the transformer backend (``auto``, ``cpu``, ``cuda``,
    ``cuda:1``, ``mps``, ...). Model files are deserialised with pickle (joblib) or
    PyTorch, so only load models you trained yourself or otherwise trust.
    """
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    loader = _LOADERS.get(manifest.backend)
    if loader is None:
        raise ModelLoadError(
            f"{model_dir / MANIFEST_FILE} names unknown backend {manifest.backend!r}; "
            f"this version of ytfakenews knows {', '.join(BACKENDS)}"
        )
    return loader(model_dir, manifest, device)
