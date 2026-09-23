"""TF-IDF + logistic-regression baseline.

Word uni- and bigrams with sublinear term frequency feed an L2-regularised logistic
regression. It trains in seconds on a CPU and is the default model of the CLI.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ytfakenews.artifacts import METRICS_FILE, write_json, write_manifest
from ytfakenews.data import DEFAULT_DATA_PATH, SplitConfig, prepare_splits
from ytfakenews.errors import ModelLoadError
from ytfakenews.evaluation import evaluate_classifier

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "MODEL_FILE",
    "BaselineClassifier",
    "BaselineConfig",
    "build_pipeline",
    "train_baseline",
]

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("models") / "baseline"
MODEL_FILE = "model.joblib"


@dataclass(frozen=True)
class BaselineConfig:
    """Hyper-parameters of the baseline.

    ``c`` (inverse regularisation strength) was chosen on the validation split; the
    validation F1 is flat for values between roughly 16 and 256.
    """

    ngram_max: int = 2
    min_df: int = 3
    max_df: float = 0.9
    c: float = 32.0
    max_iter: int = 1000
    seed: int = 42


def build_pipeline(config: BaselineConfig | None = None) -> Pipeline:
    """Create the unfitted scikit-learn pipeline."""
    config = config or BaselineConfig()
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, config.ngram_max),
                    min_df=config.min_df,
                    max_df=config.max_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=config.c,
                    solver="liblinear",
                    max_iter=config.max_iter,
                    random_state=config.seed,
                ),
            ),
        ]
    )


class BaselineClassifier:
    """Fitted TF-IDF + logistic-regression pipeline behind the common interface."""

    backend = "baseline"

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def fit(
        cls, texts: Sequence[str], labels: ArrayLike, config: BaselineConfig | None = None
    ) -> BaselineClassifier:
        pipeline = build_pipeline(config)
        pipeline.fit(list(texts), np.asarray(labels, dtype=int))
        return cls(pipeline)

    @property
    def n_features(self) -> int:
        return len(self.pipeline.named_steps["tfidf"].vocabulary_)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if len(texts) == 0:
            return np.empty(0, dtype=np.float64)
        probabilities = self.pipeline.predict_proba(list(texts))
        fake_column = list(self.pipeline.classes_).index(1)
        return np.asarray(probabilities[:, fake_column], dtype=np.float64)

    def top_features(self, n: int = 15) -> dict[str, list[list[Any]]]:
        """The ``n`` n-grams with the largest weights towards each label."""
        vocabulary = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        weights = self.pipeline.named_steps["logreg"].coef_[0]
        order = np.argsort(weights)
        return {
            "FAKE": [[str(vocabulary[i]), round(float(weights[i]), 3)] for i in order[::-1][:n]],
            "REAL": [[str(vocabulary[i]), round(float(weights[i]), 3)] for i in order[:n]],
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``model.joblib`` into ``directory`` (created if needed)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tfidf = self.pipeline.named_steps["tfidf"]
        # Older scikit-learn keeps every pruned term for introspection; inference does
        # not need it and it can dominate the file size.
        if getattr(tfidf, "stop_words_", None) is not None:
            tfidf.stop_words_ = None
        path = directory / MODEL_FILE
        joblib.dump(self.pipeline, path, compress=3)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> BaselineClassifier:
        """Load ``model.joblib``. Joblib files are pickles: only load trusted models."""
        path = Path(directory) / MODEL_FILE
        if not path.is_file():
            raise ModelLoadError(f"baseline model file not found: {path}")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, Pipeline):
            raise ModelLoadError(f"{path} does not contain a scikit-learn pipeline")
        return cls(pipeline)


def train_baseline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    config: BaselineConfig | None = None,
    split: SplitConfig | None = None,
) -> dict[str, Any]:
    """Train on the train split, evaluate on validation and test, save everything.

    Writes ``model.joblib``, ``metrics.json`` and ``manifest.json`` to ``output_dir``
    and returns the metrics.
    """
    config = config or BaselineConfig()
    splits, provenance = prepare_splits(data_path, split)

    logger.info("Fitting TF-IDF + logistic regression on %d articles", len(splits.train))
    started = time.perf_counter()
    classifier = BaselineClassifier.fit(
        splits.train["text"].tolist(), splits.train["label"].to_numpy(), config
    )
    fit_seconds = time.perf_counter() - started

    metrics: dict[str, Any] = {"backend": "baseline"}
    for name, frame in (("validation", splits.val), ("test", splits.test)):
        metrics[name] = evaluate_classifier(
            classifier, frame["text"].tolist(), frame["label"].to_numpy()
        )
    metrics.update(
        {
            "fit_seconds": round(fit_seconds, 2),
            "n_features": classifier.n_features,
            "top_features": classifier.top_features(),
            "data": provenance,
        }
    )

    output_dir = Path(output_dir)
    classifier.save(output_dir)
    write_json(output_dir / METRICS_FILE, metrics)
    write_manifest(output_dir, backend="baseline", config=asdict(config), data=provenance)
    logger.debug("Saved baseline model to %s", output_dir)
    return metrics
