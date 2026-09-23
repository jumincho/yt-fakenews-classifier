from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ytfakenews.artifacts import read_manifest
from ytfakenews.baseline import BaselineClassifier, BaselineConfig, train_baseline
from ytfakenews.data import clean_dataset, split_dataset
from ytfakenews.errors import ModelLoadError
from ytfakenews.predict import Classifier


@pytest.fixture
def fitted(news_frame: pd.DataFrame) -> tuple[BaselineClassifier, pd.DataFrame]:
    splits = split_dataset(clean_dataset(news_frame)[0])
    model = BaselineClassifier.fit(splits.train["text"].tolist(), splits.train["label"].to_numpy())
    return model, splits.test


def test_fit_and_predict(fitted: tuple[BaselineClassifier, pd.DataFrame]) -> None:
    model, test = fitted
    p_fake = model.predict_proba(test["text"].tolist())
    assert p_fake.shape == (len(test),)
    assert np.all((p_fake >= 0) & (p_fake <= 1))
    assert np.array_equal(p_fake >= 0.5, test["label"].to_numpy() == 1)
    assert model.predict_proba([]).shape == (0,)
    assert isinstance(model, Classifier)


def test_top_features_point_to_the_right_label(
    fitted: tuple[BaselineClassifier, pd.DataFrame],
) -> None:
    features = fitted[0].top_features(5)
    assert len(features["FAKE"]) == len(features["REAL"]) == 5
    assert all(weight > 0 for _, weight in features["FAKE"])
    assert all(weight < 0 for _, weight in features["REAL"])


def test_save_and_load_round_trip(
    fitted: tuple[BaselineClassifier, pd.DataFrame], tmp_path: Path
) -> None:
    model, test = fitted
    model.save(tmp_path / "model")
    loaded = BaselineClassifier.load(tmp_path / "model")
    texts = test["text"].tolist()
    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        BaselineClassifier.load(tmp_path)
    joblib.dump({"not": "a pipeline"}, tmp_path / "model.joblib")
    with pytest.raises(ModelLoadError, match="scikit-learn pipeline"):
        BaselineClassifier.load(tmp_path)


def test_train_baseline_writes_model_metrics_and_manifest(news_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "baseline"
    config = BaselineConfig(c=4.0, min_df=2, seed=1)
    metrics = train_baseline(news_zip, out, config=config)

    assert sorted(path.name for path in out.iterdir()) == [
        "manifest.json",
        "metrics.json",
        "model.joblib",
    ]
    assert metrics == json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation"]["n"] == metrics["test"]["n"] == 6
    assert metrics["test"]["accuracy"] == 1.0
    assert metrics["data"]["cleaning"]["rows_kept"] == 60

    manifest = read_manifest(out)
    assert manifest.backend == "baseline"
    assert manifest.config["c"] == 4.0
    assert manifest.data["split"]["seed"] == 42
