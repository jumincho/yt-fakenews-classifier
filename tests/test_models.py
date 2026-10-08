"""Opening model directories through the backend registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ytfakenews.artifacts import MANIFEST_FILE, read_manifest
from ytfakenews.errors import ModelLoadError
from ytfakenews.models import BACKENDS, Classifier, load_classifier
from ytfakenews.models.baseline import BaselineClassifier


def test_backends() -> None:
    assert BACKENDS == ("baseline", "transformer")


def test_load_classifier_opens_a_baseline_directory(baseline_dir: Path) -> None:
    classifier = load_classifier(baseline_dir)
    assert isinstance(classifier, BaselineClassifier)
    assert isinstance(classifier, Classifier)
    assert classifier.backend == "baseline"


def test_models_saved_by_version_0_1_still_load(baseline_dir: Path, tmp_path: Path) -> None:
    # The manifest as ytfakenews 0.1.0 wrote it, without the "environment" field.
    shutil.copy(baseline_dir / "model.joblib", tmp_path)
    manifest = {
        "backend": "baseline",
        "package_version": "0.1.0",
        "created_at": "2026-09-23T16:54:55+00:00",
        "config": {"ngram_max": 2, "min_df": 3, "max_df": 0.9, "c": 32.0, "seed": 42},
        "data": {"path": "data/fake_or_real_news.zip", "split": {"seed": 42}},
        "format_version": 1,
    }
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    classifier = load_classifier(tmp_path)
    assert classifier.predict_proba(["officials said the budget passed"]).shape == (1,)
    assert read_manifest(tmp_path).environment == {}


def test_load_classifier_needs_a_model_directory(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="model directory not found"):
        load_classifier(tmp_path / "missing")
    with pytest.raises(ModelLoadError, match="not a ytfakenews model directory"):
        load_classifier(tmp_path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("{not json", "not valid JSON"),
        ('["a", "list"]', "does not contain a JSON object"),
        ('{"format_version": 99, "backend": "baseline"}', "format_version 99, expected 1"),
        ('{"format_version": 1}', "does not name a backend"),
        ('{"format_version": 1, "backend": "baseline", "data": [1]}', "'data' must be a JSON"),
        ('{"format_version": 1, "backend": "svm"}', "unknown backend 'svm'"),
    ],
)
def test_load_classifier_rejects_bad_manifests(tmp_path: Path, manifest: str, message: str) -> None:
    (tmp_path / MANIFEST_FILE).write_text(manifest, encoding="utf-8")
    with pytest.raises(ModelLoadError, match=message):
        load_classifier(tmp_path)
