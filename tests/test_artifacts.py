"""The files of a model directory: manifest.json and metrics.json."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from ytfakenews import __version__
from ytfakenews.artifacts import (
    FORMAT_VERSION,
    MANIFEST_FILE,
    METRICS_FILE,
    environment,
    read_manifest,
    save_model_dir,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path) -> None:
    written = write_manifest(tmp_path, backend="baseline", config={"c": 4.0}, data={"n": 1})
    assert read_manifest(tmp_path) == written
    assert written.package_version == __version__
    assert written.format_version == FORMAT_VERSION


def test_environment_records_python_and_library_versions() -> None:
    versions = environment()
    assert versions["python"] == platform.python_version()
    assert versions["numpy"] == np.__version__
    assert "scikit-learn" in versions


def test_save_model_dir_writes_the_manifest_last(tmp_path: Path) -> None:
    write_manifest(tmp_path, backend="baseline", config={}, data={})  # an earlier model
    seen: list[list[str]] = []

    def write_model(directory: Path) -> None:
        seen.append(sorted(path.name for path in directory.iterdir()))
        (directory / "weights.bin").write_bytes(b"\x00")

    metrics = {"test": {"f1": 1.0}}
    save_model_dir(
        tmp_path, write_model, backend="baseline", config={"c": 1.0}, data={}, metrics=metrics
    )
    assert seen == [[]]  # the old manifest was gone before the model was written
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        MANIFEST_FILE,
        METRICS_FILE,
        "weights.bin",
    ]
    assert json.loads((tmp_path / METRICS_FILE).read_text(encoding="utf-8")) == metrics
    assert read_manifest(tmp_path).config == {"c": 1.0}


def test_an_interrupted_save_leaves_no_manifest(tmp_path: Path) -> None:
    write_manifest(tmp_path, backend="baseline", config={}, data={})

    def write_model(directory: Path) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_model_dir(tmp_path, write_model, backend="baseline", config={}, data={}, metrics={})
    assert not (tmp_path / MANIFEST_FILE).exists()
