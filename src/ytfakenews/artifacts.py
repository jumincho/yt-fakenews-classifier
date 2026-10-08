"""On-disk layout of trained model directories.

A model directory holds the backend's own files, a ``metrics.json`` with the validation
and test results and a ``manifest.json`` that names the backend and records the
training configuration, the data provenance (dataset path and SHA-256, cleaning
statistics and split) and the Python and library versions. Only a directory with a
manifest is a model, and :func:`save_model_dir` writes the manifest last.

``FORMAT_VERSION`` changes only when a directory written by one version can no longer
be read by the other; a new optional manifest field does not change it.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ytfakenews._version import __version__
from ytfakenews.errors import ModelLoadError

__all__ = [
    "FORMAT_VERSION",
    "MANIFEST_FILE",
    "METRICS_FILE",
    "Manifest",
    "environment",
    "read_manifest",
    "save_model_dir",
    "write_json",
    "write_manifest",
]

MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.json"
FORMAT_VERSION = 1

# Distributions whose versions decide whether a saved model loads and what it predicts.
_RECORDED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "torch",
    "transformers",
    "tokenizers",
)


@dataclass(frozen=True)
class Manifest:
    """Contents of ``manifest.json``."""

    backend: str
    package_version: str
    created_at: str
    config: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    """Python and library versions at training time; empty for older models."""
    format_version: int = FORMAT_VERSION


def environment() -> dict[str, str]:
    """The Python version and the versions of the installed libraries listed above."""
    versions = {"python": platform.python_version()}
    for name in _RECORDED_PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON with a trailing newline."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")


def write_manifest(
    directory: str | Path, *, backend: str, config: dict[str, Any], data: dict[str, Any]
) -> Manifest:
    """Write ``manifest.json`` into a model directory and return it."""
    manifest = Manifest(
        backend=backend,
        package_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config,
        data=data,
        environment=environment(),
    )
    write_json(Path(directory) / MANIFEST_FILE, asdict(manifest))
    return manifest


def save_model_dir(
    directory: str | Path,
    write_model: Callable[[Path], object],
    *,
    backend: str,
    config: dict[str, Any],
    data: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write a complete model directory and return its path.

    ``write_model(directory)`` saves the backend's own files; ``metrics.json`` and then
    ``manifest.json`` follow. The manifest of an earlier model in the same directory is
    deleted before anything is written, so an interrupted run never leaves a manifest
    next to files it does not describe.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).unlink(missing_ok=True)
    write_model(directory)
    write_json(directory / METRICS_FILE, metrics)
    write_manifest(directory, backend=backend, config=config, data=data)
    return directory


def read_manifest(directory: str | Path) -> Manifest:
    """Read and validate a model directory's ``manifest.json``.

    Checks the format version and the type of every field. Whether the backend exists is
    checked by :func:`ytfakenews.models.load_classifier`.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ModelLoadError(
            f"model directory not found: {directory}. Train a model first, "
            "e.g. `ytfakenews train baseline`, or pass --model DIR."
        )
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise ModelLoadError(
            f"{directory} is not a ytfakenews model directory (no {MANIFEST_FILE})"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelLoadError(f"{path} does not contain a JSON object")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ModelLoadError(
            f"{path} has format_version {payload.get('format_version')!r}, expected "
            f"{FORMAT_VERSION}; retrain the model with this version of ytfakenews"
        )
    backend = payload.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ModelLoadError(f"{path} does not name a backend")
    sections: dict[str, dict[str, Any]] = {}
    for key in ("config", "data", "environment"):
        value = payload.get(key)
        if value is not None and not isinstance(value, dict):
            raise ModelLoadError(f"{path}: {key!r} must be a JSON object")
        sections[key] = value or {}
    return Manifest(
        backend=backend,
        package_version=str(payload.get("package_version", "unknown")),
        created_at=str(payload.get("created_at", "")),
        config=sections["config"],
        data=sections["data"],
        environment=sections["environment"],
    )
