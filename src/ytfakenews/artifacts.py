"""On-disk layout of trained model directories.

Every model directory holds a ``manifest.json`` that names the backend and records
the training configuration and data provenance (dataset path, SHA-256, cleaning
statistics and split), plus a ``metrics.json`` with the validation and test results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ytfakenews.errors import ModelLoadError

__all__ = [
    "BACKENDS",
    "MANIFEST_FILE",
    "METRICS_FILE",
    "Manifest",
    "read_manifest",
    "write_json",
    "write_manifest",
]

MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.json"
FORMAT_VERSION = 1
BACKENDS = ("baseline", "transformer")


@dataclass(frozen=True)
class Manifest:
    """Contents of ``manifest.json``."""

    backend: str
    package_version: str
    created_at: str
    config: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON with a trailing newline."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")


def write_manifest(
    directory: str | Path, *, backend: str, config: dict[str, Any], data: dict[str, Any]
) -> Manifest:
    """Write ``manifest.json`` into a model directory and return it."""
    from ytfakenews import __version__  # imported here: the package imports this module

    manifest = Manifest(
        backend=backend,
        package_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config,
        data=data,
    )
    write_json(Path(directory) / MANIFEST_FILE, asdict(manifest))
    return manifest


def read_manifest(directory: str | Path) -> Manifest:
    """Read and validate a model directory's ``manifest.json``."""
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
    if backend not in BACKENDS:
        raise ModelLoadError(f"{path} names unknown backend {backend!r}")
    return Manifest(
        backend=str(backend),
        package_version=str(payload.get("package_version", "unknown")),
        created_at=str(payload.get("created_at", "")),
        config=dict(payload.get("config") or {}),
        data=dict(payload.get("data") or {}),
    )
