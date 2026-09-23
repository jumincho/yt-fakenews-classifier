"""Loading, cleaning and splitting the REAL/FAKE news training data.

The bundled dataset is ``data/fake_or_real_news.zip``: a single CSV with ``title``,
``text`` and ``label`` (``REAL``/``FAKE``) columns. Any CSV (or single-file ZIP) with
the same columns can be used instead.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ytfakenews.errors import DatasetError

__all__ = [
    "DEFAULT_DATA_PATH",
    "ID2LABEL",
    "LABEL2ID",
    "LABELS",
    "CleaningStats",
    "SplitConfig",
    "Splits",
    "clean_dataset",
    "file_sha256",
    "load_dataset",
    "prepare_splits",
    "read_raw_dataset",
    "split_dataset",
]

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data") / "fake_or_real_news.zip"
LABELS: tuple[str, str] = ("REAL", "FAKE")
LABEL2ID: dict[str, int] = {"REAL": 0, "FAKE": 1}
ID2LABEL: dict[int, str] = {index: name for name, index in LABEL2ID.items()}


@dataclass(frozen=True)
class CleaningStats:
    """What :func:`clean_dataset` removed, in the order the steps are applied."""

    rows_read: int
    dropped_empty_text: int
    dropped_conflicting_labels: int
    dropped_duplicate_text: int
    identical_rows: int
    """Of the dropped duplicates, rows whose title, text and label all match an earlier row."""
    rows_kept: int
    label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitConfig:
    """Fractions and seed of the stratified train/validation/test split."""

    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0 < self.val_size < 1 and 0 < self.test_size < 1):
            raise ValueError("val_size and test_size must be between 0 and 1")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size must be smaller than 1")


@dataclass(frozen=True)
class Splits:
    """The three data splits; each frame has ``id``, ``title``, ``text`` and ``label``."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def get(self, name: str) -> pd.DataFrame:
        """Return a split by name: ``train``, ``val`` or ``test``."""
        if name not in ("train", "val", "test"):
            raise ValueError(f"unknown split {name!r}; expected 'train', 'val' or 'test'")
        frame: pd.DataFrame = getattr(self, name)
        return frame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, recorded with trained models for provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the dataset CSV as-is. A ``.zip`` holding a single CSV is read directly."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(
            f"dataset not found: {path}. Run the command from the repository root, where "
            f"{DEFAULT_DATA_PATH} lives, or pass --data PATH to a CSV/ZIP with 'title', "
            "'text' and 'label' columns."
        )
    try:
        return pd.read_csv(path)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DatasetError(f"could not read {path}: {exc}") from exc


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Normalise columns and labels and drop unusable rows.

    Steps, in order:

    1. Lower-case and strip column names; the unnamed index column becomes ``id``.
    2. Strip whitespace from ``title`` and ``text``; map labels (case-insensitive)
       ``REAL`` -> 0 and ``FAKE`` -> 1. Any other label is an error.
    3. Drop rows whose ``text`` is empty.
    4. Drop every copy of a ``text`` that occurs with both labels (none in the bundled
       data, but a contradiction should never reach training).
    5. Drop repeated ``text`` values, keeping the first. The article body is what the
       models see, so a body shared by two rows would otherwise leak across splits.
    """
    df = raw.rename(columns=lambda column: str(column).strip().lower())
    if "id" not in df.columns and "unnamed: 0" in df.columns:
        df = df.rename(columns={"unnamed: 0": "id"})
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise DatasetError(f"dataset is missing required column(s): {', '.join(sorted(missing))}")
    if "title" not in df.columns:
        df["title"] = ""
    if "id" not in df.columns:
        df["id"] = range(len(df))
    df = df[["id", "title", "text", "label"]].copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    labels = df["label"].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - set(LABEL2ID))
    if unknown:
        raise DatasetError(f"unknown label value(s) {unknown}; expected REAL or FAKE")
    df["label"] = labels.map(LABEL2ID).astype(int)
    rows_read = len(df)

    empty = df["text"] == ""
    df = df[~empty]

    conflicting = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~conflicting]

    identical_rows = int(df.duplicated(["title", "text", "label"]).sum())
    duplicate = df.duplicated("text", keep="first")
    df = df[~duplicate].reset_index(drop=True)

    counts = df["label"].value_counts()
    stats = CleaningStats(
        rows_read=rows_read,
        dropped_empty_text=int(empty.sum()),
        dropped_conflicting_labels=int(conflicting.sum()),
        dropped_duplicate_text=int(duplicate.sum()),
        identical_rows=identical_rows,
        rows_kept=len(df),
        label_counts={name: int(counts.get(index, 0)) for name, index in LABEL2ID.items()},
    )
    return df, stats


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, CleaningStats]:
    """Read and clean the dataset; see :func:`clean_dataset` for the rules."""
    df, stats = clean_dataset(read_raw_dataset(path))
    logger.info(
        "Loaded %d rows from %s: dropped %d empty, %d conflicting and %d duplicate texts; "
        "kept %d (REAL %d, FAKE %d)",
        stats.rows_read,
        path,
        stats.dropped_empty_text,
        stats.dropped_conflicting_labels,
        stats.dropped_duplicate_text,
        stats.rows_kept,
        stats.label_counts["REAL"],
        stats.label_counts["FAKE"],
    )
    return df, stats


def split_dataset(df: pd.DataFrame, config: SplitConfig | None = None) -> Splits:
    """Deterministic stratified train/validation/test split (80/10/10 by default).

    The validation and test sets each get ``round(fraction * len(df))`` rows; the
    test set is drawn first, then the validation set from the remainder.
    """
    config = config or SplitConfig()
    n_test = max(1, round(config.test_size * len(df)))
    n_val = max(1, round(config.val_size * len(df)))
    try:
        train_val, test = train_test_split(
            df, test_size=n_test, random_state=config.seed, stratify=df["label"]
        )
        train, val = train_test_split(
            train_val, test_size=n_val, random_state=config.seed, stratify=train_val["label"]
        )
    except ValueError as exc:
        raise DatasetError(f"cannot split {len(df)} rows with {config}: {exc}") from exc
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def prepare_splits(
    path: str | Path = DEFAULT_DATA_PATH, config: SplitConfig | None = None
) -> tuple[Splits, dict[str, Any]]:
    """Load, clean and split the dataset; also return provenance to store with a model."""
    config = config or SplitConfig()
    df, stats = load_dataset(path)
    splits = split_dataset(df, config)
    provenance = {
        "path": Path(path).as_posix(),
        "sha256": file_sha256(path),
        "cleaning": asdict(stats),
        "split": asdict(config),
        "split_sizes": splits.sizes(),
    }
    return splits, provenance
