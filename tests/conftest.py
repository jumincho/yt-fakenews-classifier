"""Shared fixtures: small synthetic datasets and a trained baseline, all offline."""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

_REAL_WORDS = (
    "officials said the committee report budget percent tuesday senator statement "
    "data vote agency confirmed according spokesperson quarterly"
).split()
_FAKE_WORDS = (
    "shocking truth exposed secret they hide share wake up elite coverup bombshell "
    "insider leaked unbelievable globalist hoax"
).split()
_SHARED_WORDS = "the a of and to in city people new year".split()


def make_news_frame(n_per_label: int = 30, seed: int = 0) -> pd.DataFrame:
    """Deterministic toy corpus whose two classes use different vocabularies."""
    rng = random.Random(seed)
    rows = []
    for label, vocabulary in (("REAL", _REAL_WORDS), ("FAKE", _FAKE_WORDS)):
        words = vocabulary + _SHARED_WORDS
        for index in range(n_per_label):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(40, 80)))
            rows.append({"title": f"{label.title()} story {index}", "text": text, "label": label})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def write_zipped_csv(frame: pd.DataFrame, directory: Path, name: str = "news") -> Path:
    """Write ``frame`` like the bundled dataset: a CSV with an unnamed index, zipped."""
    csv_path = directory / f"{name}.csv"
    frame.to_csv(csv_path)
    zip_path = directory / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)
    csv_path.unlink()
    return zip_path


@pytest.fixture
def news_frame() -> pd.DataFrame:
    return make_news_frame()


@pytest.fixture
def news_zip(tmp_path: Path, news_frame: pd.DataFrame) -> Path:
    return write_zipped_csv(news_frame, tmp_path)


@pytest.fixture(scope="session")
def baseline_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A baseline model trained once per test session on the toy corpus."""
    from ytfakenews.baseline import train_baseline

    root = tmp_path_factory.mktemp("baseline")
    data = write_zipped_csv(make_news_frame(), root)
    train_baseline(data, root / "model")
    return root / "model"
