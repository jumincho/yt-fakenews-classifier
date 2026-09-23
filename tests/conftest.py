"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import FakeYouTube, make_news_frame, write_zipped_csv


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


@pytest.fixture
def fake_youtube(monkeypatch: pytest.MonkeyPatch) -> FakeYouTube:
    fake = FakeYouTube()
    monkeypatch.setattr("ytfakenews.transcribe._extract_info", fake.extract_info)
    monkeypatch.setattr("ytfakenews.transcribe._load_whisper_model", fake.load_whisper_model)
    return fake
