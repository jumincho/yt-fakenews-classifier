from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ytfakenews.data import (
    LABEL2ID,
    SplitConfig,
    clean_dataset,
    file_sha256,
    load_dataset,
    prepare_splits,
    read_raw_dataset,
    split_dataset,
)
from ytfakenews.errors import DatasetError

FIXTURES = Path(__file__).parent / "fixtures"


def test_label_mapping() -> None:
    assert LABEL2ID == {"REAL": 0, "FAKE": 1}


def test_clean_dataset_normalises_and_reports() -> None:
    df, stats = clean_dataset(read_raw_dataset(FIXTURES / "messy_news.csv"))

    assert list(df.columns) == ["id", "title", "text", "label"]
    assert df["id"].tolist() == [0, 2, 7]
    assert df["title"].tolist() == ["First story", "Third story", "Eighth story"]
    assert df["label"].tolist() == [0, 1, 0]

    assert stats.rows_read == 8
    assert stats.dropped_empty_text == 1
    assert stats.dropped_conflicting_labels == 2
    assert stats.dropped_duplicate_text == 2
    assert stats.identical_rows == 1
    assert stats.rows_kept == 3
    assert stats.label_counts == {"REAL": 2, "FAKE": 1}


def test_clean_dataset_without_optional_columns() -> None:
    raw = pd.DataFrame({"text": ["a b c", "d e f"], "label": ["REAL", "FAKE"]})
    df, _ = clean_dataset(raw)
    assert df["title"].tolist() == ["", ""]
    assert df["id"].tolist() == [0, 1]


def test_clean_dataset_rejects_unknown_labels() -> None:
    raw = pd.DataFrame({"text": ["a", "b"], "label": ["REAL", "SATIRE"]})
    with pytest.raises(DatasetError, match="SATIRE"):
        clean_dataset(raw)


def test_clean_dataset_requires_text_and_label() -> None:
    with pytest.raises(DatasetError, match="label"):
        clean_dataset(pd.DataFrame({"text": ["a"]}))


def test_load_dataset_reads_a_zipped_csv(news_zip: Path, news_frame: pd.DataFrame) -> None:
    df, stats = load_dataset(news_zip)
    assert stats.rows_kept == len(news_frame) == len(df)
    assert stats.label_counts == {"REAL": 30, "FAKE": 30}
    assert df["text"].tolist() == news_frame["text"].tolist()


def test_read_raw_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="dataset not found"):
        read_raw_dataset(tmp_path / "missing.zip")
    archive = tmp_path / "two.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.csv", "text,label\nx,REAL\n")
        handle.writestr("b.csv", "text,label\ny,FAKE\n")
    with pytest.raises(DatasetError, match="could not read"):
        read_raw_dataset(archive)


def test_split_sizes_are_stratified_and_disjoint(news_frame: pd.DataFrame) -> None:
    df, _ = clean_dataset(news_frame)
    splits = split_dataset(df)
    assert splits.sizes() == {"train": 48, "val": 6, "test": 6}
    ids = [set(splits.get(name)["id"]) for name in ("train", "val", "test")]
    assert set.union(*ids) == set(df["id"])
    assert sum(map(len, ids)) == len(df)
    for name in ("val", "test"):
        assert splits.get(name)["label"].value_counts().to_dict() == {0: 3, 1: 3}


def test_split_is_deterministic(news_frame: pd.DataFrame) -> None:
    df, _ = clean_dataset(news_frame)
    first = split_dataset(df, SplitConfig(seed=7))
    second = split_dataset(df, SplitConfig(seed=7))
    other = split_dataset(df, SplitConfig(seed=8))
    assert first.test["id"].tolist() == second.test["id"].tolist()
    assert first.test["id"].tolist() != other.test["id"].tolist()


@pytest.mark.parametrize(("val", "test"), [(0.0, 0.1), (0.1, 1.0), (0.5, 0.5)])
def test_split_config_validation(val: float, test: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        SplitConfig(val_size=val, test_size=test)


def test_split_unknown_name(news_frame: pd.DataFrame) -> None:
    splits = split_dataset(clean_dataset(news_frame)[0])
    with pytest.raises(ValueError, match="unknown split"):
        splits.get("dev")


def test_split_too_small_dataset() -> None:
    df, _ = clean_dataset(
        pd.DataFrame({"text": ["a", "b", "c"], "label": ["REAL", "FAKE", "FAKE"]})
    )
    with pytest.raises(DatasetError, match="cannot split"):
        split_dataset(df)


def test_prepare_splits_records_provenance(news_zip: Path) -> None:
    splits, provenance = prepare_splits(news_zip, SplitConfig(seed=3))
    assert provenance["sha256"] == file_sha256(news_zip)
    assert provenance["split"] == {"val_size": 0.1, "test_size": 0.1, "seed": 3}
    assert provenance["split_sizes"] == splits.sizes()
    assert provenance["cleaning"]["rows_kept"] == 60
