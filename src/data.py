"""REAL/FAKE news CSV loading and the PyTorch `NewsDataset`."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# fake_or_real_news.csv schema: title, text, label (REAL / FAKE)
LABEL_MAP = {"REAL": 0, "FAKE": 1}
LABEL_NAMES = {0: "REAL", 1: "FAKE"}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _csv_candidates(data_dir: Path) -> List[Path]:
    """Where the CSV might already live (local repo + Colab session)."""
    return [
        data_dir / "fake_or_real_news.csv",
        Path("../fake_or_real_news.csv"),
        Path("/content/data/fake_or_real_news.csv"),
        Path("/content/fake_or_real_news.csv"),
        Path("fake_or_real_news.csv"),
    ]


def _zip_candidates(data_dir: Path) -> List[Path]:
    return [
        data_dir / "dataset.zip",
        Path("data/dataset.zip"),
        Path("/content/data/dataset.zip"),
        Path("/content/dataset.zip"),
        Path("dataset.zip"),
    ]


def find_or_extract_csv(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Return the CSV path, transparently unzipping `dataset.zip` if needed."""
    for p in _csv_candidates(data_dir):
        if p.exists():
            return p
    for z in _zip_candidates(data_dir):
        if z.exists():
            extract_to = z.parent if z.parent.as_posix() else Path(".")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(extract_to)
            break
    for p in _csv_candidates(data_dir):
        if p.exists():
            return p
    raise FileNotFoundError(
        "fake_or_real_news.csv not found. Place data/dataset.zip in the repo "
        "(auto-extracts on first run), or upload the CSV directly in Colab."
    )


def load_news_dataframe(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the CSV, map REAL/FAKE → 0/1, return as DataFrame."""
    df = pd.read_csv(find_or_extract_csv(data_dir))
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["fake"] = df["label"].map(LABEL_MAP)
    return df.drop(columns=["label"])


def split_train_val_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/16/4 → train/val/test by default (`test_size=0.2`, then `val_size=0.2`)."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=val_size, random_state=seed)
    return train, val, test


class NewsDataset(Dataset):
    """Pair-encode (title + text) and serve `fake` labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 512,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = f"{row['title']} {row['text']}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["fake"]), dtype=torch.long),
        }
