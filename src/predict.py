"""Predict REAL/FAKE for a text file using a fine-tuned model directory.

CLI:
    python -m src.predict path/to/transcript.txt --model-dir ./output/final
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .data import LABEL_NAMES
from .model import load_classifier, load_tokenizer


def classify_text(
    text: str,
    *,
    model_dir: Path,
    tokenizer_dir: Optional[Path] = None,
    max_length: int = 512,
) -> str:
    """Return `"REAL"` or `"FAKE"` for the given text."""
    tokenizer = load_tokenizer(str(tokenizer_dir or model_dir))
    model = load_classifier(str(model_dir), num_labels=2)
    model.eval()
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits
    return LABEL_NAMES[int(torch.argmax(logits, dim=1).item())]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify a .txt file as REAL or FAKE.")
    p.add_argument("text_file", type=Path)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    text = args.text_file.read_text(encoding="utf-8")
    print(classify_text(
        text,
        model_dir=args.model_dir,
        max_length=args.max_length,
    ))


if __name__ == "__main__":
    main()
