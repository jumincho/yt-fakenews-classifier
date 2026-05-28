"""YouTube → caption → fake-news classifier pipeline.

Sub-modules:
    src.extract_subtitles  — YouTube URL → MP3 → SRT (WhisperX) → cleaned TXT.
    src.data               — REAL/FAKE news CSV loading + `NewsDataset`.
    src.model              — Tokenizer / classifier factory (binary head).
    src.train              — Fine-tuning pipeline (notebook and CLI).
    src.predict            — Inference on a `.txt` file (notebook and CLI).
"""

__all__ = ["extract_subtitles", "data", "model", "train", "predict"]
