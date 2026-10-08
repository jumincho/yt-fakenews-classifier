"""Training configuration: where data and models live, how the data is split and the
hyper-parameters of both backends.

Every training default is defined here, once, in plain dataclasses that import nothing
heavy. The CLI builds its options from them, the trainers take them as arguments and
each trained model records the values it used in its ``manifest.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_BASELINE_DIR",
    "DEFAULT_DATA_PATH",
    "DEFAULT_TRANSFORMER_DIR",
    "BaselineConfig",
    "SplitConfig",
    "TransformerConfig",
]

DEFAULT_DATA_PATH = Path("data") / "fake_or_real_news.zip"
"""The bundled dataset, relative to the repository root."""

DEFAULT_BASELINE_DIR = Path("models") / "baseline"
"""Where ``train baseline`` saves its model, and the model that ``predict``,
``evaluate`` and ``run`` use unless told otherwise."""

DEFAULT_TRANSFORMER_DIR = Path("models") / "transformer"
"""Where ``train transformer`` saves its model."""


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
class BaselineConfig:
    """Hyper-parameters of the TF-IDF + logistic-regression baseline.

    ``c`` (inverse regularisation strength) was chosen on the validation split; the
    validation F1 is flat for values between roughly 16 and 256.
    """

    ngram_max: int = 2
    min_df: int = 3
    max_df: float = 0.9
    c: float = 32.0
    max_iter: int = 1000
    seed: int = 42


@dataclass(frozen=True)
class TransformerConfig:
    """Fine-tuning settings. The defaults target a single 16 GB GPU (e.g. a Colab T4)."""

    model_name: str = "xlm-roberta-base"
    """A Hugging Face Hub id or a local directory."""
    max_length: int = 512
    """Tokens per input, including the special tokens."""
    head_tokens: int = 128
    """Tokens kept from the start of a long text; the rest of the budget goes to its end."""
    epochs: float = 4.0
    batch_size: int = 8
    grad_accum_steps: int = 2
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    """Stop after this many evaluations (epochs) without a better validation F1."""
    seed: int = 42
    fp16: bool | None = None
    """Mixed precision; ``None`` enables it whenever a CUDA GPU is used."""
    max_train_samples: int | None = None
    """Train on a random subset (for quick experiments); validation/test stay complete."""
    cpu: bool = False
    """Train on the CPU even if a GPU is available."""
