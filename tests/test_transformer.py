"""Transformer backend: pure helpers always run; the smoke test needs the extra.

The smoke test builds a tiny, randomly initialised BERT and a word-level tokenizer
trained on the fly, so it runs fully offline in seconds on a CPU. It checks that the
training, saving, loading and prediction plumbing works, not model quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import import_or_skip, make_news_frame, write_zipped_csv
from ytfakenews.cli import main
from ytfakenews.transformer import truncate_head_tail


def test_truncate_head_tail() -> None:
    ids = list(range(10))
    assert truncate_head_tail(ids, 3, 2) == [0, 1, 2, 8, 9]
    assert truncate_head_tail(ids, 4, 0) == [0, 1, 2, 3]
    assert truncate_head_tail(ids, 0, 3) == [7, 8, 9]
    assert truncate_head_tail(ids, 6, 4) == ids
    assert truncate_head_tail(ids[:3], 2, 2) == [0, 1, 2]


# ------------------------------------------------------------------ with the extra


def build_tiny_base_model(directory: Path) -> Path:
    """Save a word-level tokenizer and a 2-layer BERT with random weights.

    The model gets a three-way head, like an NLI checkpoint, so the tests also cover
    replacing it with the two-way REAL/FAKE head.
    """
    import_or_skip("torch")
    transformers = import_or_skip("transformers")
    tokenizers = import_or_skip("tokenizers")

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    backend.normalizer = tokenizers.normalizers.Lowercase()
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.train_from_iterator(
        [*make_news_frame()["text"], "a short probe text"],
        tokenizers.trainers.WordLevelTrainer(special_tokens=specials),
    )
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[(token, backend.token_to_id(token)) for token in ("[CLS]", "[SEP]")],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )
    config = transformers.BertConfig(
        vocab_size=backend.get_vocab_size(),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
    )
    tokenizer.save_pretrained(str(directory))
    transformers.BertForSequenceClassification(config).save_pretrained(str(directory))
    return directory


@pytest.fixture(scope="session")
def tiny_base_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_base_model(tmp_path_factory.mktemp("tiny-bert"))


@pytest.fixture(scope="session")
def trained_transformer(
    tmp_path_factory: pytest.TempPathFactory, tiny_base_model: Path
) -> tuple[Path, dict[str, Any], Path]:
    from ytfakenews.transformer import TransformerConfig, train_transformer

    root = tmp_path_factory.mktemp("transformer")
    data = write_zipped_csv(make_news_frame(), root)
    config = TransformerConfig(
        model_name=str(tiny_base_model),
        max_length=64,
        head_tokens=16,
        epochs=1,
        batch_size=8,
        grad_accum_steps=1,
        eval_batch_size=8,
        cpu=True,
    )
    metrics = train_transformer(data, root / "model", config=config)
    return root / "model", metrics, data


def test_head_tail_encoder(tiny_base_model: Path) -> None:
    transformers = import_or_skip("transformers")
    from ytfakenews.transformer import HeadTailEncoder, special_affixes

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_base_model))
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    assert special_affixes(tokenizer) == ([cls_id], [sep_id])

    text = " ".join(make_news_frame()["text"].tolist()[:3])
    plain = tokenizer(text, add_special_tokens=False)["input_ids"]
    encoder = HeadTailEncoder(tokenizer, max_length=16, head_tokens=4)
    (ids,) = encoder.encode([text])
    assert len(plain) > 16
    assert ids == [cls_id, *plain[:4], *plain[-10:], sep_id]
    (short,) = encoder.encode(["officials said"])
    assert len(short) == 4

    with pytest.raises(ValueError, match="head_tokens must be between 0 and 14"):
        HeadTailEncoder(tokenizer, max_length=16, head_tokens=15)


def test_training_writes_a_loadable_model(
    trained_transformer: tuple[Path, dict[str, Any], Path],
) -> None:
    from ytfakenews.artifacts import read_manifest

    model_dir, metrics, _ = trained_transformer
    names = {path.name for path in model_dir.iterdir()}
    assert {"config.json", "tokenizer.json", "metrics.json", "manifest.json"} <= names
    assert any(name.endswith(".safetensors") for name in names)
    assert "checkpoints" not in names
    assert metrics["validation"]["n"] == metrics["test"]["n"] == 6
    assert metrics["epochs_run"] == 1
    assert json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["backend"] == (
        "transformer"
    )
    manifest = read_manifest(model_dir)
    assert manifest.backend == "transformer"
    assert (manifest.config["max_length"], manifest.config["head_tokens"]) == (64, 16)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["id2label"] == {"0": "REAL", "1": "FAKE"}


def test_loaded_model_predicts_through_the_common_interface(
    trained_transformer: tuple[Path, dict[str, Any], Path],
) -> None:
    from ytfakenews.predict import classify_text, load_classifier
    from ytfakenews.transformer import TransformerClassifier

    classifier = load_classifier(trained_transformer[0], device="cpu")
    assert isinstance(classifier, TransformerClassifier)

    texts = make_news_frame()["text"].tolist()[:5]
    batch = classifier.predict_proba(texts)
    assert batch.shape == (5,)
    assert np.all((batch >= 0) & (batch <= 1))
    one_by_one = np.array([classifier.predict_proba([text])[0] for text in texts])
    np.testing.assert_allclose(batch, one_by_one, atol=1e-5)
    assert classifier.predict_proba([]).shape == (0,)

    prediction = classify_text(" ".join(texts * 3), classifier, chunk_words=40, overlap=10)
    assert prediction.n_chunks > 1
    assert prediction.label in {"REAL", "FAKE"}


def test_cli_with_a_transformer_model(
    capsys: pytest.CaptureFixture[str],
    trained_transformer: tuple[Path, dict[str, Any], Path],
) -> None:
    model_dir, _, data = trained_transformer
    code = main(
        ["predict", "--model", str(model_dir), "--device", "cpu", "--text", "x y", "--json"]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["model"]["backend"] == "transformer"

    code = main(
        ["evaluate", "--model", str(model_dir), "--data", str(data), "--device", "cpu", "-q"]
    )
    assert code == 0
    assert "(transformer) on the test split" in capsys.readouterr().out


def test_cli_train_transformer(
    capsys: pytest.CaptureFixture[str], tiny_base_model: Path, news_zip: Path, tmp_path: Path
) -> None:
    out = tmp_path / "model"
    args = ["--model-name", str(tiny_base_model), "--max-length", "32", "--head-tokens", "8"]
    args += ["--epochs", "1", "--cpu", "--max-train-samples", "16", "--output-dir", str(out)]
    code = main(["train", "transformer", "--data", str(news_zip), *args])
    assert code == 0
    assert f"Saved transformer model to {out}" in capsys.readouterr().out
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["train_samples"] == 16

    code = main(["train", "transformer", "--data", str(news_zip), *args, "--head-tokens", "40"])
    assert code == 1
    assert "head_tokens must be between 0 and 30" in capsys.readouterr().err


def test_unknown_base_model_is_reported(
    capsys: pytest.CaptureFixture[str], news_zip: Path, tmp_path: Path
) -> None:
    import_or_skip("torch")
    import_or_skip("transformers")
    missing = tmp_path / "no-such-model"
    missing.mkdir()
    code = main(["train", "transformer", "--data", str(news_zip), "--model-name", str(missing)])
    assert code == 1
    assert f"could not load {str(missing)!r}" in capsys.readouterr().err
