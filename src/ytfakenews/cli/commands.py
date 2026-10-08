"""What each command does with its parsed arguments.

Every handler returns the exit code. Errors the user can fix raise
:class:`~ytfakenews.errors.YTFakeNewsError`, which :func:`ytfakenews.cli.main` prints as
one line; options that contradict each other are reported like any other argument
error, with the usage and exit code 2. Modules that import pandas, scikit-learn or an
optional extra are imported inside the handlers, so the CLI starts quickly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from ytfakenews.cli.output import (
    describe_transcript,
    format_confusion_matrix,
    format_metrics_table,
    format_prediction,
    format_training_metrics,
    model_info,
    print_json,
    transcript_info,
)
from ytfakenews.config import DEFAULT_DATA_PATH, BaselineConfig, SplitConfig, TransformerConfig
from ytfakenews.errors import ModelLoadError, YTFakeNewsError

if TYPE_CHECKING:
    from ytfakenews.asr import Transcript, TranscriptFiles
    from ytfakenews.models import Classifier
    from ytfakenews.predict import Prediction

__all__ = [
    "baseline_config",
    "evaluate",
    "predict",
    "run",
    "split_config",
    "train_baseline",
    "train_transformer",
    "transcribe",
    "transformer_config",
]

logger = logging.getLogger(__name__)


def _usage_error(args: argparse.Namespace, message: str) -> NoReturn:
    """Exit with the command's usage and ``message``, as argparse does (code 2)."""
    error: Callable[[str], NoReturn] = args.usage_error
    error(message)


# --------------------------------------------------------------------------- training


def split_config(args: argparse.Namespace) -> SplitConfig:
    """The split chosen with ``--val-size``, ``--test-size`` and ``--seed``."""
    if args.val_size + args.test_size >= 1:
        _usage_error(
            args,
            f"--val-size ({args.val_size}) and --test-size ({args.test_size}) must add up "
            "to less than 1",
        )
    return SplitConfig(val_size=args.val_size, test_size=args.test_size, seed=args.seed)


def baseline_config(args: argparse.Namespace) -> BaselineConfig:
    """The baseline hyper-parameters chosen on the command line."""
    return BaselineConfig(
        ngram_max=args.ngram_max, min_df=args.min_df, max_df=args.max_df, c=args.c, seed=args.seed
    )


def transformer_config(args: argparse.Namespace) -> TransformerConfig:
    """The fine-tuning settings chosen on the command line."""
    return TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        head_tokens=args.head_tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        fp16=args.fp16,
        max_train_samples=args.max_train_samples,
        cpu=args.cpu,
    )


def train_baseline(args: argparse.Namespace) -> int:
    """``ytfakenews train baseline``"""
    from ytfakenews.models import baseline

    metrics = baseline.train_baseline(
        args.data, args.output_dir, config=baseline_config(args), split=split_config(args)
    )
    print(
        f"Saved baseline model to {args.output_dir} "
        f"({metrics['n_features']:,} features, fitted in {metrics['fit_seconds']:.1f} s)\n"
    )
    print(format_training_metrics(metrics))
    return 0


def train_transformer(args: argparse.Namespace) -> int:
    """``ytfakenews train transformer``"""
    from ytfakenews.models import transformer

    metrics = transformer.train_transformer(
        args.data, args.output_dir, config=transformer_config(args), split=split_config(args)
    )
    print(
        f"Saved transformer model to {args.output_dir} (fine-tuned {args.model_name} for "
        f"{metrics['epochs_run']:g} epochs in {metrics['train_seconds']:.0f} s)\n"
    )
    print(format_training_metrics(metrics))
    return 0


# ------------------------------------------------------------------------- evaluation


def evaluate(args: argparse.Namespace) -> int:
    """``ytfakenews evaluate``"""
    from ytfakenews.artifacts import read_manifest, write_json
    from ytfakenews.data import prepare_splits
    from ytfakenews.evaluation import evaluate_classifier
    from ytfakenews.models import load_classifier

    _check_chunking(args)
    # The model first: a broken model directory or a missing extra fails before the
    # dataset is read.
    classifier = load_classifier(args.model, device=args.device)
    manifest = read_manifest(args.model)
    data_path = args.data or Path(manifest.data.get("path", DEFAULT_DATA_PATH))
    try:
        split = SplitConfig(**manifest.data.get("split", {}))
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(
            f"the manifest in {args.model} records an invalid split: {exc}"
        ) from exc
    splits, provenance = prepare_splits(data_path, split)
    if manifest.data.get("sha256") not in (None, provenance["sha256"]):
        logger.warning("%s differs from the dataset this model was trained on", data_path)
    frame = splits.get(args.split)
    metrics = evaluate_classifier(
        classifier,
        frame["text"].tolist(),
        frame["label"].to_numpy(),
        chunked=args.chunked,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        threshold=args.threshold,
        max_words=args.max_words,
    )
    metrics = {"model": model_info(args.model, classifier.backend), "split": args.split, **metrics}
    if args.output:
        write_json(args.output, metrics)
    if args.json:
        print_json(metrics)
        return 0
    mode = metrics["input"]["mode"]
    if mode == "chunked":
        mode += f", {args.chunk_words}-word chunks, overlap {args.overlap}"
    if args.max_words:
        mode += f", first {args.max_words} words"
    print(f"Model {args.model} ({classifier.backend}) on the {args.split} split ({mode})\n")
    print(format_metrics_table([(args.split, metrics)]))
    print()
    print(format_confusion_matrix(metrics, title="Confusion matrix"))
    return 0


# ------------------------------------------------------------------------- prediction


def predict(args: argparse.Namespace) -> int:
    """``ytfakenews predict``"""
    from ytfakenews.models import load_classifier

    _check_chunking(args)
    text = _read_input_text(args)
    classifier = load_classifier(args.model, device=args.device)
    prediction = _classify(text, classifier, args)
    if args.json:
        print_json(
            {
                "model": model_info(args.model, classifier.backend),
                "prediction": prediction.to_dict(),
            }
        )
    else:
        print(format_prediction(prediction, args.model, classifier.backend))
    return 0


def transcribe(args: argparse.Namespace) -> int:
    """``ytfakenews transcribe``"""
    _check_transcription_options(args)
    transcript, files = _transcribe(args)
    print(describe_transcript(transcript))
    for path in (files.txt, files.srt, files.json):
        print(f"  {path.as_posix()}")
    return 0


def run(args: argparse.Namespace) -> int:
    """``ytfakenews run``: ``transcribe``, then ``predict``."""
    from ytfakenews.models import load_classifier

    _check_chunking(args)
    _check_transcription_options(args)
    classifier = load_classifier(args.model, device=args.device)  # before any download
    transcript, files = _transcribe(args)
    if not transcript.text.strip():
        raise YTFakeNewsError(
            f"the transcript of {args.source} is empty (no speech found); "
            f"see {files.json.as_posix()}"
        )
    prediction = _classify(transcript.text, classifier, args)
    if args.json:
        print_json(
            {
                "source": args.source,
                "video": asdict(transcript.video) if transcript.video else None,
                "transcript": transcript_info(transcript, files),
                "model": model_info(args.model, classifier.backend),
                "prediction": prediction.to_dict(),
            }
        )
        return 0
    if transcript.video:
        print(f"Video:      {transcript.video.title or transcript.video.id}")
        if transcript.video.url:
            print(f"            {transcript.video.url}")
    print(f"{describe_transcript(transcript)} -> {files.txt.as_posix()}\n")
    print(format_prediction(prediction, args.model, classifier.backend))
    return 0


# -------------------------------------------------------------------------- helpers


def _check_chunking(args: argparse.Namespace) -> None:
    if args.overlap >= args.chunk_words:
        _usage_error(
            args,
            f"--overlap ({args.overlap}) must be smaller than --chunk-words ({args.chunk_words})",
        )


def _check_transcription_options(args: argparse.Namespace) -> None:
    if args.captions and args.translate:
        _usage_error(args, "--translate applies to Whisper; it cannot be combined with --captions")


def _read_input_text(args: argparse.Namespace) -> str:
    from ytfakenews.text import read_transcript

    if (args.file is None) == (args.text is None):
        _usage_error(args, "give exactly one of FILE or --text")
    if args.text is not None:
        return str(args.text)
    if str(args.file) == "-":
        return sys.stdin.read()
    if not args.file.is_file():
        raise YTFakeNewsError(f"file not found: {args.file}")
    try:
        return read_transcript(args.file)
    except UnicodeDecodeError as exc:
        raise YTFakeNewsError(f"{args.file} is not UTF-8 text: {exc}") from exc


def _transcribe(args: argparse.Namespace) -> tuple[Transcript, TranscriptFiles]:
    from ytfakenews.asr import transcribe_source

    return transcribe_source(
        args.source,
        output_dir=args.output_dir,
        captions=args.captions,
        language=args.language,
        translate=args.translate,
        model_size=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        keep_audio=args.keep_audio,
    )


def _classify(text: str, classifier: Classifier, args: argparse.Namespace) -> Prediction:
    from ytfakenews.predict import classify_text

    return classify_text(
        text,
        classifier,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        threshold=args.threshold,
    )
