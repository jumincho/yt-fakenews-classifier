"""Command-line interface: ``ytfakenews COMMAND [OPTIONS]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ytfakenews import __version__
from ytfakenews.data import DEFAULT_DATA_PATH
from ytfakenews.errors import YTFakeNewsError
from ytfakenews.predict import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_MODEL_DIR,
    DEFAULT_OVERLAP,
    DEFAULT_THRESHOLD,
    Classifier,
    Prediction,
)
from ytfakenews.transcribe import DEFAULT_OUTPUT_DIR, DEFAULT_WHISPER_MODEL

if TYPE_CHECKING:
    from ytfakenews.transcribe import Transcript, TranscriptFiles

__all__ = ["build_parser", "main"]

logger = logging.getLogger("ytfakenews")

DESCRIPTION = """\
Classify YouTube videos as REAL or FAKE news from what is said in them.

The pipeline downloads the audio, transcribes it with faster-whisper (or uses the
video's own captions), splits the transcript into overlapping chunks and averages a
text classifier's P(fake) over the chunks. The classifier is trained on 2016-era
English news articles: it picks up style and source cues, it does not check facts.
"""

EPILOG = """\
examples:
  ytfakenews train baseline
  ytfakenews train transformer --epochs 4
  ytfakenews run "https://www.youtube.com/watch?v=VIDEO_ID"
  ytfakenews transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --captions
  ytfakenews predict examples/transcript_local_news.txt
  ytfakenews predict --text "Officials confirmed the figures on Tuesday." --json

Run `ytfakenews COMMAND --help` for the options of a command.
"""


# --------------------------------------------------------------------- argument types


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {number}")
        return number

    return parse


def _float_in(low: float, high: float, *, inclusive: bool) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
        ok = low <= number <= high if inclusive else low < number < high
        if not ok:
            bounds = f"[{low}, {high}]" if inclusive else f"({low}, {high})"
            raise argparse.ArgumentTypeError(f"must be in {bounds}, got {number}")
        return number

    return parse


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


_fraction = _float_in(0.0, 1.0, inclusive=False)
_probability = _float_in(0.0, 1.0, inclusive=True)


# --------------------------------------------------------------------------- parsers


def _add_data_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        metavar="PATH",
        help="CSV or single-CSV ZIP with title, text and label columns (default: %(default)s)",
    )
    group.add_argument(
        "--seed", type=int, default=42, help="seed for the split and the model (default: 42)"
    )
    group.add_argument(
        "--val-size", type=_fraction, default=0.1, help="validation fraction (default: 0.1)"
    )
    group.add_argument(
        "--test-size", type=_fraction, default=0.1, help="test fraction (default: 0.1)"
    )


def _add_classification_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("classification")
    group.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        metavar="DIR",
        help="trained model directory (default: %(default)s)",
    )
    group.add_argument(
        "--chunk-words",
        type=_int_at_least(1),
        default=DEFAULT_CHUNK_WORDS,
        metavar="N",
        help="words per chunk (default: %(default)s)",
    )
    group.add_argument(
        "--overlap",
        type=_int_at_least(0),
        default=DEFAULT_OVERLAP,
        metavar="N",
        help="minimum words shared by neighbouring chunks (default: %(default)s)",
    )
    group.add_argument(
        "--threshold",
        type=_probability,
        default=DEFAULT_THRESHOLD,
        metavar="P",
        help="label FAKE when the mean P(fake) is at least P (default: %(default)s)",
    )


def _add_device_option(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--device", default="auto", help=f"{help_text} (default: auto)")


_TRANSFORMER_DEVICE_HELP = "device for the transformer backend: auto, cpu, cuda, cuda:1, mps"


def _add_transcription_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("transcription")
    group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="where to write <id>.txt, .srt and .json (default: %(default)s)",
    )
    group.add_argument(
        "--captions",
        action="store_true",
        help="use the video's own subtitles or automatic captions instead of Whisper",
    )
    group.add_argument(
        "--language",
        metavar="CODE",
        help="spoken language for Whisper, e.g. en or ko (default: detect); with "
        "--captions the caption language (default: en)",
    )
    group.add_argument(
        "--translate",
        action="store_true",
        help="have Whisper translate the speech into English",
    )
    group.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        metavar="NAME",
        help="faster-whisper model: tiny, base, small, medium, large-v3, turbo, ... "
        "(default: %(default)s)",
    )
    group.add_argument(
        "--compute-type",
        default="auto",
        metavar="TYPE",
        help="CTranslate2 compute type, e.g. int8 or float16 (default: auto)",
    )
    group.add_argument(
        "--keep-audio", action="store_true", help="keep the downloaded audio next to the transcript"
    )


def _add_train_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    train = commands.add_parser(
        "train",
        help="train a classifier",
        description="Train a classifier on the news dataset and save it with its metrics.",
    )
    backends = train.add_subparsers(
        title="backends", dest="backend", metavar="BACKEND", required=True
    )

    baseline = backends.add_parser(
        "baseline",
        parents=[common],
        help="TF-IDF + logistic regression (CPU, seconds)",
        description=(
            "Train the TF-IDF (word 1-2-grams, sublinear tf) + logistic-regression "
            "baseline, evaluate it on the validation and test splits and save "
            "model.joblib, metrics.json and manifest.json."
        ),
    )
    _add_data_options(baseline)
    baseline.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        metavar="DIR",
        help="where to save the model (default: %(default)s)",
    )
    model = baseline.add_argument_group("model")
    model.add_argument(
        "--C",
        dest="c",
        type=_positive_float,
        default=32.0,
        help="inverse regularisation strength (default: %(default)s)",
    )
    model.add_argument(
        "--min-df",
        type=_int_at_least(1),
        default=3,
        metavar="N",
        help="ignore n-grams in fewer than N training articles (default: %(default)s)",
    )
    model.add_argument(
        "--max-df",
        type=_float_in(0.0, 1.0, inclusive=True),
        default=0.9,
        metavar="F",
        help="ignore n-grams in more than this fraction of articles (default: %(default)s)",
    )
    model.add_argument(
        "--ngram-max",
        type=_int_at_least(1),
        default=2,
        metavar="N",
        help="longest word n-gram (default: %(default)s)",
    )
    baseline.set_defaults(handler=_cmd_train_baseline)

    transformer = backends.add_parser(
        "transformer",
        parents=[common],
        help="fine-tune a transformer encoder (GPU recommended)",
        description=(
            "Fine-tune a transformer encoder (default: multilingual XLM-RoBERTa base) with "
            "head+tail truncation and early stopping on validation F1, then evaluate it and "
            "save the model, tokenizer, metrics.json and manifest.json. The base model is "
            "downloaded from the Hugging Face Hub unless --model-name is a local directory. "
            "Needs the 'transformer' extra."
        ),
    )
    _add_data_options(transformer)
    transformer.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "transformer",
        metavar="DIR",
        help="where to save the model (default: %(default)s)",
    )
    fine_tuning = transformer.add_argument_group("fine-tuning")
    fine_tuning.add_argument(
        "--model-name",
        default="xlm-roberta-base",
        metavar="NAME",
        help="Hugging Face model id or local directory (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--epochs", type=_positive_float, default=4.0, help="maximum epochs (default: 4)"
    )
    fine_tuning.add_argument(
        "--patience",
        type=_int_at_least(1),
        default=2,
        metavar="N",
        help="stop after N epochs without a better validation F1 (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--batch-size", type=_int_at_least(1), default=8, metavar="N", help="(default: 8)"
    )
    fine_tuning.add_argument(
        "--grad-accum",
        type=_int_at_least(1),
        default=2,
        metavar="N",
        help="gradient accumulation steps (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--lr", type=_positive_float, default=2e-5, help="learning rate (default: %(default)s)"
    )
    fine_tuning.add_argument(
        "--max-length",
        type=_int_at_least(8),
        default=512,
        metavar="N",
        help="tokens per input, including special tokens (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--head-tokens",
        type=_int_at_least(0),
        default=128,
        metavar="N",
        help="tokens kept from the start of a long text; the rest come from its end "
        "(default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--max-train-samples",
        type=_int_at_least(1),
        metavar="N",
        help="train on a random subset of N articles, e.g. for a quick CPU run",
    )
    fine_tuning.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mixed precision (default: on when a CUDA GPU is used)",
    )
    fine_tuning.add_argument(
        "--cpu", action="store_true", help="train on the CPU even if a GPU exists"
    )
    transformer.set_defaults(handler=_cmd_train_transformer)


def _add_evaluate_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    evaluate = commands.add_parser(
        "evaluate",
        parents=[common],
        help="evaluate a trained model on a dataset split",
        description=(
            "Re-create the split a model was trained with (seed and fractions come from "
            "its manifest) and report metrics on one split. By default every article is "
            "scored as one document; --chunked scores it through the same clean -> chunk "
            "-> average path that transcripts take."
        ),
    )
    _add_classification_options(evaluate)
    _add_device_option(evaluate, help_text=_TRANSFORMER_DEVICE_HELP)
    evaluate.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="PATH",
        help="dataset to use (default: the path recorded in the model's manifest)",
    )
    evaluate.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="split to evaluate (default: %(default)s)",
    )
    evaluate.add_argument(
        "--chunked", action="store_true", help="score chunks and average, as for transcripts"
    )
    evaluate.add_argument(
        "--max-words",
        type=_int_at_least(1),
        metavar="N",
        help="score only the first N words of every article, to mimic short transcripts",
    )
    evaluate.add_argument("--json", action="store_true", help="print the metrics as JSON")
    evaluate.add_argument(
        "--output", type=Path, metavar="FILE", help="also write the metrics as JSON to FILE"
    )
    evaluate.set_defaults(handler=_cmd_evaluate, usage_error=evaluate.error)


def _add_predict_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    predict = commands.add_parser(
        "predict",
        parents=[common],
        help="classify a transcript file or a text",
        description=(
            "Classify a transcript (.txt, .srt or .vtt; '-' reads standard input) or a "
            "text given with --text."
        ),
    )
    predict.add_argument("file", nargs="?", type=Path, metavar="FILE", help="transcript file")
    predict.add_argument("--text", help="classify this text instead of a file")
    _add_classification_options(predict)
    _add_device_option(predict, help_text=_TRANSFORMER_DEVICE_HELP)
    predict.add_argument("--json", action="store_true", help="print the result as JSON")
    predict.set_defaults(handler=_cmd_predict, usage_error=predict.error)


def _add_transcribe_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    transcribe = commands.add_parser(
        "transcribe",
        parents=[common],
        help="transcribe a video (or local audio file) to .txt/.srt/.json",
        description=(
            "Download the audio of a video with yt-dlp and transcribe it with "
            "faster-whisper (voice-activity filter on), or fetch the video's captions "
            "with --captions. A local audio or video file can be given instead of a URL. "
            "Needs the 'asr' extra."
        ),
    )
    transcribe.add_argument("source", metavar="URL_OR_FILE", help="video URL or media file")
    _add_transcription_options(transcribe)
    _add_device_option(transcribe, help_text="device for Whisper: auto, cpu or cuda")
    transcribe.set_defaults(handler=_cmd_transcribe, usage_error=transcribe.error)


def _add_run_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    run = commands.add_parser(
        "run",
        parents=[common],
        help="transcribe a video and classify it (end to end)",
        description=(
            "Transcribe a video (see `ytfakenews transcribe`) and classify the transcript "
            "(see `ytfakenews predict`). The model is loaded first, so a missing model "
            "fails before anything is downloaded."
        ),
    )
    run.add_argument("source", metavar="URL_OR_FILE", help="video URL or media file")
    _add_transcription_options(run)
    _add_classification_options(run)
    _add_device_option(run, help_text="device for Whisper and the transformer: auto, cpu, cuda")
    run.add_argument("--json", action="store_true", help="print the result as JSON")
    run.set_defaults(handler=_cmd_run, usage_error=run.error)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for documentation and tests)."""
    common = argparse.ArgumentParser(add_help=False)
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="only show warnings and errors"
    )

    parser = argparse.ArgumentParser(
        prog="ytfakenews",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(
        title="commands", dest="command", metavar="COMMAND", required=True
    )
    _add_transcribe_parser(commands, common)
    _add_train_parser(commands, common)
    _add_evaluate_parser(commands, common)
    _add_predict_parser(commands, common)
    _add_run_parser(commands, common)
    return parser


# -------------------------------------------------------------------------- commands


def _cmd_train_baseline(args: argparse.Namespace) -> int:
    from ytfakenews.baseline import BaselineConfig, train_baseline
    from ytfakenews.data import SplitConfig

    config = BaselineConfig(
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        c=args.c,
        seed=args.seed,
    )
    split = SplitConfig(val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    metrics = train_baseline(args.data, args.output_dir, config=config, split=split)
    print(
        f"Saved baseline model to {args.output_dir} "
        f"({metrics['n_features']:,} features, fitted in {metrics['fit_seconds']:.1f} s)\n"
    )
    print(_format_metrics_table([("validation", metrics["validation"]), ("test", metrics["test"])]))
    print()
    print(_format_confusion_matrix(metrics["test"], title="Test confusion matrix"))
    return 0


def _cmd_train_transformer(args: argparse.Namespace) -> int:
    from ytfakenews.data import SplitConfig
    from ytfakenews.transformer import TransformerConfig, train_transformer

    config = TransformerConfig(
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
    split = SplitConfig(val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    metrics = train_transformer(args.data, args.output_dir, config=config, split=split)
    print(
        f"Saved transformer model to {args.output_dir} (fine-tuned {args.model_name} for "
        f"{metrics['epochs_run']:g} epochs in {metrics['train_seconds']:.0f} s)\n"
    )
    print(_format_metrics_table([("validation", metrics["validation"]), ("test", metrics["test"])]))
    print()
    print(_format_confusion_matrix(metrics["test"], title="Test confusion matrix"))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from ytfakenews.artifacts import read_manifest
    from ytfakenews.data import SplitConfig, prepare_splits
    from ytfakenews.evaluation import evaluate_classifier
    from ytfakenews.predict import load_classifier

    _check_chunking(args)
    manifest = read_manifest(args.model)
    data_path = args.data or Path(manifest.data.get("path", DEFAULT_DATA_PATH))
    split = SplitConfig(**manifest.data.get("split", {}))
    splits, provenance = prepare_splits(data_path, split)
    if manifest.data.get("sha256") not in (None, provenance["sha256"]):
        logger.warning("%s differs from the dataset this model was trained on", data_path)
    frame = splits.get(args.split)
    classifier = load_classifier(args.model, device=args.device)
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
    metrics = {"model": _model_info(args.model, classifier.backend), "split": args.split, **metrics}
    if args.output:
        from ytfakenews.artifacts import write_json

        write_json(args.output, metrics)
    if args.json:
        print(json.dumps(metrics, indent=2))
        return 0
    mode = metrics["input"]["mode"]
    if mode == "chunked":
        mode += f", {args.chunk_words}-word chunks, overlap {args.overlap}"
    if args.max_words:
        mode += f", first {args.max_words} words"
    print(f"Model {args.model} ({classifier.backend}) on the {args.split} split ({mode})\n")
    print(_format_metrics_table([(args.split, metrics)]))
    print()
    print(_format_confusion_matrix(metrics, title="Confusion matrix"))
    return 0


def _check_chunking(args: argparse.Namespace) -> None:
    if args.overlap >= args.chunk_words:
        args.usage_error(
            f"--overlap ({args.overlap}) must be smaller than --chunk-words ({args.chunk_words})"
        )


def _read_input_text(args: argparse.Namespace) -> str:
    from ytfakenews.text import read_transcript

    if (args.file is None) == (args.text is None):
        args.usage_error("give exactly one of FILE or --text")
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


def _cmd_predict(args: argparse.Namespace) -> int:
    from ytfakenews.predict import load_classifier

    _check_chunking(args)
    text = _read_input_text(args)
    classifier = load_classifier(args.model, device=args.device)
    prediction = _classify(text, classifier, args)
    if args.json:
        payload = {
            "model": _model_info(args.model, classifier.backend),
            "prediction": prediction.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_prediction(prediction, args.model, classifier.backend))
    return 0


def _check_transcription_args(args: argparse.Namespace) -> None:
    if args.captions and args.translate:
        args.usage_error("--translate applies to Whisper; it cannot be combined with --captions")


def _transcribe(args: argparse.Namespace) -> tuple[Transcript, TranscriptFiles]:
    from ytfakenews.transcribe import transcribe_source

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


def _cmd_transcribe(args: argparse.Namespace) -> int:
    _check_transcription_args(args)
    transcript, files = _transcribe(args)
    print(_describe_transcript(transcript))
    for path in (files.txt, files.srt, files.json):
        print(f"  {path.as_posix()}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from ytfakenews.predict import load_classifier

    _check_chunking(args)
    _check_transcription_args(args)
    classifier = load_classifier(args.model, device=args.device)
    transcript, files = _transcribe(args)
    if not transcript.text.strip():
        raise YTFakeNewsError(
            f"the transcript of {args.source} is empty (no speech found); "
            f"see {files.json.as_posix()}"
        )
    prediction = _classify(transcript.text, classifier, args)
    if args.json:
        payload = {
            "source": args.source,
            "video": asdict(transcript.video) if transcript.video else None,
            "transcript": {
                "source": transcript.source,
                "language": transcript.language,
                "details": transcript.details,
                "n_segments": len(transcript.segments),
                "files": {kind: path.as_posix() for kind, path in asdict(files).items()},
            },
            "model": _model_info(args.model, classifier.backend),
            "prediction": prediction.to_dict(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if transcript.video:
        print(f"Video:      {transcript.video.title or transcript.video.id}")
        if transcript.video.url:
            print(f"            {transcript.video.url}")
    print(f"{_describe_transcript(transcript)} -> {files.txt.as_posix()}\n")
    print(_format_prediction(prediction, args.model, classifier.backend))
    return 0


def _classify(text: str, classifier: Classifier, args: argparse.Namespace) -> Prediction:
    from ytfakenews.predict import classify_text

    try:
        return classify_text(
            text,
            classifier,
            chunk_words=args.chunk_words,
            overlap=args.overlap,
            threshold=args.threshold,
        )
    except ValueError as exc:
        raise YTFakeNewsError(str(exc)) from exc


# ------------------------------------------------------------------------ formatting


def _model_info(model_dir: Path, backend: str) -> dict[str, str]:
    return {"path": model_dir.as_posix(), "backend": backend}


def _describe_transcript(transcript: Transcript) -> str:
    details = transcript.details
    if transcript.source == "whisper":
        origin = f"faster-whisper {details.get('model')}"
        if details.get("task") == "translate":
            origin += f", translated from {details.get('spoken_language')}"
    else:
        origin = f"{details.get('kind')} captions, track {details.get('track')}"
        if details.get("machine_translated"):
            origin += ", machine-translated by YouTube"
    words = len(transcript.text.split())
    return (
        f"Transcript: {len(transcript.segments)} segments, {words:,} words, "
        f"language {transcript.language or 'unknown'} ({origin})"
    )


def _format_metrics_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    lines = [
        f"{'split':<11}{'n':>6}{'accuracy':>10}{'precision':>11}"
        f"{'recall':>8}{'F1':>8}{'ROC-AUC':>9}"
    ]
    for name, m in rows:
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        lines.append(
            f"{name:<11}{m['n']:>6}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
            f"{m['recall']:>8.4f}{m['f1']:>8.4f}{auc:>9}"
        )
    return "\n".join(lines)


def _format_confusion_matrix(metrics: dict[str, Any], *, title: str) -> str:
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["values"]
    return "\n".join(
        [
            f"{title} (rows: true label, columns: predicted)",
            f"{'':>10}{'REAL':>7}{'FAKE':>7}",
            f"{'REAL':>10}{tn:>7}{fp:>7}",
            f"{'FAKE':>10}{fn:>7}{tp:>7}",
        ]
    )


def _format_prediction(prediction: Prediction, model_dir: Path, backend: str) -> str:
    lines = [
        f"{prediction.label}  P(fake) = {prediction.p_fake:.3f}  "
        f"(threshold {prediction.threshold:.2f}, mean over {prediction.n_chunks} "
        f"chunk{'s' if prediction.n_chunks != 1 else ''})",
        f"model: {model_dir.as_posix()} ({backend}); input: {prediction.n_words:,} words",
    ]
    if prediction.n_chunks > 1:
        lines += ["", f"{'chunk':>5}  {'words':<13}{'P(fake)':>7}  preview"]
        lines += [
            f"{chunk.index + 1:>5}  {f'{chunk.start_word}-{chunk.end_word}':<13}"
            f"{chunk.p_fake:>7.3f}  {chunk.preview}"
            for chunk in prediction.chunks
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ main


class _LogFormatter(logging.Formatter):
    """Plain messages for progress; ``warning: ...``/``error: ...`` prefixes otherwise."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


_log_handler: logging.Handler | None = None


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    # A fresh handler per invocation binds to the current sys.stderr.
    global _log_handler
    if _log_handler is not None:
        logger.removeHandler(_log_handler)
    _log_handler = logging.StreamHandler(sys.stderr)
    _log_handler.setFormatter(_LogFormatter("%(message)s"))
    logger.addHandler(_log_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``ytfakenews`` console script; returns the exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)
    try:
        return int(args.handler(args))
    except YTFakeNewsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
