"""Command-line interface: ``ytfakenews COMMAND [OPTIONS]``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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
    group.add_argument(
        "--device",
        default="auto",
        help="transformer backend only: auto, cpu, cuda, cuda:1, mps (default: auto)",
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
    predict.add_argument("--json", action="store_true", help="print the result as JSON")
    predict.set_defaults(handler=_cmd_predict, usage_error=predict.error)


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
    _add_train_parser(commands, common)
    _add_evaluate_parser(commands, common)
    _add_predict_parser(commands, common)
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
    return read_transcript(args.file)


def _cmd_predict(args: argparse.Namespace) -> int:
    from ytfakenews.predict import load_classifier

    _check_chunking(args)
    text = _read_input_text(args)
    classifier = load_classifier(args.model, device=args.device)
    prediction = _classify(text, classifier, args)
    if args.json:
        payload = {"model": _model_info(args.model, classifier.backend), **prediction.to_dict()}
        print(json.dumps(payload, indent=2))
    else:
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
