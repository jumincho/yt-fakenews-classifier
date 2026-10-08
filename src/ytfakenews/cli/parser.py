"""The commands of ``ytfakenews`` and their options.

The training options default to the fields of the dataclasses in
:mod:`ytfakenews.config`, so the CLI and the Python API cannot drift apart. Each
subcommand stores its handler from :mod:`ytfakenews.cli.commands` as ``args.handler``
and its ``error`` method as ``args.usage_error``, which handlers call when two options
contradict each other.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from ytfakenews import __version__
from ytfakenews.asr import DEFAULT_TRANSCRIPT_DIR, DEFAULT_WHISPER_MODEL
from ytfakenews.cli import commands
from ytfakenews.config import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_TRANSFORMER_DIR,
    BaselineConfig,
    SplitConfig,
    TransformerConfig,
)
from ytfakenews.predict import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP, DEFAULT_THRESHOLD

__all__ = ["build_parser"]

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

_SPLIT = SplitConfig()
_BASELINE = BaselineConfig()
_TRANSFORMER = TransformerConfig()


# ----------------------------------------------------------------------- argument types


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


# ----------------------------------------------------------------------- option groups


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
        "--seed",
        type=int,
        default=_SPLIT.seed,
        help="seed for the split and the model (default: %(default)s)",
    )
    group.add_argument(
        "--val-size",
        type=_fraction,
        default=_SPLIT.val_size,
        metavar="F",
        help="validation fraction (default: %(default)s)",
    )
    group.add_argument(
        "--test-size",
        type=_fraction,
        default=_SPLIT.test_size,
        metavar="F",
        help="test fraction (default: %(default)s)",
    )


def _add_classification_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("classification")
    group.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
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
        default=DEFAULT_TRANSCRIPT_DIR,
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


# ---------------------------------------------------------------------------- commands


def _add_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    train = subparsers.add_parser(
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
        default=DEFAULT_BASELINE_DIR,
        metavar="DIR",
        help="where to save the model (default: %(default)s)",
    )
    model = baseline.add_argument_group("model")
    model.add_argument(
        "--C",
        dest="c",
        type=_positive_float,
        default=_BASELINE.c,
        help="inverse regularisation strength (default: %(default)s)",
    )
    model.add_argument(
        "--min-df",
        type=_int_at_least(1),
        default=_BASELINE.min_df,
        metavar="N",
        help="ignore n-grams in fewer than N training articles (default: %(default)s)",
    )
    model.add_argument(
        "--max-df",
        type=_float_in(0.0, 1.0, inclusive=True),
        default=_BASELINE.max_df,
        metavar="F",
        help="ignore n-grams in more than this fraction of articles (default: %(default)s)",
    )
    model.add_argument(
        "--ngram-max",
        type=_int_at_least(1),
        default=_BASELINE.ngram_max,
        metavar="N",
        help="longest word n-gram (default: %(default)s)",
    )
    baseline.set_defaults(handler=commands.train_baseline, usage_error=baseline.error)

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
        default=DEFAULT_TRANSFORMER_DIR,
        metavar="DIR",
        help="where to save the model (default: %(default)s)",
    )
    fine_tuning = transformer.add_argument_group("fine-tuning")
    fine_tuning.add_argument(
        "--model-name",
        default=_TRANSFORMER.model_name,
        metavar="NAME",
        help="Hugging Face model id or local directory (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--epochs",
        type=_positive_float,
        default=_TRANSFORMER.epochs,
        metavar="N",
        help="maximum epochs (default: %(default)g)",
    )
    fine_tuning.add_argument(
        "--patience",
        type=_int_at_least(1),
        default=_TRANSFORMER.patience,
        metavar="N",
        help="stop after N epochs without a better validation F1 (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--batch-size",
        type=_int_at_least(1),
        default=_TRANSFORMER.batch_size,
        metavar="N",
        help="training batch size per device (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--grad-accum",
        type=_int_at_least(1),
        default=_TRANSFORMER.grad_accum_steps,
        metavar="N",
        help="gradient accumulation steps (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--lr",
        type=_positive_float,
        default=_TRANSFORMER.learning_rate,
        help="learning rate (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--max-length",
        type=_int_at_least(8),
        default=_TRANSFORMER.max_length,
        metavar="N",
        help="tokens per input, including special tokens (default: %(default)s)",
    )
    fine_tuning.add_argument(
        "--head-tokens",
        type=_int_at_least(0),
        default=_TRANSFORMER.head_tokens,
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
        default=_TRANSFORMER.fp16,
        help="mixed precision (default: on when a CUDA GPU is used)",
    )
    fine_tuning.add_argument(
        "--cpu", action="store_true", help="train on the CPU even if a GPU exists"
    )
    transformer.set_defaults(handler=commands.train_transformer, usage_error=transformer.error)


def _add_evaluate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    evaluate = subparsers.add_parser(
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
    evaluate.set_defaults(handler=commands.evaluate, usage_error=evaluate.error)


def _add_predict_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    predict = subparsers.add_parser(
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
    predict.set_defaults(handler=commands.predict, usage_error=predict.error)


def _add_transcribe_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    transcribe = subparsers.add_parser(
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
    transcribe.set_defaults(handler=commands.transcribe, usage_error=transcribe.error)


def _add_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    run = subparsers.add_parser(
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
    run.set_defaults(handler=commands.run, usage_error=run.error)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``ytfakenews`` command."""
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
    subparsers = parser.add_subparsers(
        title="commands", dest="command", metavar="COMMAND", required=True
    )
    _add_transcribe_parser(subparsers, common)
    _add_train_parser(subparsers, common)
    _add_evaluate_parser(subparsers, common)
    _add_predict_parser(subparsers, common)
    _add_run_parser(subparsers, common)
    return parser
