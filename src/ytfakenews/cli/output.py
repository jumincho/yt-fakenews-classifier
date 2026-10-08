"""What the commands print: metric tables, verdicts, transcript summaries and JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ytfakenews.asr import Transcript, TranscriptFiles
    from ytfakenews.predict import Prediction

__all__ = [
    "describe_transcript",
    "format_confusion_matrix",
    "format_metrics_table",
    "format_prediction",
    "format_training_metrics",
    "model_info",
    "print_json",
    "transcript_info",
]


def print_json(payload: Any) -> None:
    """Print ``payload`` as indented JSON; non-ASCII text (transcripts!) stays readable."""
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def model_info(model_dir: Path, backend: str) -> dict[str, str]:
    """The ``model`` entry of the JSON output."""
    return {"path": model_dir.as_posix(), "backend": backend}


def transcript_info(transcript: Transcript, files: TranscriptFiles) -> dict[str, Any]:
    """The ``transcript`` entry of ``run --json``."""
    return {
        "source": transcript.source,
        "language": transcript.language,
        "details": transcript.details,
        "n_segments": len(transcript.segments),
        "files": {kind: path.as_posix() for kind, path in asdict(files).items()},
    }


def describe_transcript(transcript: Transcript) -> str:
    """One line: segments, words, language and where the text came from."""
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


def format_metrics_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """One row of metrics per ``(split name, metrics)`` pair."""
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


def format_confusion_matrix(metrics: dict[str, Any], *, title: str) -> str:
    """The 2x2 confusion matrix of :func:`ytfakenews.evaluation.compute_metrics`."""
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["values"]
    return "\n".join(
        [
            f"{title} (rows: true label, columns: predicted)",
            f"{'':>10}{'REAL':>7}{'FAKE':>7}",
            f"{'REAL':>10}{tn:>7}{fp:>7}",
            f"{'FAKE':>10}{fn:>7}{tp:>7}",
        ]
    )


def format_training_metrics(metrics: dict[str, Any]) -> str:
    """The validation and test rows and the test confusion matrix of a training run."""
    table = format_metrics_table([("validation", metrics["validation"]), ("test", metrics["test"])])
    matrix = format_confusion_matrix(metrics["test"], title="Test confusion matrix")
    return f"{table}\n\n{matrix}"


def format_prediction(prediction: Prediction, model_dir: Path, backend: str) -> str:
    """The verdict, the model and, for more than one chunk, a table of chunk scores."""
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
