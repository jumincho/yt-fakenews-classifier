"""Transcribe audio with faster-whisper.

faster-whisper runs Whisper on CTranslate2 and decodes audio with PyAV, so neither
PyTorch nor a system ffmpeg is needed. Models are downloaded from the Hugging Face Hub
on first use, inside :func:`_load_whisper_model`, which tests replace. Requires the
``asr`` extra.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ytfakenews._optional import require
from ytfakenews.asr.transcript import Transcript
from ytfakenews.errors import TranscriptionError
from ytfakenews.text import Segment

__all__ = ["DEFAULT_WHISPER_MODEL", "transcribe"]

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "small"


def _load_whisper_model(model_size: str, *, device: str, compute_type: str) -> Any:
    """Load a faster-whisper model (downloaded from the Hugging Face Hub on first use)."""
    faster_whisper = require("faster_whisper", extra="asr")
    logger.info("Loading faster-whisper model %r (device=%s)", model_size, device)
    try:
        return faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # download, CUDA and model errors come in many types
        raise TranscriptionError(
            f"could not load the faster-whisper model {model_size!r} (models are "
            f"downloaded from the Hugging Face Hub on first use): {exc}"
        ) from exc


def transcribe(
    audio: str | Path,
    *,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
    translate: bool = False,
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Transcript:
    """Transcribe an audio or video file with faster-whisper.

    ``language`` is the spoken language (ISO 639-1, e.g. ``"en"``); ``None`` lets
    Whisper detect it. With ``translate=True`` Whisper translates the speech to
    English. The Silero voice-activity filter is on by default, which skips silence
    and music and reduces hallucinated text.
    """
    audio = Path(audio)
    model = _load_whisper_model(model_size, device=device, compute_type=compute_type)
    task = "translate" if translate else "transcribe"
    logger.info("Transcribing %s (%s)", audio.name, task)
    segments = []
    try:
        raw_segments, info = model.transcribe(
            str(audio), language=language, task=task, beam_size=beam_size, vad_filter=vad_filter
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        next_report = 0.25
        for raw in raw_segments:  # lazy: decoding happens while iterating
            text = raw.text.strip()
            if text:
                segments.append(Segment(float(raw.start), float(raw.end), text))
            while duration and next_report < 1 and raw.end >= next_report * duration:
                logger.info("  %d%% of %.0f s transcribed", round(next_report * 100), duration)
                next_report += 0.25
    except (OSError, RuntimeError, ValueError) as exc:  # undecodable media, CUDA errors, ...
        raise TranscriptionError(f"could not transcribe {audio}: {exc}") from exc
    spoken = getattr(info, "language", None)
    return Transcript(
        segments=tuple(segments),
        source="whisper",
        language="en" if translate else spoken,
        details={
            "model": model_size,
            "task": task,
            "spoken_language": spoken,
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            "duration": duration,
            "vad_filter": vad_filter,
        },
    )
