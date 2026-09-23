"""Get a transcript for a video: download audio with yt-dlp and transcribe it with
faster-whisper, or fetch the video's own captions.

faster-whisper runs Whisper on CTranslate2 and decodes audio with PyAV, so neither
PyTorch nor a system ffmpeg is needed. Every network access and model load goes
through :func:`_extract_info` (yt-dlp) or :func:`_load_whisper_model`
(faster-whisper); tests replace these two functions.

Requires the ``asr`` extra.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ytfakenews._optional import require
from ytfakenews.artifacts import write_json
from ytfakenews.errors import TranscriptionError
from ytfakenews.text import Segment, parse_srt, parse_vtt, segments_to_srt, segments_to_text

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_WHISPER_MODEL",
    "DownloadedAudio",
    "Transcript",
    "TranscriptFiles",
    "VideoInfo",
    "download_audio",
    "fetch_captions",
    "save_transcript",
    "transcribe",
    "transcribe_source",
]

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class VideoInfo:
    """The few video metadata fields worth keeping next to a transcript."""

    id: str
    title: str | None = None
    url: str | None = None
    channel: str | None = None
    duration: float | None = None
    upload_date: str | None = None

    @classmethod
    def from_info_dict(cls, info: Mapping[str, Any]) -> VideoInfo:
        """Build from a yt-dlp info dict."""
        return cls(
            id=str(info.get("id") or "video"),
            title=info.get("title"),
            url=info.get("webpage_url") or info.get("original_url"),
            channel=info.get("channel") or info.get("uploader"),
            duration=info.get("duration"),
            upload_date=info.get("upload_date"),
        )


@dataclass(frozen=True)
class Transcript:
    """Timed transcript segments plus where they came from.

    ``source`` is ``"whisper"`` or ``"captions"``; ``language`` is the language of the
    text (``"en"`` after Whisper translation); ``details`` records the Whisper model and
    detected language, or the caption track and whether it is manual, automatic or
    machine-translated.
    """

    segments: tuple[Segment, ...]
    source: str
    language: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    video: VideoInfo | None = None

    @property
    def text(self) -> str:
        """Plain text, one segment per line."""
        return segments_to_text(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "language": self.language,
            "details": self.details,
            "video": asdict(self.video) if self.video else None,
            "segments": [asdict(segment) for segment in self.segments],
        }


@dataclass(frozen=True)
class DownloadedAudio:
    path: Path
    video: VideoInfo


@dataclass(frozen=True)
class TranscriptFiles:
    """Paths written by :func:`save_transcript`."""

    srt: Path
    txt: Path
    json: Path


# ---------------------------------------------------------------- external boundaries


class _YtDlpLogger:
    """Route yt-dlp's console output into :mod:`logging`.

    Errors are logged at debug level only: they are re-raised as
    :class:`TranscriptionError`, which the CLI prints once.
    """

    def debug(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        logger.warning("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)


def _extract_info(url: str, options: Mapping[str, Any], *, download: bool) -> dict[str, Any]:
    """Run yt-dlp on one URL; the only function in the package that touches YouTube."""
    yt_dlp = require("yt_dlp", extra="asr")
    params = {
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "noplaylist": True,
        "logger": _YtDlpLogger(),
        **options,
    }
    try:
        with yt_dlp.YoutubeDL(params) as ydl:
            info = ydl.extract_info(url, download=download)
            return dict(ydl.sanitize_info(info))
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc).removeprefix("ERROR: ")
        raise TranscriptionError(f"yt-dlp failed for {url}: {message}") from exc


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


# ------------------------------------------------------------------------ operations


def download_audio(url: str, output_dir: str | Path) -> DownloadedAudio:
    """Download the best audio-only stream of a video into ``output_dir``.

    No post-processing is requested, so yt-dlp does not need ffmpeg; the file keeps the
    container YouTube serves (usually ``.webm`` or ``.m4a``), which faster-whisper
    decodes directly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"format": "bestaudio/best", "outtmpl": str(output_dir / "%(id)s.%(ext)s")}
    logger.info("Downloading audio from %s", url)
    info = _extract_info(url, options, download=True)
    downloads = info.get("requested_downloads") or []
    paths = [Path(item["filepath"]) for item in downloads if item.get("filepath")]
    if not paths or not paths[0].is_file():
        raise TranscriptionError(f"yt-dlp reported no downloaded audio file for {url}")
    return DownloadedAudio(path=paths[0], video=VideoInfo.from_info_dict(info))


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
    raw_segments, info = model.transcribe(
        str(audio), language=language, task=task, beam_size=beam_size, vad_filter=vad_filter
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments = []
    next_report = 0.25
    for raw in raw_segments:  # lazy: decoding happens while iterating
        text = raw.text.strip()
        if text:
            segments.append(Segment(float(raw.start), float(raw.end), text))
        while duration and next_report < 1 and raw.end >= next_report * duration:
            logger.info("  %d%% of %.0f s transcribed", round(next_report * 100), duration)
            next_report += 0.25
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


def _caption_rank(track: str, language: str, manual: set[str]) -> tuple[bool, int, str]:
    closeness = {language: 0, f"{language}-orig": 1}.get(track, 2)
    return track not in manual, closeness, track


def fetch_captions(url: str, *, language: str = "en") -> Transcript:
    """Use the video's own subtitles instead of running Whisper.

    Manual subtitles are preferred over YouTube's automatic captions, and the exact
    language code over regional variants (``en`` before ``en-GB``). Automatic captions
    in a language other than the spoken one are machine translations; this is recorded
    in ``details["machine_translated"]``. Rolling automatic captions are de-duplicated
    (see :func:`ytfakenews.text.parse_vtt`).
    """
    with tempfile.TemporaryDirectory(prefix="ytfakenews-") as tmp:
        options = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [re.escape(language), f"{re.escape(language)}-.+"],
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s"),
        }
        logger.info("Fetching %r captions for %s", language, url)
        info = _extract_info(url, options, download=True)
        requested = info.get("requested_subtitles") or {}
        tracks = {
            track: Path(sub["filepath"])
            for track, sub in requested.items()
            if sub.get("filepath") and Path(sub["filepath"]).is_file()
        }
        if not tracks:
            raise TranscriptionError(
                f"no {language!r} captions available for {url}; try another --language, "
                "or drop --captions to transcribe the audio with Whisper"
            )
        manual = set(info.get("subtitles") or {})
        track = min(tracks, key=lambda name: _caption_rank(name, language, manual))
        path = tracks[track]
        content = path.read_text(encoding="utf-8")
    if path.suffix == ".vtt":
        segments = parse_vtt(content)
    elif path.suffix == ".srt":
        segments = parse_srt(content)
    else:
        raise TranscriptionError(f"captions came in an unsupported format: {path.suffix}")

    automatic = set(info.get("automatic_captions") or {})
    is_manual = track in manual
    return Transcript(
        segments=tuple(segments),
        source="captions",
        language=track.removesuffix("-orig"),
        details={
            "track": track,
            "kind": "manual" if is_manual else "automatic",
            "machine_translated": not is_manual
            and not track.endswith("-orig")
            and f"{track}-orig" not in automatic,
        },
        video=VideoInfo.from_info_dict(info),
    )


def save_transcript(transcript: Transcript, output_dir: str | Path, stem: str) -> TranscriptFiles:
    """Write ``<stem>.srt``, ``<stem>.txt`` and ``<stem>.json`` (segments plus metadata)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "transcript"
    files = TranscriptFiles(
        srt=output_dir / f"{stem}.srt",
        txt=output_dir / f"{stem}.txt",
        json=output_dir / f"{stem}.json",
    )
    files.srt.write_text(segments_to_srt(transcript.segments), encoding="utf-8")
    files.txt.write_text(transcript.text + "\n", encoding="utf-8")
    write_json(files.json, transcript.to_dict())
    return files


def transcribe_source(
    source: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    captions: bool = False,
    language: str | None = None,
    translate: bool = False,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = "auto",
    compute_type: str = "auto",
    keep_audio: bool = False,
) -> tuple[Transcript, TranscriptFiles]:
    """Transcribe a video URL or a local audio/video file and save the transcript.

    For a URL the audio goes to a temporary directory unless ``keep_audio`` is set, in
    which case it is kept in ``output_dir``. With ``captions=True`` the video's own
    subtitles in ``language`` (default ``en``) are used instead of Whisper.
    """
    output_dir = Path(output_dir)
    whisper_options: dict[str, Any] = {
        "model_size": model_size,
        "language": language,
        "translate": translate,
        "device": device,
        "compute_type": compute_type,
    }
    local = Path(source)
    if local.is_file():
        if captions:
            raise TranscriptionError("--captions needs a video URL, not a local file")
        transcript = transcribe(local, **whisper_options)
        stem = local.stem
    elif re.match(r"^https?://", source):
        if captions:
            transcript = fetch_captions(source, language=language or "en")
        else:
            with tempfile.TemporaryDirectory(prefix="ytfakenews-") as tmp:
                audio = download_audio(source, output_dir if keep_audio else tmp)
                transcript = replace(transcribe(audio.path, **whisper_options), video=audio.video)
        stem = transcript.video.id if transcript.video else "transcript"
    else:
        raise TranscriptionError(f"{source!r} is neither an existing file nor an http(s) URL")
    files = save_transcript(transcript, output_dir, stem)
    logger.info("Wrote %s (%d segments)", files.txt, len(transcript.segments))
    return transcript, files
