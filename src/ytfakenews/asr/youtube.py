"""Download a video's audio or fetch its captions with yt-dlp.

:func:`_extract_info` is the only function in the package that contacts a video site,
so tests replace it. yt-dlp supports many sites besides YouTube; the caption handling
follows YouTube's conventions (``-orig`` tracks, machine-translated automatic captions).
Requires the ``asr`` extra.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ytfakenews._optional import require
from ytfakenews.asr.transcript import Transcript, VideoInfo
from ytfakenews.errors import TranscriptionError
from ytfakenews.text import parse_srt, parse_vtt

__all__ = ["DownloadedAudio", "download_audio", "fetch_captions"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedAudio:
    """An audio file written by :func:`download_audio` and the video it came from."""

    path: Path
    video: VideoInfo


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
    """Run yt-dlp on one URL; the only function in the package that touches a video site."""
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
