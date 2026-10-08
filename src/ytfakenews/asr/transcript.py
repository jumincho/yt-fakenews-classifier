"""What a transcription returns, and the files saved for it."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ytfakenews.artifacts import write_json
from ytfakenews.text import Segment, segments_to_srt, segments_to_text

__all__ = ["Transcript", "TranscriptFiles", "VideoInfo", "save_transcript"]


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
        """JSON-serialisable representation, as saved in ``<stem>.json``."""
        return {
            "source": self.source,
            "language": self.language,
            "details": self.details,
            "video": asdict(self.video) if self.video else None,
            "segments": [asdict(segment) for segment in self.segments],
        }


@dataclass(frozen=True)
class TranscriptFiles:
    """Paths written by :func:`save_transcript`."""

    srt: Path
    txt: Path
    json: Path


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
