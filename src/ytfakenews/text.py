"""Transcript text utilities: SRT/WebVTT parsing, cleanup and word-window chunking."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Segment",
    "chunk_spans",
    "chunk_words",
    "clean_text",
    "format_timestamp",
    "parse_srt",
    "parse_timestamp",
    "parse_vtt",
    "read_transcript",
    "segments_to_srt",
    "segments_to_text",
]


@dataclass(frozen=True)
class Segment:
    """A timed piece of transcript text; ``start`` and ``end`` are in seconds."""

    start: float
    end: float
    text: str


# ----------------------------------------------------------------------------- timestamps

_TIMESTAMP = r"(?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_CUE_TIMING_RE = re.compile(rf"^\s*(?P<start>{_TIMESTAMP})\s*-->\s*(?P<end>{_TIMESTAMP})")


def parse_timestamp(value: str) -> float:
    """Convert an SRT (``01:02:03,450``) or WebVTT (``02:03.450``) timestamp to seconds."""
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid timestamp: {value!r}")
    try:
        hours = int(parts[0]) if len(parts) == 3 else 0
        minutes = int(parts[-2])
        seconds = float(parts[-1])
    except ValueError:
        raise ValueError(f"invalid timestamp: {value!r}") from None
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float, *, decimal_marker: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT); pass ``decimal_marker="."`` for WebVTT."""
    if seconds < 0:
        raise ValueError(f"timestamp must be non-negative, got {seconds}")
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


# ------------------------------------------------------------------------- SRT and WebVTT

# HTML-like markup (<i>, <font ...>, <c.color>, <v Speaker>) and WebVTT inline
# timestamps (<00:00:01.250>). Requiring an alphanumeric first character keeps
# comparisons such as "a < b > c" intact.
_TAG_RE = re.compile(r"</?[A-Za-z0-9][^<>\n]*>")
# ASS/SSA override blocks that some SRT files carry, e.g. {\an8}.
_ASS_OVERRIDE_RE = re.compile(r"\{\\[^{}]*\}")


def _clean_cue_line(line: str) -> str:
    line = _ASS_OVERRIDE_RE.sub("", line)
    line = _TAG_RE.sub("", line)
    return " ".join(html.unescape(line).split())


def _iter_cues(
    content: str, *, ends_cue: Callable[[str], bool]
) -> Iterator[tuple[float, float, list[str]]]:
    """Yield ``(start, end, payload_lines)`` for every cue in an SRT or WebVTT document.

    A cue starts at a timing line (``start --> end``) and runs until a line for which
    ``ends_cue`` is true. Anything outside cues (headers, cue numbers, NOTE and STYLE
    blocks) is skipped.
    """
    lines = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = end = 0.0
    payload: list[str] | None = None
    for line in lines:
        timing = _CUE_TIMING_RE.match(line)
        if timing:
            if payload is not None:
                # Malformed file without a blank line between cues: the line just
                # before this timing line is the next cue's number, not text.
                if payload and payload[-1].strip().isdigit():
                    payload.pop()
                yield start, end, payload
            start = parse_timestamp(timing["start"])
            end = parse_timestamp(timing["end"])
            payload = []
        elif payload is not None:
            if ends_cue(line):
                yield start, end, payload
                payload = None
            else:
                payload.append(line)
    if payload is not None:
        yield start, end, payload


def parse_srt(content: str) -> list[Segment]:
    """Parse SubRip (``.srt``) content into segments, dropping markup and empty cues."""
    segments = []
    for start, end, payload in _iter_cues(content, ends_cue=lambda line: not line.strip()):
        text = " ".join(filter(None, map(_clean_cue_line, payload)))
        if text:
            segments.append(Segment(start, end, text))
    return segments


def parse_vtt(content: str, *, dedupe: bool = True) -> list[Segment]:
    """Parse WebVTT (``.vtt``) content into segments.

    YouTube's automatic captions are "rolling": every cue repeats the line that is
    already on screen and adds the words being spoken, and short transitional cues
    repeat the previous text verbatim. With ``dedupe=True`` (the default) a line that
    was already shown by the previous cue is dropped, and a line that merely extends
    a line of the previous cue contributes only its new words, so each spoken word
    appears once. The trade-off is that a line repeated verbatim in two consecutive
    cues of a hand-made subtitle file is kept only once.
    """
    cues = []
    # Only a truly empty line ends a WebVTT cue: YouTube's cues contain lines that
    # hold a single space, which are part of the payload.
    for start, end, payload in _iter_cues(content, ends_cue=lambda line: line == ""):
        lines = [cleaned for cleaned in map(_clean_cue_line, payload) if cleaned]
        cues.append((start, end, lines))
    if not dedupe:
        return [Segment(start, end, " ".join(lines)) for start, end, lines in cues if lines]
    return _dedupe_rolling(cues)


def _dedupe_rolling(cues: Iterable[tuple[float, float, list[str]]]) -> list[Segment]:
    segments = []
    previous: list[str] = []
    for start, end, lines in cues:
        fresh = []
        for line in lines:
            if line in previous:
                continue
            # A line that grows word by word: keep only the words not shown before.
            shown = max((p for p in previous if line.startswith(p + " ")), key=len, default="")
            new_words = line[len(shown) :].strip()
            if new_words:
                fresh.append(new_words)
        previous = lines
        if fresh:
            segments.append(Segment(start, end, " ".join(fresh)))
    return segments


def segments_to_srt(segments: Iterable[Segment]) -> str:
    """Render segments as SubRip text."""
    blocks = [
        f"{index}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text}\n"
        for index, seg in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def segments_to_text(segments: Iterable[Segment]) -> str:
    """Render segments as plain text, one segment per line."""
    return "\n".join(seg.text for seg in segments)


def read_transcript(path: str | Path) -> str:
    """Read a ``.srt``, ``.vtt`` or plain-text transcript and return its text."""
    path = Path(path)
    content = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return segments_to_text(parse_srt(content))
    if suffix == ".vtt":
        return segments_to_text(parse_vtt(content))
    return content


# ------------------------------------------------------------------------------- cleanup

_TIMING_RANGE_RE = re.compile(rf"{_TIMESTAMP}\s*-->\s*{_TIMESTAMP}[^\n]*")
# Only full subtitle timestamps with milliseconds, so spoken times ("at 10:30") survive.
_SUBTITLE_TIMESTAMP_RE = re.compile(r"\b(?:\d+:)?\d{1,2}:\d{2}[.,]\d{3}\b")
# Caption annotations for non-speech audio: [Music], [Applause], [BLANK_AUDIO], ...
_ANNOTATION_RE = re.compile(r"\[[^\[\]\n]*\]")
_MUSIC_NOTES_RE = re.compile("[\u266a\u266b\u266c]")  # music notes
_SPEAKER_CHANGE_RE = re.compile(r">>+")  # YouTube marks a new speaker with ">>"


def clean_text(text: str) -> str:
    """Normalise transcript text before classification.

    Removes subtitle timing lines and timestamps, markup, bracketed annotations such as
    ``[Music]`` or ``[Applause]``, music notes and ``>>`` speaker-change markers, then
    collapses all whitespace (including newlines) to single spaces.
    """
    text = html.unescape(text)
    text = _TIMING_RANGE_RE.sub(" ", text)
    text = _SUBTITLE_TIMESTAMP_RE.sub(" ", text)
    text = _TAG_RE.sub("", text)
    text = _ANNOTATION_RE.sub(" ", text)
    text = _MUSIC_NOTES_RE.sub(" ", text)
    text = _SPEAKER_CHANGE_RE.sub(" ", text)
    return " ".join(text.split())


# ------------------------------------------------------------------------------ chunking


def chunk_spans(n_words: int, *, max_words: int = 300, overlap: int = 50) -> list[tuple[int, int]]:
    """Return ``(start, end)`` word offsets that cover ``n_words`` words.

    A text of at most ``max_words`` words is a single window. Longer texts are covered
    by the smallest number of windows of exactly ``max_words`` words such that
    neighbouring windows share at least ``overlap`` words; the windows are spaced
    evenly, the first starts at word 0 and the last ends at the final word. Unlike a
    fixed stride this never produces a short tail window that is mostly overlap.
    """
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")
    if not 0 <= overlap < max_words:
        raise ValueError(f"overlap must satisfy 0 <= overlap < max_words, got {overlap}")
    if n_words <= 0:
        return []
    if n_words <= max_words:
        return [(0, n_words)]
    stride = max_words - overlap
    n_chunks = math.ceil((n_words - overlap) / stride)
    last_start = n_words - max_words
    starts = [i * last_start // (n_chunks - 1) for i in range(n_chunks)]
    return [(start, start + max_words) for start in starts]


def chunk_words(text: str, *, max_words: int = 300, overlap: int = 50) -> list[str]:
    """Split ``text`` into overlapping word windows (see :func:`chunk_spans`)."""
    words = text.split()
    spans = chunk_spans(len(words), max_words=max_words, overlap=overlap)
    return [" ".join(words[start:end]) for start, end in spans]
