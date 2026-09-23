"""Test helpers: synthetic data and fake yt-dlp/faster-whisper layers (no network)."""

from __future__ import annotations

import importlib
import math
import os
import random
import re
import struct
import wave
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def import_or_skip(module: str) -> ModuleType:
    """Skip a test when an optional dependency is missing.

    CI's full-install job sets YTFAKENEWS_REQUIRE_EXTRAS=1 so that a missing
    dependency fails there instead of silently skipping the test.
    """
    if os.environ.get("YTFAKENEWS_REQUIRE_EXTRAS") == "1":
        return importlib.import_module(module)
    return pytest.importorskip(module)


_REAL_WORDS = (
    "officials said the committee report budget percent tuesday senator statement "
    "data vote agency confirmed according spokesperson quarterly"
).split()
_FAKE_WORDS = (
    "shocking truth exposed secret they hide share wake up elite coverup bombshell "
    "insider leaked unbelievable globalist hoax"
).split()
_SHARED_WORDS = "the a of and to in city people new year".split()


def make_news_frame(n_per_label: int = 30, seed: int = 0) -> pd.DataFrame:
    """Deterministic toy corpus whose two classes use different vocabularies."""
    rng = random.Random(seed)
    rows = []
    for label, vocabulary in (("REAL", _REAL_WORDS), ("FAKE", _FAKE_WORDS)):
        words = vocabulary + _SHARED_WORDS
        for index in range(n_per_label):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(40, 80)))
            rows.append({"title": f"{label.title()} story {index}", "text": text, "label": label})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def write_zipped_csv(frame: pd.DataFrame, directory: Path, name: str = "news") -> Path:
    """Write ``frame`` like the bundled dataset: a CSV with an unnamed index, zipped."""
    csv_path = directory / f"{name}.csv"
    frame.to_csv(csv_path)
    zip_path = directory / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)
    csv_path.unlink()
    return zip_path


# --------------------------------------------------------------------- fake ASR layer

VIDEO: dict[str, Any] = {
    "id": "abc123XYZ_-",
    "title": "Evening news",
    "webpage_url": "https://www.youtube.com/watch?v=abc123XYZ_-",
    "channel": "Example channel",
    "duration": 12.0,
    "upload_date": "20240101",
}
VIDEO_URL = VIDEO["webpage_url"]


class FakeYouTube:
    """Replaces ``_extract_info`` (yt-dlp) and ``_load_whisper_model`` (faster-whisper).

    It mimics what the real libraries return and write to disk and records every
    call, so tests can check which options the package passed.
    """

    def __init__(self) -> None:
        auto_captions = (FIXTURES / "youtube_auto.vtt").read_text(encoding="utf-8")
        self.manual_tracks: dict[str, str] = {}
        self.automatic_tracks: dict[str, str] = {"en": auto_captions, "en-orig": auto_captions}
        self.spoken_language = "en"
        self.segments = [
            SimpleNamespace(start=0.0, end=4.0, text=" Good evening, officials said today."),
            SimpleNamespace(start=4.0, end=8.0, text="   "),
            SimpleNamespace(start=8.0, end=12.0, text=" The committee report is out."),
        ]
        self.extract_calls: list[tuple[str, dict[str, Any], bool]] = []
        self.whisper_loads: list[dict[str, Any]] = []
        self.whisper_calls: list[tuple[str, dict[str, Any]]] = []

    def extract_info(
        self, url: str, options: Mapping[str, Any], *, download: bool
    ) -> dict[str, Any]:
        self.extract_calls.append((url, dict(options), download))
        directory = Path(options["outtmpl"]).parent
        info = dict(VIDEO)
        if options.get("writesubtitles"):
            patterns = [re.compile(pattern) for pattern in options["subtitleslangs"]]
            # Like yt-dlp: a manual track replaces the automatic one of the same name.
            available = {**self.automatic_tracks, **self.manual_tracks}
            requested = {}
            for track, content in available.items():
                if any(pattern.fullmatch(track) for pattern in patterns):
                    path = directory / f"{VIDEO['id']}.{track}.vtt"
                    path.write_text(content, encoding="utf-8")
                    requested[track] = {"ext": "vtt", "filepath": str(path)}
            info["requested_subtitles"] = requested
            info["subtitles"] = {track: [{"ext": "vtt"}] for track in self.manual_tracks}
            info["automatic_captions"] = {
                track: [{"ext": "vtt"}] for track in self.automatic_tracks
            }
        elif download:
            path = directory / f"{VIDEO['id']}.webm"
            path.write_bytes(b"not really audio")
            info["requested_downloads"] = [{"filepath": str(path)}]
        return info

    def load_whisper_model(self, model_size: str, *, device: str, compute_type: str) -> Any:
        self.whisper_loads.append(
            {"model_size": model_size, "device": device, "compute_type": compute_type}
        )
        return SimpleNamespace(transcribe=self._transcribe)

    def _transcribe(self, audio: str, **kwargs: Any) -> tuple[Iterator[Any], Any]:
        self.whisper_calls.append((audio, kwargs))
        info = SimpleNamespace(
            language=self.spoken_language, language_probability=0.9871, duration=12.0
        )
        return iter(self.segments), info


def write_tone(path: Path, seconds: float = 1.0, rate: int = 16_000) -> Path:
    """A mono 16-bit sine-wave WAV file."""
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path
